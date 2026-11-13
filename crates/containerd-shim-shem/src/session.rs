//! The shim's client for model sessions on `shem_neural.server`.
//!
//! Sessions are keyed by container ID. Stdin lines and `docker exec` add turns;
//! pausing for `docker commit` saves the conversation and block state under
//! `/.shem`. Terminal echo is handled here because there is no guest terminal.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, bail};
use serde_json::json;
use shem::linux::Stdio;

use crate::bundle::{Bundle, DEFAULT_NEURAL_SOCKET, VcpuKind};
use crate::state;

const GENERATE: u8 = 14;
const TURN: u8 = 15;
const SAVE: u8 = 16;
const CLOSE: u8 = 17;
const OK: u8 = 0;
const CHUNK: u8 = 3;

/// A connection to the model server.
pub struct Server {
    stream: UnixStream,
}

impl Server {
    pub fn connect(socket: &Path) -> anyhow::Result<Self> {
        let stream = UnixStream::connect(socket)
            .with_context(|| format!("connecting to the model server at {}", socket.display()))?;
        Ok(Self { stream })
    }

    /// Send `request`, pass each streamed chunk of the reply to `chunk`, and
    /// return the final result.
    pub fn call(
        &mut self,
        op: u8,
        request: &serde_json::Value,
        mut chunk: impl FnMut(&[u8]) -> std::io::Result<()>,
    ) -> anyhow::Result<Vec<u8>> {
        let payload = [&[op][..], &serde_json::to_vec(request)?].concat();
        let length = (payload.len() as u32).to_le_bytes();
        self.stream.write_all(&[&length[..], &payload].concat())?;
        loop {
            let mut header = [0u8; 4];
            self.stream.read_exact(&mut header)?;
            let mut frame = vec![0u8; u32::from_le_bytes(header) as usize];
            self.stream.read_exact(&mut frame)?;
            match frame.split_first() {
                Some((&CHUNK, data)) => chunk(data)?,
                Some((&OK, result)) => return Ok(result.to_vec()),
                Some((_, message)) => bail!("{}", String::from_utf8_lossy(message)),
                None => bail!("the model server sent an empty frame"),
            }
        }
    }
}

/// The model server's socket for a bundle: `dev.shem.neural-socket`, or the default.
pub fn socket(bundle: &Bundle) -> PathBuf {
    match &bundle.vcpu {
        VcpuKind::Neural(socket) => socket.clone(),
        VcpuKind::Reference => PathBuf::from(DEFAULT_NEURAL_SOCKET),
    }
}

/// Output for a client that may be a terminal, which wants "\r\n".
fn writer(
    out: Arc<Mutex<Box<dyn Write + Send>>>,
    terminal: bool,
) -> impl FnMut(&[u8]) -> std::io::Result<()> {
    move |data| {
        let mut out = out.lock().unwrap();
        if terminal {
            let text = String::from_utf8_lossy(data).replace('\n', "\r\n");
            out.write_all(text.as_bytes())?;
        } else {
            out.write_all(data)?;
        }
        out.flush()
    }
}

/// Run a model container's main process. Returns its exit status.
pub fn run(bundle: &Bundle, stdio: Stdio, kill: &AtomicU32) -> anyhow::Result<u8> {
    if bundle.args[0] != "shem:generate" {
        bail!("unknown driver {}: shem has shem:generate", bundle.args[0]);
    }
    // A session kept for docker exec, or an API served, lasts until stopped
    let keep = bundle.args.iter().any(|a| a == "--keep" || a == "--serve");
    let mut server = Server::connect(&socket(bundle))?;
    let out: Arc<Mutex<Box<dyn Write + Send>>> = Arc::new(Mutex::new(stdio.stdout));
    let mut write = writer(out.clone(), bundle.terminal);

    // The session, answering the command line's prompt if it has one
    let mut request = json!({
        "model": bundle.root,
        "argv": &bundle.args[1..],
        "session_id": bundle.id,
    });
    // The image's session, if it has one the model server can read without
    // following links the image put there
    if let Some(saved) = state::dir(&bundle.root)
        && state::has(&bundle.root, "session.json")
        && state::has(&bundle.root, "session.safetensors")
    {
        request["session"] = saved.display().to_string().into();
    }
    // Where `--serve` listens: in the container's network
    if let Some(netns) = &bundle.netns {
        request["netns"] = netns.display().to_string().into();
    }
    server.call(GENERATE, &request, &mut write)?;

    // Then each line of stdin is a turn
    let lines = lines(stdio.stdin, bundle.terminal, out.clone());
    let mut prompted = false;
    let status = loop {
        if bundle.terminal && !prompted {
            write(b"> ")?;
            prompted = true;
        }
        match lines.recv_timeout(Duration::from_millis(100)) {
            Ok(line) => {
                prompted = false;
                if !line.trim().is_empty() {
                    let turn = json!({"session_id": bundle.id, "text": line});
                    server.call(TURN, &turn, &mut write)?;
                }
            }
            Err(RecvTimeoutError::Timeout) => match kill.load(Ordering::SeqCst) {
                0 => continue,
                signal => break 128 + signal,
            },
            Err(RecvTimeoutError::Disconnected) => {
                // With --keep, the session stays open for docker exec
                break loop {
                    match kill.load(Ordering::SeqCst) {
                        0 if keep => std::thread::sleep(Duration::from_millis(100)),
                        0 => break 0,
                        signal => break 128 + signal,
                    }
                };
            }
        }
    };
    server.call(CLOSE, &json!({"session_id": bundle.id}), |_| Ok(()))?;
    Ok(status.min(255) as u8)
}

/// Lines of stdin, read on a thread of their own so that the main loop can
/// notice a kill. On a terminal, typed keys arrive one by one, unechoed: echo
/// them, handle backspace, and end a line on Enter; Ctrl-D or Ctrl-C ends
/// the input.
fn lines(
    mut stdin: Box<dyn Read + Send>,
    terminal: bool,
    echo: Arc<Mutex<Box<dyn Write + Send>>>,
) -> Receiver<String> {
    let (send, receive) = channel();
    std::thread::spawn(move || {
        let mut line: Vec<u8> = Vec::new();
        let mut byte = [0u8; 1];
        let say = |bytes: &[u8]| {
            let mut out = echo.lock().unwrap();
            let _ = out.write_all(bytes);
            let _ = out.flush();
        };
        while let Ok(1) = stdin.read(&mut byte) {
            match byte[0] {
                b'\n' | b'\r' => {
                    if terminal {
                        say(b"\r\n");
                    }
                    let text = String::from_utf8_lossy(&line).into_owned();
                    line.clear();
                    if send.send(text).is_err() {
                        return;
                    }
                }
                0x03 | 0x04 if terminal => break,
                0x7f | 0x08 if terminal => {
                    // Remove one character, continuation bytes and all
                    while let Some(last) = line.pop() {
                        if last & 0xc0 != 0x80 {
                            break;
                        }
                    }
                    say(b"\x08 \x08");
                }
                other => {
                    line.push(other);
                    if terminal {
                        say(&[other]);
                    }
                }
            }
        }
        if !line.is_empty() {
            let _ = send.send(String::from_utf8_lossy(&line).into_owned());
        }
    });
    receive
}

/// One turn of a running container's session, as `docker exec c
/// shem:generate TEXT` gives it: the reply goes to `stdout`.
pub fn exec(bundle: &Bundle, args: &[String], stdout: &mut dyn Write) -> anyhow::Result<()> {
    if args.first().map(String::as_str) != Some("shem:generate") {
        bail!("docker exec runs `shem:generate TEXT` in a model container");
    }
    let mut server = Server::connect(&socket(bundle))?;
    let turn = json!({"session_id": bundle.id, "argv": &args[1..]});
    server.call(TURN, &turn, |data| {
        stdout.write_all(data)?;
        stdout.flush()
    })?;
    Ok(())
}

/// Save a running container's session into its `/.shem`, as `docker commit`
/// needs before it snapshots the container's filesystem.
pub fn save(bundle: &Bundle) -> anyhow::Result<()> {
    let mut server = Server::connect(&socket(bundle))?;
    let files = server.call(SAVE, &json!({"session_id": bundle.id}), |_| Ok(()))?;
    let mut rest = &files[..];
    while !rest.is_empty() {
        let (name, after) = field(rest)?;
        let (data, after) = field(after)?;
        let name = std::str::from_utf8(name)?;
        // New files in a /.shem that is a directory, whatever links the image left there
        state::write(&bundle.root, name, data)?;
        rest = after;
    }
    Ok(())
}

/// A u32 length, then that many bytes.
fn field(data: &[u8]) -> anyhow::Result<(&[u8], &[u8])> {
    let length =
        u32::from_le_bytes(data.get(..4).context("a truncated field")?.try_into()?) as usize;
    let rest = &data[4..];
    if rest.len() < length {
        bail!("a truncated field");
    }
    Ok(rest.split_at(length))
}
