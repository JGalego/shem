//! A model server that outlives `shem run`, so that the second run of a model
//! does not load its weights again.
//!
//! A plain run of a model image asks the user's own server, at
//! [`neural::default_socket`], to reply: the server keeps the model loaded
//! for a while after each run. If none is running, `shem run` starts one,
//! which exits when nothing has used it for a while. `SHEM_DAEMON=0` turns it
//! off, and a run that needs the driver's own process (a receipt, a
//! conversation, a terminal's prompt) never uses it.

use crate::sys::net::UnixStream;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{IsTerminal, Read, Write};
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use serde_json::json;

use crate::{image, neural};

const GENERATE: u8 = 14;
const OK: u8 = 0;
const CHUNK: u8 = 3;
/// Seconds a model stays loaded after its last run, and a server stays up
/// after its last connection.
const KEEP_WARM: &str = "600";
const IDLE_EXIT: &str = "900";
/// How long to wait for a server just started: the first run may have uv
/// download PyTorch.
const START_WAIT: Duration = Duration::from_secs(900);

/// Whether the server is on, which `SHEM_DAEMON=0` (or `off`, `no`, `false`) turns off.
pub fn enabled() -> bool {
    // The server is reached over a Unix socket, which a run on Windows does not have
    cfg!(unix) && on(std::env::var("SHEM_DAEMON").ok().as_deref())
}

fn on(setting: Option<&str>) -> bool {
    !matches!(setting, Some("0" | "off" | "no" | "false"))
}

/// The user's server, started if there is none. None when it cannot be
/// reached: someone else's socket is there, or the server would not start.
fn connect(log: &Path) -> Option<UnixStream> {
    let socket = neural::default_socket();
    match neural::connect_stream(&socket) {
        Ok(stream) => return Some(stream),
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return None,
        Err(_) => {}
    }
    // One run at a time starts it: the others wait, then find it listening
    let lock = socket.with_extension("lock");
    if let Some(dir) = lock.parent() {
        std::fs::create_dir_all(dir).ok()?;
    }
    let _lock = crate::sys::lock_file(&lock).ok()?;
    if let Ok(stream) = neural::connect_stream(&socket) {
        return Some(stream);
    }
    eprintln!(
        "shem: starting the model server, which keeps models loaded for {} minutes (SHEM_DAEMON=0 runs without it)",
        KEEP_WARM.parse::<u32>().unwrap_or(0) / 60
    );
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut options, libc::O_NOFOLLOW);
    let log = options.open(log).ok()?;
    let mut server = image::model::python("shem_neural.server");
    server
        .args(["--keep-warm", KEEP_WARM, "--idle-exit", IDLE_EXIT])
        .stdin(Stdio::null())
        .stdout(log.try_clone().ok()?)
        .stderr(log);
    // Its own group: a Ctrl-C at this terminal is not for it
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut server, 0);
    let mut child = server.spawn().ok()?;
    let started = Instant::now();
    loop {
        if let Ok(stream) = neural::connect_stream(&socket) {
            return Some(stream);
        }
        // It ended before it listened: the run will do without it, and say why
        if child.try_wait().ok()?.is_some() || started.elapsed() > START_WAIT {
            return None;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Reply to a run of the driver with the options `entrypoint` has and the
/// prompt in `args`, on the model in `root`, through the user's server. None
/// if the run is not one the server takes, or it could not be reached, and
/// the driver should run itself; otherwise its exit status.
pub fn run(
    entrypoint: &[String],
    host: &[OsString],
    args: &[String],
    root: &Path,
    session: Option<&Path>,
    log: &Path,
) -> Option<anyhow::Result<u8>> {
    if !enabled() || !host.is_empty() {
        return None;
    }
    // A prompt from stdin is read here; a terminal with none is a conversation
    let stdin = if args.is_empty() {
        let mut stdin = std::io::stdin();
        if stdin.is_terminal() {
            return None;
        }
        let mut text = String::new();
        stdin.read_to_string(&mut text).ok()?;
        Some(text)
    } else {
        None
    };
    let mut stream = connect(log)?;
    let (_, options) = entrypoint.split_first()?;
    let mut argv: Vec<&str> = options.iter().map(String::as_str).collect();
    argv.push("--");
    argv.extend(args.iter().map(String::as_str));
    let mut request = json!({ "model": root, "argv": argv });
    if let Some(session) = session {
        request["session"] = json!(session);
    }
    if let Some(stdin) = stdin {
        request["stdin"] = json!(stdin);
    }
    Some(reply(&mut stream, &request))
}

/// Send the request and print the streamed reply as it comes.
fn reply(stream: &mut UnixStream, request: &serde_json::Value) -> anyhow::Result<u8> {
    let payload = [&[GENERATE][..], &serde_json::to_vec(request)?].concat();
    stream.write_all(&[&(payload.len() as u32).to_le_bytes()[..], &payload].concat())?;
    let mut out = std::io::stdout().lock();
    loop {
        let mut header = [0u8; 4];
        stream.read_exact(&mut header)?;
        let mut frame = vec![0u8; u32::from_le_bytes(header) as usize];
        stream.read_exact(&mut frame)?;
        match frame.split_first() {
            Some((&CHUNK, data)) => {
                out.write_all(data)?;
                out.flush()?;
            }
            Some((&OK, _)) => return Ok(0),
            Some((_, message)) => {
                eprintln!("shem: {}", String::from_utf8_lossy(message));
                return Ok(1);
            }
            None => anyhow::bail!("the model server sent an empty frame"),
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A server that answers as `shem_neural.server` does, with `frames`.
    fn answers(frames: Vec<Vec<u8>>) -> UnixStream {
        let (client, mut server) = UnixStream::pair().unwrap();
        std::thread::spawn(move || {
            let mut header = [0u8; 4];
            server.read_exact(&mut header).unwrap();
            let mut payload = vec![0u8; u32::from_le_bytes(header) as usize];
            server.read_exact(&mut payload).unwrap();
            assert_eq!(payload[0], GENERATE);
            let request: serde_json::Value = serde_json::from_slice(&payload[1..]).unwrap();
            assert_eq!(request["argv"], json!(["--max-tokens", "4", "--", "Hi"]));
            for frame in frames {
                server
                    .write_all(&[&(frame.len() as u32).to_le_bytes()[..], &frame].concat())
                    .unwrap();
            }
        });
        client
    }

    fn request() -> serde_json::Value {
        json!({"model": "/m", "argv": ["--max-tokens", "4", "--", "Hi"]})
    }

    #[test]
    fn a_reply_ends_with_status_zero() {
        let mut stream = answers(vec![vec![CHUNK, b'o', b'k'], vec![OK]]);
        assert_eq!(reply(&mut stream, &request()).unwrap(), 0);
    }

    #[test]
    fn an_error_is_status_one() {
        let mut stream = answers(vec![vec![1, b'b', b'a', b'd']]);
        assert_eq!(reply(&mut stream, &request()).unwrap(), 1);
    }

    #[test]
    fn a_server_that_hangs_up_is_an_error() {
        let mut stream = answers(vec![vec![CHUNK, b'x']]);
        assert!(reply(&mut stream, &request()).is_err());
    }

    #[test]
    fn it_can_be_turned_off() {
        assert!(on(None) && on(Some("1")) && on(Some("")));
        assert!(!on(Some("0")) && !on(Some("off")) && !on(Some("false")));
    }
}
