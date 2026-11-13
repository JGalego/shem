//! Load OCI runtime bundles and run their processes on shem.
//!
//! `config.json` supplies the [runtime spec]. `dev.shem.vcpu=neural` selects
//! the neural server, with `dev.shem.neural-socket` overriding its socket.
//! Commands starting with `shem:` use the model-session driver instead.
//!
//! [runtime spec]: https://github.com/opencontainers/runtime-spec/blob/main/config.md

use std::io::Read;
use std::os::fd::{AsFd, AsRawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use anyhow::{Context, bail};
use serde_json::Value;
use shem::container::{Container, Frozen, FrozenHost, RunError, TERMINAL_SIGNAL};
use shem::image::{WARM_DIR, find_program};
use shem::linux::{RootFs, Stdio};
use shem::neural::NeuralVcpu;
use shem::reference::ReferenceVcpu;
use shem::vcpu::{MachineState, Vcpu};

/// Guest memory when the bundle sets no memory limit.
pub const DEFAULT_MEMORY: u32 = 64 << 20;
/// Instructions between checks for a kill request: about a hundredth of a
/// second on the reference vCPU, and a second on a neural one.
const SLICE: u64 = 1 << 20;
const NEURAL_SLICE: u64 = 1 << 10;
pub const DEFAULT_NEURAL_SOCKET: &str = "/tmp/shem-neural.sock";

/// Which vCPU runs the container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VcpuKind {
    Reference,
    /// A neural vCPU, on the server at this socket.
    Neural(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bundle {
    pub args: Vec<String>,
    pub env: Vec<String>,
    pub cwd: String,
    pub root: PathBuf,
    pub memory: u32,
    pub vcpu: VcpuKind,
    /// The container's ID: the bundle directory's name, as containerd makes it.
    pub id: String,
    /// Whether the process has a terminal (`docker run -t`).
    pub terminal: bool,
    /// Whether this is a Kubernetes pod's sandbox, which only holds the pod's
    /// namespaces: its pause program is not a guest.
    pub sandbox: bool,
    /// The container's network namespace, where a model container serves
    /// its API: the shim's holder's, when it has one (see `holder`).
    pub netns: Option<PathBuf>,
}

impl Bundle {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let config: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .context("parsing config.json")?;
        let strings = |value: &Value| -> Vec<String> {
            let items = value.as_array().map(Vec::as_slice).unwrap_or_default();
            items
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        };
        let process = &config["process"];
        let args = strings(&process["args"]);
        if args.is_empty() {
            bail!("the bundle's process has no args");
        }
        let root = config["root"]["path"].as_str().unwrap_or("rootfs");
        let limit = config["linux"]["resources"]["memory"]["limit"].as_i64();
        let memory = match limit {
            Some(limit) if limit > 0 && limit < u32::MAX as i64 => (limit as u32) & !0xfff,
            _ => DEFAULT_MEMORY,
        };
        let annotations = &config["annotations"];
        let vcpu = match annotations["dev.shem.vcpu"].as_str() {
            None | Some("reference") => VcpuKind::Reference,
            Some("neural") => VcpuKind::Neural(PathBuf::from(
                annotations["dev.shem.neural-socket"]
                    .as_str()
                    .unwrap_or(DEFAULT_NEURAL_SOCKET),
            )),
            Some(other) => bail!("unknown dev.shem.vcpu {other:?}: reference or neural"),
        };
        Ok(Self {
            args,
            env: strings(&process["env"]),
            cwd: process["cwd"].as_str().unwrap_or("/").to_string(),
            root: dir.join(root),
            memory,
            vcpu,
            id: (dir.file_name())
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
            terminal: process["terminal"].as_bool().unwrap_or(false),
            sandbox: annotations["io.kubernetes.cri.container-type"] == "sandbox",
            netns: None,
        })
    }
}

/// What the shim asks of a running program, which it does between slices,
/// or while it waits for input: stop with a signal, or pause, as `docker
/// pause` and `docker commit` ask, until it is resumed.
#[derive(Debug, Default)]
pub struct Control {
    /// The signal to stop with, or 0.
    pub kill: AtomicU32,
    pause: Mutex<Pausing>,
    changed: Condvar,
    /// Whether a read of stdin gave up waiting for input to let the program
    /// stop, so that a program blocked on nothing else is not deadlocked.
    woke: AtomicBool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Pausing {
    #[default]
    No,
    /// Asked for; the program has not stopped yet.
    Asked,
    /// Stopped, until it is resumed.
    Paused,
}

impl Control {
    /// Whether the program should stop running for now: to end, or to pause.
    pub fn interrupted(&self) -> bool {
        self.kill.load(Ordering::SeqCst) != 0 || *self.pause.lock().unwrap() == Pausing::Asked
    }

    /// Whether a read of input should stop waiting: the program is to stop,
    /// or a key typed at its terminal sends it a signal.
    fn wakes(&self) -> bool {
        self.interrupted() || TERMINAL_SIGNAL.load(Ordering::SeqCst) != 0
    }

    /// Stop the program with `signal`, paused or not.
    pub fn kill(&self, signal: u32) {
        self.kill.store(signal, Ordering::SeqCst);
        let _state = self.pause.lock().unwrap();
        self.changed.notify_all();
    }

    /// Ask the program to pause, and wait until it has. Gives up, and lets
    /// it go on, after `timeout`: one that has ended, or that waits on its
    /// host for something other than its input, cannot pause. Returns
    /// whether it paused.
    pub fn pause(&self, timeout: Duration) -> bool {
        let mut state = self.pause.lock().unwrap();
        if *state == Pausing::No {
            *state = Pausing::Asked;
        }
        let (mut state, _) = self
            .changed
            .wait_timeout_while(state, timeout, |s| *s == Pausing::Asked)
            .unwrap();
        if *state == Pausing::Asked {
            *state = Pausing::No;
        }
        *state == Pausing::Paused
    }

    /// Let a paused program go on.
    pub fn resume(&self) {
        *self.pause.lock().unwrap() = Pausing::No;
        self.changed.notify_all();
    }

    /// In the program's thread, once it has stopped for a pause: say so, and
    /// wait to be resumed or killed.
    fn paused(&self) {
        let mut state = self.pause.lock().unwrap();
        *state = Pausing::Paused;
        self.changed.notify_all();
        let _state = self
            .changed
            .wait_while(state, |s| {
                *s == Pausing::Paused && self.kill.load(Ordering::SeqCst) == 0
            })
            .unwrap();
    }
}

/// A program's stdin, whose reads, while they wait for input, give up when
/// the program is to stop, or to take a signal from its terminal: the read
/// reports that it would block, the guest is parked on its system call, and
/// it can stop there, or take the signal, to retry the call when it goes on.
pub struct Wakeable<R> {
    input: R,
    control: Arc<Control>,
}

impl<R: Read + AsFd> Wakeable<R> {
    /// Read `input`, which this makes nonblocking, to wait for it here.
    pub fn new(input: R, control: Arc<Control>) -> Self {
        let fd = input.as_fd().as_raw_fd();
        // SAFETY: fcntl on a descriptor `input` owns
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }
        Self { input, control }
    }
}

impl<R: Read + AsFd> Read for Wakeable<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if self.control.wakes() {
                self.control.woke.store(true, Ordering::SeqCst);
                return Err(std::io::ErrorKind::WouldBlock.into());
            }
            // Input, or its end: a FIFO no one ever opened to write reads as
            // ended, though poll does not say so
            match self.input.read(buf) {
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                result => return result,
            }
            let mut watch = libc::pollfd {
                fd: self.input.as_fd().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one pollfd, on the stack. Wait for input a while, then
            // look at the control again.
            unsafe { libc::poll(&mut watch, 1, 100) };
        }
    }
}

/// The frozen container of a warm image (`shem save` puts its state under
/// `/.shem`), if the root filesystem holds one. The files are removed, since
/// the guest was frozen without them.
fn warm_state(root: &Path) -> anyhow::Result<Option<Frozen>> {
    let dir = root.join(WARM_DIR);
    if !dir.join("container.json").is_file() {
        return Ok(None);
    }
    let host: FrozenHost = serde_json::from_slice(&std::fs::read(dir.join("container.json"))?)
        .context("the warm image's container.json")?;
    let machine = MachineState::decode(&std::fs::read(dir.join("machine.state"))?)
        .context("the warm image's machine.state")?;
    std::fs::remove_dir_all(&dir)?;
    Ok(Some(Frozen { machine, host }))
}

/// Run the bundle's process to completion and return its exit status. A warm
/// image resumes where it was frozen, with the memory it was frozen with,
/// ignoring the bundle's arguments. A kill (a signal number) stops it at the
/// next slice boundary, with the status a shell reports for that signal; a
/// pause freezes it there (see `pause`). With `terminal`, a host terminal's
/// descriptor, the program's terminal requests on its standard streams go
/// to that terminal.
pub fn run(
    bundle: &Bundle,
    stdio: Stdio,
    control: &Control,
    terminal: Option<i32>,
) -> anyhow::Result<u8> {
    let kill = &control.kill;
    if bundle.sandbox {
        // A pod's sandbox: the holder keeps its namespaces, and it waits to be stopped
        loop {
            match kill.load(Ordering::SeqCst) {
                0 => std::thread::sleep(std::time::Duration::from_millis(100)),
                signal => return Ok(128u32.saturating_add(signal) as u8),
            }
        }
    }
    if bundle.args[0].starts_with("shem:") {
        return crate::session::run(bundle, stdio, kill);
    }
    match &bundle.vcpu {
        VcpuKind::Reference => run_on(bundle, stdio, control, terminal, SLICE, |mem| {
            Ok(ReferenceVcpu::new(mem))
        }),
        VcpuKind::Neural(socket) => run_on(bundle, stdio, control, terminal, NEURAL_SLICE, |mem| {
            NeuralVcpu::connect(socket, mem).with_context(|| {
                format!(
                    "connecting to the neural vCPU server at {}",
                    socket.display()
                )
            })
        }),
    }
}

fn run_on<V: Vcpu + Clone>(
    bundle: &Bundle,
    stdio: Stdio,
    control: &Control,
    terminal: Option<i32>,
    slice: u64,
    vcpu: impl FnOnce(u32) -> anyhow::Result<V>,
) -> anyhow::Result<u8> {
    let mut container = if let Some(frozen) = warm_state(&bundle.root)? {
        let vcpu = vcpu(frozen.machine.mem.len() as u32)?;
        let mut container = Container::thaw(vcpu, &frozen, stdio)?;
        container.set_root(&bundle.root)?;
        container
    } else {
        let rootfs = RootFs::new(&bundle.root).context("the bundle's root filesystem")?;
        let program = find_program(&rootfs, &bundle.cwd, &bundle.env, &bundle.args[0])?;
        let elf = std::fs::read(program)?;
        let vcpu = vcpu(bundle.memory)?;
        let mut container = Container::from_elf(vcpu, &elf, &bundle.args, &bundle.env, stdio)?;
        container.set_root(&bundle.root)?;
        if bundle.cwd != "/" {
            container.set_cwd(&bundle.cwd).map_err(|errno| {
                anyhow::anyhow!("working directory {}: errno {}", bundle.cwd, errno.0)
            })?;
        }
        container
    };
    container.linux.terminal = terminal;
    loop {
        match container.run(Some(container.steps + slice)) {
            Ok(status) => return Ok(status),
            Err(RunError::StepLimit(_)) => {}
            // Every process waits, one of them on input that stopped waiting
            // so the program could stop; if it need not now, it waits again
            Err(RunError::Deadlock) if control.woke.swap(false, Ordering::SeqCst) => {}
            Err(err) => return Err(err.into()),
        }
        if control.interrupted() {
            match control.kill.load(Ordering::SeqCst) {
                0 => pause(&container, bundle, control)?,
                signal => return Ok(128u32.saturating_add(signal) as u8),
            }
        }
    }
}

/// Pause the program until it is resumed, frozen into `/.shem` in its root
/// filesystem first, where `docker commit` takes it: a container of the
/// committed image resumes the program where it paused. The state goes
/// again when it is resumed. A program that cannot be frozen, as one of
/// several processes, pauses all the same, and a commit holds its files.
fn pause<V: Vcpu + Clone>(
    container: &Container<V>,
    bundle: &Bundle,
    control: &Control,
) -> anyhow::Result<()> {
    let dir = bundle.root.join(WARM_DIR);
    let frozen = match container.freeze() {
        Ok(frozen) => {
            std::fs::create_dir_all(&dir)?;
            let host = serde_json::to_vec(&frozen.host)?;
            std::fs::write(dir.join("container.json"), host)?;
            std::fs::write(dir.join("machine.state"), frozen.machine.encode())?;
            true
        }
        Err(err) => {
            log::warn!(
                "pausing without freezing the program, so a commit holds only its files: {err}"
            );
            false
        }
    };
    control.paused();
    if frozen {
        std::fs::remove_dir_all(&dir)?;
    }
    Ok(())
}
