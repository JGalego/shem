//! Load OCI runtime bundles and run their processes on shem.
//!
//! `config.json` supplies the [runtime spec]. `dev.shem.vcpu=neural` selects
//! the neural server, with `dev.shem.neural-socket` overriding its socket.
//! Commands starting with `shem:` use the model-session driver instead.
//!
//! [runtime spec]: https://github.com/opencontainers/runtime-spec/blob/main/config.md

use std::collections::VecDeque;
use std::io::Read;
use std::os::fd::{AsFd, AsRawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use anyhow::{Context, bail};
use serde_json::Value;

use crate::state;
use shem::container::{Container, Frozen, FrozenHost, RunError, TERMINAL_SIGNAL};
use shem::image::find_program;
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
pub const DEFAULT_NEURAL_SOCKET: &str = "/run/shem/neural.sock";

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
/// A program `docker exec` adds to a running container: where its streams
/// are, and where to say how it ended.
pub struct ExecRequest {
    pub args: Vec<String>,
    pub env: Vec<String>,
    pub cwd: String,
    pub stdin: String,
    pub stdout: String,
    pub stderr: String,
    /// Told the exit status, or 137 if the container ends first
    pub done: Sender<u8>,
    /// With `docker exec -t`: the program gets a terminal of its own
    pub terminal: bool,
    /// Where the program's terminal's resizer is put, once it has one
    pub resizer: ResizerSlot,
}

impl std::fmt::Debug for ExecRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecRequest")
            .field("args", &self.args)
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

/// Sets the window size of a terminal, from its rows and columns.
pub type Resizer = Box<dyn Fn(u16, u16) + Send>;
/// Where a program's `Resizer` waits for a request to change the window size.
pub type ResizerSlot = Arc<Mutex<Option<Resizer>>>;

/// What closes with a program `docker exec` added, if anything does: its terminal.
type Finish = Option<Box<dyn FnOnce()>>;

/// A program `docker exec` added that is running: its process, who is waiting
/// to hear how it ended, and what ends with it (its terminal).
struct Joined {
    pid: u32,
    done: Sender<u8>,
    finish: Finish,
}

#[derive(Debug, Default)]
pub struct Control {
    /// Programs waiting to join the container, taken between slices.
    spawns: Mutex<VecDeque<ExecRequest>>,
    /// Set once the container has ended: nothing can join it then.
    ended: AtomicBool,
    /// Instructions the container has retired, and processes it has, as of the last slice: its statistics.
    pub steps: AtomicU64,
    pub processes: AtomicU32,
    /// How many programs `docker exec` added are running. Init's read of
    /// its input does not wait while there are, as it holds up the container.
    joined: AtomicU32,
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
        self.interrupted()
            || TERMINAL_SIGNAL.load(Ordering::SeqCst) != 0
            || self.joined.load(Ordering::SeqCst) > 0
            || !self.spawns.lock().unwrap().is_empty()
    }

    /// Whether a program is waiting to join the container, or it is to stop or pause.
    fn joiners_or_stop(&self) -> bool {
        self.interrupted() || !self.spawns.lock().unwrap().is_empty()
    }

    /// Wait until a program asks to join the container, or it is to stop or pause.
    fn idle(&self) {
        while !self.joiners_or_stop() {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Ask for a program to join the running container. False if the
    /// container has ended, which nothing can join: the request is dropped,
    /// and whoever waits on its `done` hears nothing.
    pub fn exec(&self, request: ExecRequest) -> bool {
        let mut spawns = self.spawns.lock().unwrap();
        if self.ended.load(Ordering::SeqCst) {
            return false;
        }
        spawns.push_back(request);
        true
    }

    fn take_execs(&self) -> Vec<ExecRequest> {
        self.spawns.lock().unwrap().drain(..).collect()
    }

    /// The container has ended: the programs that were to join it, and had
    /// not, end with the status of a kill.
    fn end(&self) {
        let mut spawns = self.spawns.lock().unwrap();
        self.ended.store(true, Ordering::SeqCst);
        for request in spawns.drain(..) {
            let _ = request.done.send(137);
        }
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
/// the guest was frozen without them. They are the image's, so they are read
/// as regular files of a bounded size, through no link (see `state`).
fn warm_state(root: &Path) -> anyhow::Result<Option<Frozen>> {
    let Some(host) = state::read(root, "container.json", 64 << 20)? else {
        return Ok(None);
    };
    let host: FrozenHost =
        serde_json::from_slice(&host).context("the warm image's container.json")?;
    let machine = state::read(root, "machine.state", state::MAX_STATE)?
        .context("the warm image has a container.json, and no machine.state")?;
    let machine = MachineState::decode(&machine).context("the warm image's machine.state")?;
    state::remove(root)?;
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
    vcpu: impl Fn(u32) -> anyhow::Result<V>,
) -> anyhow::Result<u8> {
    let mut running = Vec::new();
    let result = run_container(bundle, stdio, control, terminal, slice, &vcpu, &mut running);
    // Whatever joined the container goes with it, and nothing can join now
    control.end();
    for joined in running {
        if let Some(finish) = joined.finish {
            finish();
        }
        let _ = joined.done.send(137);
    }
    result
}

fn run_container<V: Vcpu + Clone>(
    bundle: &Bundle,
    stdio: Stdio,
    control: &Control,
    terminal: Option<i32>,
    slice: u64,
    vcpu: &impl Fn(u32) -> anyhow::Result<V>,
    running: &mut Vec<Joined>,
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
            Err(RunError::Deadlock) if control.woke.swap(false, Ordering::SeqCst) => {
                // Not a busy wait while the programs `docker exec` added take their time
                if !running.is_empty() {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            // Or the ones `docker exec` added are still to run or to end
            Err(RunError::Deadlock) if !running.is_empty() => {
                std::thread::sleep(Duration::from_millis(5));
            }
            // Or all of them wait for what only a client can give, as a
            // server does for the client `docker exec` adds: on Linux that
            // waits as long as it takes, until a program joins or it is stopped
            Err(RunError::Deadlock) => {
                if !control.joiners_or_stop() {
                    log::warn!(
                        "every process waits, on nothing a program running here can give: \
                         waiting for `docker exec` or a stop"
                    );
                    control.idle();
                }
            }
            Err(err) => return Err(err.into()),
        }
        join_programs(&mut container, bundle, control, vcpu, running);
        if control.interrupted() {
            match control.kill.load(Ordering::SeqCst) {
                0 => pause(&container, bundle, control)?,
                signal => return Ok(128u32.saturating_add(signal) as u8),
            }
        }
    }
}

/// Start the programs `docker exec` asked for, and say how those that have
/// ended did. The statistics `docker stats` shows are taken here too.
fn join_programs<V: Vcpu + Clone>(
    container: &mut Container<V>,
    bundle: &Bundle,
    control: &Control,
    vcpu: &impl Fn(u32) -> anyhow::Result<V>,
    running: &mut Vec<Joined>,
) {
    for request in control.take_execs() {
        match join(container, bundle, &request, vcpu) {
            Ok((pid, finish)) => running.push(Joined {
                pid,
                done: request.done,
                finish,
            }),
            Err(err) => {
                log::warn!("exec {:?}: {err:#}", request.args);
                if let Ok(mut stderr) = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&request.stderr)
                {
                    use std::io::Write;
                    let _ = writeln!(stderr, "shem: {err:#}");
                }
                let _ = request.done.send(126);
            }
        }
    }
    let mut ended = Vec::new();
    running.retain_mut(|joined| match container.exit_of(joined.pid) {
        Some(status) => {
            ended.push((joined.finish.take(), joined.done.clone(), status));
            false
        }
        None => true,
    });
    for (finish, done, status) in ended {
        // Its last output goes through its terminal before it is said to have ended
        if let Some(finish) = finish {
            finish();
        }
        let _ = done.send(status);
    }
    control.joined.store(running.len() as u32, Ordering::SeqCst);
    control.steps.store(container.steps, Ordering::SeqCst);
    control
        .processes
        .store(container.processes() as u32, Ordering::SeqCst);
}

/// The streams of a program `docker exec` added, from the FIFOs containerd made.
/// Its input is read without waiting, so that a program with none yet parks
/// where the container's own loop can go on; a program that has no streams
/// gets empty ones.
fn exec_stdio(request: &ExecRequest) -> std::io::Result<Stdio> {
    use std::os::unix::fs::OpenOptionsExt;
    let input: Box<dyn Read + Send> = if request.stdin.is_empty() {
        Box::new(std::io::empty())
    } else {
        Box::new(
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&request.stdin)?,
        )
    };
    let output = |path: &str| -> std::io::Result<Box<dyn std::io::Write + Send>> {
        if path.is_empty() {
            return Ok(Box::new(std::io::sink()));
        }
        Ok(Box::new(
            std::fs::OpenOptions::new().write(true).open(path)?,
        ))
    };
    Ok(Stdio {
        stdin: input,
        stdout: output(&request.stdout)?,
        stderr: output(&request.stderr)?,
    })
}

fn join<V: Vcpu + Clone>(
    container: &mut Container<V>,
    bundle: &Bundle,
    request: &ExecRequest,
    vcpu: &impl Fn(u32) -> anyhow::Result<V>,
) -> anyhow::Result<(u32, Finish)> {
    let name = request.args.first().context("no program to run")?;
    let rootfs = RootFs::new(&bundle.root).context("the bundle's root filesystem")?;
    let cwd = if request.cwd.is_empty() {
        &bundle.cwd
    } else {
        &request.cwd
    };
    let env = if request.env.is_empty() {
        &bundle.env
    } else {
        &request.env
    };
    let program = find_program(&rootfs, cwd, env, name)?;
    let elf = std::fs::read(program)?;
    let (stdio, terminal, finish) = if request.terminal {
        exec_terminal(request)?
    } else {
        (exec_stdio(request)?, None, None)
    };
    let pid = container.spawn(
        vcpu(bundle.memory)?,
        &elf,
        &request.args,
        env,
        cwd,
        stdio,
        terminal,
    )?;
    Ok((pid, finish))
}

/// The streams of a program `docker exec -t` added, the terminal's descriptor
/// with where its typed signals are noted, and what closes it.
type ExecTerminal = (Stdio, Option<(i32, Arc<AtomicU32>)>, Finish);

/// The streams of a program `docker exec -t` added: a terminal between the
/// FIFOs containerd made and the program, and what closes it when the
/// program has ended.
#[cfg(target_os = "linux")]
fn exec_terminal(request: &ExecRequest) -> anyhow::Result<ExecTerminal> {
    use crate::console::Console;
    use std::os::unix::fs::OpenOptionsExt;
    let stdin = if request.stdin.is_empty() {
        None
    } else {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&request.stdin)?;
        // Typing is copied on a thread of its own, which waits for it
        // SAFETY: fcntl on a descriptor `file` owns
        unsafe {
            let fd = file.as_raw_fd();
            let flags = libc::fcntl(fd, libc::F_GETFL);
            libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_NONBLOCK);
        }
        Some(file)
    };
    let typed = Arc::new(AtomicU32::new(0));
    let (console, stdio) = Console::open_exec(stdin, &request.stdout, typed.clone())?;
    let resizer = console.resizer()?;
    let signals = typed.clone();
    *request.resizer.lock().unwrap() = Some(Box::new(move |rows, cols| {
        resizer.resize_for(rows, cols, Some(&signals));
    }));
    let fd = console.fd();
    Ok((
        stdio,
        Some((fd, typed)),
        Some(Box::new(|| console.finish())),
    ))
}

#[cfg(not(target_os = "linux"))]
fn exec_terminal(_: &ExecRequest) -> anyhow::Result<ExecTerminal> {
    bail!("a terminal for `docker exec -t` needs Linux")
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
    // The guest may have left links at /.shem: written as new files, not through them
    let frozen = match container.freeze() {
        Ok(frozen) => {
            let host = serde_json::to_vec(&frozen.host)?;
            state::write(&bundle.root, "container.json", &host)?;
            state::write(&bundle.root, "machine.state", &frozen.machine.encode())?;
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
        state::remove(&bundle.root)?;
    }
    Ok(())
}
