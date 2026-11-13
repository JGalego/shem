//! The containerd shim v2 protocol: containerd starts this binary, talks to it
//! over ttrpc, and calls the task service below to create, start, wait for,
//! kill and delete the container. The shim reports what happens to the task
//! as events, which is how Docker learns that a container has exited.
//!
//! In a model container, `docker exec` runs exec processes, each a turn of the
//! container's live session, and `docker commit`'s pause saves that session
//! into the container's filesystem (see `session`). A program container's
//! pause freezes the program there, for the committed image to resume.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use containerd_shim::event::Event;
use containerd_shim::protos::events::task::{
    TaskCreate, TaskDelete, TaskExecAdded, TaskExecStarted, TaskExit, TaskPaused, TaskResumed,
    TaskStart,
};
use containerd_shim::protos::protobuf::well_known_types::timestamp::Timestamp;
use containerd_shim::protos::protobuf::{EnumOrUnknown, MessageField};
use containerd_shim::protos::ttrpc::{self, context::Context};
use containerd_shim::synchronous::publisher::RemotePublisher;
use containerd_shim::{
    self as shim, Config, DeleteResponse, ExitSignal, Flags, StartOpts, TtrpcContext, TtrpcResult,
    api,
};
use log::{info, warn};
use shem::linux::Stdio;

use crate::bundle::{self, Bundle, Control, Wakeable};
use crate::console::{Console, Resizer};
use crate::holder::Holder;

const SIGKILL: u32 = 9;
/// How long a pause waits for the program to stop between slices.
const PAUSE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone)]
pub struct Service {
    exit: Arc<ExitSignal>,
    task: Arc<(Mutex<Option<Task>>, Condvar)>,
    /// The containerd namespace, which events are published in.
    namespace: String,
    events: Option<Arc<RemotePublisher>>,
}

struct Task {
    id: String,
    bundle: PathBuf,
    stdin: String,
    stdout: String,
    stderr: String,
    status: api::Status,
    exited: Option<(u32, Timestamp)>,
    control: Arc<Control>,
    /// A program's terminal, when it has one, to set its window size.
    terminal: Option<Resizer>,
    /// The task's PID: its namespace holder's, or the shim's when there is none.
    pid: u32,
    holder: Option<Holder>,
    execs: HashMap<String, ExecProcess>,
}

/// A process `docker exec` added to the container: in a model container, a
/// turn of its live session.
struct ExecProcess {
    args: Vec<String>,
    stdout: String,
    stderr: String,
    status: api::Status,
    exited: Option<(u32, Timestamp)>,
}

/// Append a line to `/run/shem/shim.log` if that directory exists: an opt-in
/// trace of every request, for when containerd's own logs are out of reach.
fn trace(message: impl std::fmt::Display) {
    let dir = std::path::Path::new("/run/shem");
    if !dir.is_dir() {
        return;
    }
    if let Ok(mut log) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("shim.log"))
    {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        let _ = writeln!(log, "{now:.3} [{}] {message}", std::process::id());
    }
}

fn error(message: impl std::fmt::Display) -> ttrpc::Error {
    ttrpc::Error::Others(message.to_string())
}

fn pid() -> u32 {
    std::process::id()
}

impl shim::Shim for Service {
    type T = Service;

    fn new(_runtime_id: &str, args: &Flags, _config: &mut Config) -> Self {
        Self {
            exit: Arc::new(ExitSignal::default()),
            task: Arc::new((Mutex::new(None), Condvar::new())),
            namespace: args.namespace.clone(),
            events: None,
        }
    }

    fn start_shim(&mut self, opts: StartOpts) -> Result<String, shim::Error> {
        trace("start_shim");
        let grouping = opts.id.clone();
        let (_pid, address) = shim::spawn(opts, &grouping, Vec::new())?;
        trace(format!("start_shim: serving at {address}"));
        Ok(address)
    }

    fn delete_shim(&mut self) -> Result<DeleteResponse, shim::Error> {
        Ok(DeleteResponse::new())
    }

    fn wait(&mut self) {
        self.exit.wait();
    }

    fn create_task_service(&self, publisher: RemotePublisher) -> Self::T {
        Self {
            events: Some(Arc::new(publisher)),
            ..self.clone()
        }
    }
}

impl Service {
    /// Start an exec process: one turn of a model container's live session,
    /// streamed to the exec's stdout. It ends when the reply does.
    fn start_exec(&self, exec_id: String) -> TtrpcResult<api::StartResponse> {
        let (container_id, bundle_dir, task_pid, args, stdout, stderr) =
            self.with_task(|task| {
                let exec = (task.execs.get_mut(&exec_id))
                    .ok_or_else(|| error(format!("no exec {exec_id}")))?;
                exec.status = api::Status::RUNNING;
                let streams = (exec.args.clone(), exec.stdout.clone(), exec.stderr.clone());
                Ok::<_, ttrpc::Error>((
                    task.id.clone(),
                    task.bundle.clone(),
                    task.pid,
                    streams.0,
                    streams.1,
                    streams.2,
                ))
            })??;
        self.publish(TaskExecStarted {
            container_id: container_id.clone(),
            exec_id: exec_id.clone(),
            pid: task_pid,
            ..Default::default()
        });
        let service = self.clone();
        std::thread::spawn(move || {
            let open = |path: &str| -> Box<dyn Write + Send> {
                match OpenOptions::new().write(true).open(path) {
                    Ok(file) if !path.is_empty() => Box::new(file),
                    _ => Box::new(std::io::sink()),
                }
            };
            let mut out = open(&stdout);
            let status = match Bundle::load(&bundle_dir)
                .and_then(|bundle| crate::session::exec(&bundle, &args, &mut out))
            {
                Ok(()) => 0,
                Err(err) => {
                    let _ = writeln!(open(&stderr), "shem: {err:#}");
                    1
                }
            };
            drop(out);
            let at = Timestamp::now();
            let (lock, exited) = &*service.task;
            if let Some(task) = lock.lock().unwrap().as_mut()
                && let Some(exec) = task.execs.get_mut(&exec_id)
            {
                exec.status = api::Status::STOPPED;
                exec.exited = Some((status, at.clone()));
            }
            exited.notify_all();
            service.publish(TaskExit {
                container_id,
                id: exec_id,
                pid: task_pid,
                exit_status: status,
                exited_at: MessageField::some(at),
                ..Default::default()
            });
        });
        Ok(api::StartResponse {
            pid: task_pid,
            ..Default::default()
        })
    }

    /// Tell containerd what happened to the task, as runc's shim does.
    fn publish(&self, event: impl Event + 'static) {
        let Some(events) = &self.events else { return };
        let topic = event.topic();
        if let Err(err) =
            events.publish(Context::default(), &topic, &self.namespace, Box::new(event))
        {
            warn!("publishing {topic}: {err}");
        }
    }

    fn with_task<R>(&self, f: impl FnOnce(&mut Task) -> R) -> TtrpcResult<R> {
        let mut task = self.task.0.lock().unwrap();
        let task = task
            .as_mut()
            .ok_or_else(|| error("no task has been created"))?;
        Ok(f(task))
    }
}

/// The container's standard streams, from the FIFOs containerd made for it.
/// With `wake`, reads of stdin stop waiting when the program is to stop.
fn stdio(
    stdin: &str,
    stdout: &str,
    stderr: &str,
    wake: Option<Arc<Control>>,
) -> std::io::Result<Stdio> {
    let input: Box<dyn Read + Send> = match (stdin.is_empty(), wake) {
        (true, _) => Box::new(std::io::empty()),
        (false, None) => Box::new(open_fifo_for_reading(stdin)?),
        (false, Some(control)) => Box::new(Wakeable::new(open_fifo_for_reading(stdin)?, control)),
    };
    let output = |name: &'static str, path: &str| -> std::io::Result<Box<dyn Write + Send>> {
        if path.is_empty() {
            Ok(Box::new(std::io::sink()))
        } else {
            let file = OpenOptions::new().write(true).open(path)?;
            trace(format!("{name}: opened"));
            Ok(Box::new(Traced {
                name,
                inner: file,
                written: 0,
            }))
        }
    };
    Ok(Stdio {
        stdin: input,
        stdout: output("stdout", stdout)?,
        stderr: output("stderr", stderr)?,
    })
}

/// A container's output stream that traces, as it closes, how much went to
/// it: a reader that saw less lost it after the shim.
struct Traced {
    name: &'static str,
    inner: File,
    written: usize,
}

impl Write for Traced {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(data)?;
        self.written += n;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl Drop for Traced {
    fn drop(&mut self) {
        trace(format!(
            "{}: closing after {} bytes",
            self.name, self.written
        ));
    }
}

/// Open a FIFO to read from without waiting for a writer, as runc does: with
/// no writer, reads then see end-of-file instead of hanging the container.
fn open_fifo_for_reading(path: &str) -> std::io::Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(path)?;
    // Reads should block for data once the FIFO is open.
    let fd = file.as_raw_fd();
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_NONBLOCK);
    }
    Ok(file)
}

impl shim::Task for Service {
    fn create(
        &self,
        _ctx: &TtrpcContext,
        req: api::CreateTaskRequest,
    ) -> TtrpcResult<api::CreateTaskResponse> {
        trace("create");
        info!("create {} in {}", req.id, req.bundle);
        let bundle = PathBuf::from(&req.bundle);
        let rootfs = bundle.join("rootfs");
        std::fs::create_dir_all(&rootfs).map_err(error)?;
        for mount in &req.rootfs {
            shim::mount::mount_rootfs(
                Some(&mount.type_),
                Some(&mount.source),
                &mount.options,
                &rootfs,
            )
            .map_err(error)?;
        }
        trace(format!("create: mounted {} layers", req.rootfs.len()));
        let holder = Holder::spawn(&bundle).map_err(error)?;
        let task_pid = holder.as_ref().map_or_else(pid, Holder::pid);
        trace(format!("create: the task's PID is {task_pid}"));
        self.publish(TaskCreate {
            container_id: req.id.clone(),
            bundle: req.bundle.clone(),
            pid: task_pid,
            ..Default::default()
        });
        trace("create: published TaskCreate");
        *self.task.0.lock().unwrap() = Some(Task {
            id: req.id,
            bundle,
            stdin: req.stdin,
            stdout: req.stdout,
            stderr: req.stderr,
            status: api::Status::CREATED,
            exited: None,
            control: Arc::new(Control::default()),
            terminal: None,
            pid: task_pid,
            holder,
            execs: HashMap::new(),
        });
        Ok(api::CreateTaskResponse {
            pid: task_pid,
            ..Default::default()
        })
    }

    fn start(
        &self,
        _ctx: &TtrpcContext,
        req: api::StartRequest,
    ) -> TtrpcResult<api::StartResponse> {
        trace("start");
        if !req.exec_id.is_empty() {
            return self.start_exec(req.exec_id);
        }
        let (id, bundle_dir, streams, control, task_pid, holder) = self.with_task(|task| {
            task.status = api::Status::RUNNING;
            let streams = (task.stdin.clone(), task.stdout.clone(), task.stderr.clone());
            (
                task.id.clone(),
                task.bundle.clone(),
                streams,
                task.control.clone(),
                task.pid,
                task.holder.as_ref().map(Holder::pid),
            )
        })?;
        let mut bundle = Bundle::load(&bundle_dir).map_err(error)?;
        bundle.netns = holder.map(|pid| PathBuf::from(format!("/proc/{pid}/ns/net")));
        // A program's input wakes for a kill or a pause, and with a terminal,
        // the program gets one of its own; a session reads and echoes itself
        let program = !bundle.sandbox && !bundle.args[0].starts_with("shem:");
        let console = if program && bundle.terminal {
            let stdin = (!streams.0.is_empty())
                .then(|| open_fifo_for_reading(&streams.0))
                .transpose()
                .map_err(error)?;
            let (console, stdio) =
                Console::open(stdin, &streams.1, control.clone()).map_err(error)?;
            let resizer = console.resizer().map_err(error)?;
            self.with_task(|task| task.terminal = Some(resizer))?;
            Some((console, stdio))
        } else {
            None
        };
        self.publish(TaskStart {
            container_id: id.clone(),
            pid: task_pid,
            ..Default::default()
        });
        let service = self.clone();
        trace("start: published TaskStart, running the container");
        std::thread::spawn(move || {
            let (streams_opened, console) = match console {
                Some((console, stdio)) => (Ok(stdio), Some(console)),
                None => {
                    let wake = program.then(|| control.clone());
                    (stdio(&streams.0, &streams.1, &streams.2, wake), None)
                }
            };
            let terminal = console.as_ref().map(Console::fd);
            let status = match streams_opened {
                Ok(stdio) => {
                    bundle::run(&bundle, stdio, &control, terminal).unwrap_or_else(|err| {
                        // Tell whoever reads the container's stderr, such as ctr.
                        warn!("the container failed: {err:#}");
                        if let Ok(mut stderr) = OpenOptions::new().write(true).open(&streams.2) {
                            let _ = writeln!(stderr, "containerd-shim-shem: {err:#}");
                        }
                        125
                    })
                }
                Err(err) => {
                    warn!("opening the container's streams: {err}");
                    125
                }
            };
            // The program's last output, through its terminal
            if let Some(console) = console {
                console.finish();
            }
            // The container's streams are closed by now, so readers see the end
            let at = Timestamp::now();
            let (lock, exited) = &*service.task;
            if let Some(task) = lock.lock().unwrap().as_mut() {
                task.status = api::Status::STOPPED;
                task.exited = Some((status as u32, at.clone()));
            }
            exited.notify_all();
            trace(format!("exited with {status}"));
            service.publish(TaskExit {
                container_id: id.clone(),
                id,
                pid: task_pid,
                exit_status: status as u32,
                exited_at: MessageField::some(at),
                ..Default::default()
            });
        });
        Ok(api::StartResponse {
            pid: task_pid,
            ..Default::default()
        })
    }

    fn wait(&self, _ctx: &TtrpcContext, req: api::WaitRequest) -> TtrpcResult<api::WaitResponse> {
        trace("wait");
        let (lock, exited) = &*self.task;
        let mut task = lock.lock().unwrap();
        loop {
            let done = match task.as_ref() {
                None => return Err(error("no task has been created")),
                Some(task) if !req.exec_id.is_empty() => {
                    let exec = (task.execs.get(&req.exec_id))
                        .ok_or_else(|| error(format!("no exec {}", req.exec_id)))?;
                    exec.exited.clone()
                }
                Some(task) => task.exited.clone(),
            };
            match done {
                Some((status, at)) => {
                    return Ok(api::WaitResponse {
                        exit_status: status,
                        exited_at: MessageField::some(at),
                        ..Default::default()
                    });
                }
                None => task = exited.wait(task).unwrap(),
            }
        }
    }

    fn state(
        &self,
        _ctx: &TtrpcContext,
        req: api::StateRequest,
    ) -> TtrpcResult<api::StateResponse> {
        trace("state");
        if !req.exec_id.is_empty() {
            return self.with_task(|task| {
                let exec = (task.execs.get(&req.exec_id))
                    .ok_or_else(|| error(format!("no exec {}", req.exec_id)))?;
                Ok::<_, ttrpc::Error>(api::StateResponse {
                    id: req.exec_id.clone(),
                    bundle: task.bundle.display().to_string(),
                    pid: task.pid,
                    status: EnumOrUnknown::new(exec.status),
                    stdout: exec.stdout.clone(),
                    stderr: exec.stderr.clone(),
                    exit_status: exec.exited.as_ref().map_or(0, |(status, _)| *status),
                    exited_at: MessageField::from_option(
                        exec.exited.as_ref().map(|(_, at)| at.clone()),
                    ),
                    exec_id: req.exec_id.clone(),
                    ..Default::default()
                })
            })?;
        }
        self.with_task(|task| api::StateResponse {
            id: task.id.clone(),
            bundle: task.bundle.display().to_string(),
            pid: task.pid,
            status: EnumOrUnknown::new(task.status),
            stdin: task.stdin.clone(),
            stdout: task.stdout.clone(),
            stderr: task.stderr.clone(),
            exit_status: task.exited.as_ref().map_or(0, |(status, _)| *status),
            exited_at: MessageField::from_option(task.exited.as_ref().map(|(_, at)| at.clone())),
            ..Default::default()
        })
    }

    fn kill(&self, _ctx: &TtrpcContext, req: api::KillRequest) -> TtrpcResult<api::Empty> {
        trace("kill");
        let signal = if req.signal == 0 { SIGKILL } else { req.signal };
        if !req.exec_id.is_empty() {
            return Ok(api::Empty::default()); // a turn runs to its end
        }
        let (lock, exited) = &*self.task;
        let mut guard = lock.lock().unwrap();
        let task = guard
            .as_mut()
            .ok_or_else(|| error("no task has been created"))?;
        task.control.kill(signal);
        // A running guest stops at its next slice. One that never started
        // ends now, as a process killed before it ran, and its waiters wake.
        if task.status == api::Status::CREATED {
            let at = Timestamp::now();
            task.status = api::Status::STOPPED;
            task.exited = Some((128 + signal, at.clone()));
            let event = TaskExit {
                container_id: task.id.clone(),
                id: task.id.clone(),
                pid: task.pid,
                exit_status: 128 + signal,
                exited_at: MessageField::some(at),
                ..Default::default()
            };
            drop(guard);
            exited.notify_all();
            self.publish(event);
        }
        Ok(api::Empty::default())
    }

    fn delete(
        &self,
        _ctx: &TtrpcContext,
        req: api::DeleteRequest,
    ) -> TtrpcResult<api::DeleteResponse> {
        trace("delete");
        if !req.exec_id.is_empty() {
            return self.with_task(|task| {
                let exec = task.execs.remove(&req.exec_id);
                let exited = exec.and_then(|e| e.exited);
                api::DeleteResponse {
                    pid: task.pid,
                    exit_status: exited.as_ref().map_or(0, |(status, _)| *status),
                    exited_at: MessageField::from_option(exited.map(|(_, at)| at)),
                    ..Default::default()
                }
            });
        }
        let (id, bundle, exited, task_pid, holder) = self.with_task(|task| {
            let holder = task.holder.take();
            (
                task.id.clone(),
                task.bundle.clone(),
                task.exited.clone(),
                task.pid,
                holder,
            )
        })?;
        drop(holder); // the namespace goes with the task
        let rootfs = bundle.join("rootfs");
        if let Err(err) = shim::mount::umount_recursive(rootfs.to_str(), 0) {
            warn!("unmounting {}: {err}", rootfs.display());
        }
        self.publish(TaskDelete {
            container_id: id.clone(),
            id,
            pid: task_pid,
            exit_status: exited.as_ref().map_or(0, |(status, _)| *status),
            exited_at: MessageField::from_option(exited.clone().map(|(_, at)| at)),
            ..Default::default()
        });
        Ok(api::DeleteResponse {
            pid: task_pid,
            exit_status: exited.as_ref().map_or(0, |(status, _)| *status),
            exited_at: MessageField::from_option(exited.map(|(_, at)| at)),
            ..Default::default()
        })
    }

    fn pids(&self, _ctx: &TtrpcContext, _req: api::PidsRequest) -> TtrpcResult<api::PidsResponse> {
        trace("pids");
        let task_pid = self.with_task(|task| task.pid).unwrap_or_else(|_| pid());
        Ok(api::PidsResponse {
            processes: vec![containerd_shim::protos::types::task::ProcessInfo {
                pid: task_pid,
                ..Default::default()
            }],
            ..Default::default()
        })
    }

    fn connect(
        &self,
        _ctx: &TtrpcContext,
        _req: api::ConnectRequest,
    ) -> TtrpcResult<api::ConnectResponse> {
        trace("connect");
        let task_pid = self.with_task(|task| task.pid).unwrap_or_else(|_| pid());
        Ok(api::ConnectResponse {
            shim_pid: pid(),
            task_pid,
            version: env!("CARGO_PKG_VERSION").into(),
            ..Default::default()
        })
    }

    fn exec(&self, _ctx: &TtrpcContext, req: api::ExecProcessRequest) -> TtrpcResult<api::Empty> {
        trace("exec");
        let spec: serde_json::Value = match req.spec.as_ref() {
            Some(any) => serde_json::from_slice(&any.value).map_err(error)?,
            None => return Err(error("an exec without a process spec")),
        };
        let args: Vec<String> = (spec["args"].as_array().into_iter().flatten())
            .filter_map(|a| a.as_str().map(String::from))
            .collect();
        let container_id = self.with_task(|task| {
            task.execs.insert(
                req.exec_id.clone(),
                ExecProcess {
                    args,
                    stdout: req.stdout.clone(),
                    stderr: req.stderr.clone(),
                    status: api::Status::CREATED,
                    exited: None,
                },
            );
            task.id.clone()
        })?;
        self.publish(TaskExecAdded {
            container_id,
            exec_id: req.exec_id,
            ..Default::default()
        });
        Ok(api::Empty::default())
    }

    fn pause(&self, _ctx: &TtrpcContext, _req: api::PauseRequest) -> TtrpcResult<api::Empty> {
        trace("pause");
        // `docker commit` pauses first: a model container's session, or a
        // program frozen where it paused, goes into its filesystem then, so
        // the commit carries it
        let (id, bundle_dir, control) =
            self.with_task(|task| (task.id.clone(), task.bundle.clone(), task.control.clone()))?;
        let bundle = Bundle::load(&bundle_dir).map_err(error)?;
        if bundle.args[0].starts_with("shem:") {
            crate::session::save(&bundle).map_err(error)?;
        } else if !bundle.sandbox && !control.pause(PAUSE_TIMEOUT) {
            return Err(error(
                "the program did not pause: it may have ended, or wait on its host",
            ));
        }
        self.with_task(|task| task.status = api::Status::PAUSED)?;
        trace("paused");
        self.publish(TaskPaused {
            container_id: id,
            ..Default::default()
        });
        Ok(api::Empty::default())
    }

    fn resume(&self, _ctx: &TtrpcContext, _req: api::ResumeRequest) -> TtrpcResult<api::Empty> {
        trace("resume");
        let id = self.with_task(|task| {
            task.status = api::Status::RUNNING;
            task.control.resume();
            task.id.clone()
        })?;
        self.publish(TaskResumed {
            container_id: id,
            ..Default::default()
        });
        Ok(api::Empty::default())
    }

    fn resize_pty(
        &self,
        _ctx: &TtrpcContext,
        req: api::ResizePtyRequest,
    ) -> TtrpcResult<api::Empty> {
        trace(format!("resize to {}x{}", req.width, req.height));
        // A program's terminal takes the size; a session echoes what is
        // typed, whatever the size
        self.with_task(|task| {
            if let Some(terminal) = &task.terminal {
                let clamp = |n: u32| n.min(u16::MAX as u32) as u16;
                terminal.resize(clamp(req.height), clamp(req.width));
            }
        })?;
        Ok(api::Empty::default())
    }

    fn close_io(&self, _ctx: &TtrpcContext, _req: api::CloseIORequest) -> TtrpcResult<api::Empty> {
        Ok(api::Empty::default()) // stdin ends when its writer closes the FIFO
    }

    fn shutdown(&self, _ctx: &TtrpcContext, _req: api::ShutdownRequest) -> TtrpcResult<api::Empty> {
        trace("shutdown");
        self.exit.signal();
        Ok(api::Empty::default())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::time::Duration;

    use super::*;

    #[test]
    fn stdin_fifos_without_a_writer_read_as_empty_instead_of_hanging() {
        // A session's stdin, and a program's, which a pause or a kill wakes
        for wake in [None, Some(Arc::new(Control::default()))] {
            let dir = tempfile::tempdir().unwrap();
            let fifo = dir.path().join("stdin");
            let path = CString::new(fifo.to_str().unwrap()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
            let (done, opened) = std::sync::mpsc::channel();
            let name = fifo.to_str().unwrap().to_string();
            std::thread::spawn(move || {
                let mut streams = stdio(&name, "", "", wake).unwrap();
                let mut buf = [0u8; 8];
                done.send(streams.stdin.read(&mut buf).unwrap()).unwrap();
            });
            let read = opened.recv_timeout(Duration::from_secs(5));
            assert_eq!(read, Ok(0), "opening or reading the FIFO hung");
        }
    }
}
