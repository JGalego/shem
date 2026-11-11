//! The containerd shim v2 protocol: containerd starts this binary, talks to it
//! over ttrpc, and calls the task service below to create, start, wait for,
//! kill and delete the container.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use containerd_shim::protos::protobuf::well_known_types::timestamp::Timestamp;
use containerd_shim::protos::protobuf::{EnumOrUnknown, MessageField};
use containerd_shim::protos::ttrpc;
use containerd_shim::synchronous::publisher::RemotePublisher;
use containerd_shim::{
    self as shim, Config, DeleteResponse, ExitSignal, Flags, StartOpts, TtrpcContext, TtrpcResult,
    api,
};
use log::{info, warn};
use shem::linux::Stdio;

use crate::bundle::{self, Bundle};

const SIGKILL: u32 = 9;

#[derive(Clone)]
pub struct Service {
    exit: Arc<ExitSignal>,
    task: Arc<(Mutex<Option<Task>>, Condvar)>,
}

struct Task {
    id: String,
    bundle: PathBuf,
    stdin: String,
    stdout: String,
    stderr: String,
    status: api::Status,
    exited: Option<(u32, Timestamp)>,
    kill: Arc<AtomicU32>,
}

fn error(message: impl std::fmt::Display) -> ttrpc::Error {
    ttrpc::Error::Others(message.to_string())
}

fn pid() -> u32 {
    std::process::id()
}

impl shim::Shim for Service {
    type T = Service;

    fn new(_runtime_id: &str, _args: &Flags, _config: &mut Config) -> Self {
        Self {
            exit: Arc::new(ExitSignal::default()),
            task: Arc::new((Mutex::new(None), Condvar::new())),
        }
    }

    fn start_shim(&mut self, opts: StartOpts) -> Result<String, shim::Error> {
        let grouping = opts.id.clone();
        let (_pid, address) = shim::spawn(opts, &grouping, Vec::new())?;
        Ok(address)
    }

    fn delete_shim(&mut self) -> Result<DeleteResponse, shim::Error> {
        Ok(DeleteResponse::new())
    }

    fn wait(&mut self) {
        self.exit.wait();
    }

    fn create_task_service(&self, _publisher: RemotePublisher) -> Self::T {
        self.clone()
    }
}

impl Service {
    fn with_task<R>(&self, f: impl FnOnce(&mut Task) -> R) -> TtrpcResult<R> {
        let mut task = self.task.0.lock().unwrap();
        let task = task
            .as_mut()
            .ok_or_else(|| error("no task has been created"))?;
        Ok(f(task))
    }
}

/// The container's standard streams, from the FIFOs containerd made for it.
fn stdio(stdin: &str, stdout: &str, stderr: &str) -> std::io::Result<Stdio> {
    let input: Box<dyn Read + Send> = if stdin.is_empty() {
        Box::new(std::io::empty())
    } else {
        Box::new(File::open(stdin)?)
    };
    let output = |path: &str| -> std::io::Result<Box<dyn std::io::Write + Send>> {
        if path.is_empty() {
            Ok(Box::new(std::io::sink()))
        } else {
            Ok(Box::new(OpenOptions::new().write(true).open(path)?))
        }
    };
    Ok(Stdio {
        stdin: input,
        stdout: output(stdout)?,
        stderr: output(stderr)?,
    })
}

impl shim::Task for Service {
    fn create(
        &self,
        _ctx: &TtrpcContext,
        req: api::CreateTaskRequest,
    ) -> TtrpcResult<api::CreateTaskResponse> {
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
        *self.task.0.lock().unwrap() = Some(Task {
            id: req.id,
            bundle,
            stdin: req.stdin,
            stdout: req.stdout,
            stderr: req.stderr,
            status: api::Status::CREATED,
            exited: None,
            kill: Arc::new(AtomicU32::new(0)),
        });
        Ok(api::CreateTaskResponse {
            pid: pid(),
            ..Default::default()
        })
    }

    fn start(
        &self,
        _ctx: &TtrpcContext,
        _req: api::StartRequest,
    ) -> TtrpcResult<api::StartResponse> {
        let (bundle_dir, streams, kill) = self.with_task(|task| {
            task.status = api::Status::RUNNING;
            let streams = (task.stdin.clone(), task.stdout.clone(), task.stderr.clone());
            (task.bundle.clone(), streams, task.kill.clone())
        })?;
        let bundle = Bundle::load(&bundle_dir).map_err(error)?;
        let state = self.task.clone();
        std::thread::spawn(move || {
            let status = stdio(&streams.0, &streams.1, &streams.2)
                .map_err(anyhow::Error::from)
                .and_then(|stdio| bundle::run(&bundle, stdio, &kill))
                .unwrap_or_else(|err| {
                    warn!("the container failed: {err:#}");
                    125
                });
            let (lock, exited) = &*state;
            if let Some(task) = lock.lock().unwrap().as_mut() {
                task.status = api::Status::STOPPED;
                task.exited = Some((status as u32, Timestamp::now()));
            }
            exited.notify_all();
        });
        Ok(api::StartResponse {
            pid: pid(),
            ..Default::default()
        })
    }

    fn wait(&self, _ctx: &TtrpcContext, _req: api::WaitRequest) -> TtrpcResult<api::WaitResponse> {
        let (lock, exited) = &*self.task;
        let mut task = lock.lock().unwrap();
        loop {
            match task.as_ref() {
                None => return Err(error("no task has been created")),
                Some(Task {
                    exited: Some((status, at)),
                    ..
                }) => {
                    return Ok(api::WaitResponse {
                        exit_status: *status,
                        exited_at: MessageField::some(at.clone()),
                        ..Default::default()
                    });
                }
                Some(_) => task = exited.wait(task).unwrap(),
            }
        }
    }

    fn state(
        &self,
        _ctx: &TtrpcContext,
        _req: api::StateRequest,
    ) -> TtrpcResult<api::StateResponse> {
        self.with_task(|task| api::StateResponse {
            id: task.id.clone(),
            bundle: task.bundle.display().to_string(),
            pid: pid(),
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
        let signal = if req.signal == 0 { SIGKILL } else { req.signal };
        self.with_task(|task| task.kill.store(signal, Ordering::SeqCst))?;
        Ok(api::Empty::default())
    }

    fn delete(
        &self,
        _ctx: &TtrpcContext,
        _req: api::DeleteRequest,
    ) -> TtrpcResult<api::DeleteResponse> {
        let (bundle, exited) = self.with_task(|task| (task.bundle.clone(), task.exited.clone()))?;
        let rootfs = bundle.join("rootfs");
        if let Err(err) = shim::mount::umount_recursive(rootfs.to_str(), 0) {
            warn!("unmounting {}: {err}", rootfs.display());
        }
        Ok(api::DeleteResponse {
            pid: pid(),
            exit_status: exited.as_ref().map_or(0, |(status, _)| *status),
            exited_at: MessageField::from_option(exited.map(|(_, at)| at)),
            ..Default::default()
        })
    }

    fn pids(&self, _ctx: &TtrpcContext, _req: api::PidsRequest) -> TtrpcResult<api::PidsResponse> {
        Ok(api::PidsResponse {
            processes: vec![containerd_shim::protos::types::task::ProcessInfo {
                pid: pid(),
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
        Ok(api::ConnectResponse {
            shim_pid: pid(),
            task_pid: pid(),
            version: env!("CARGO_PKG_VERSION").into(),
            ..Default::default()
        })
    }

    fn shutdown(&self, _ctx: &TtrpcContext, _req: api::ShutdownRequest) -> TtrpcResult<api::Empty> {
        self.exit.signal();
        Ok(api::Empty::default())
    }
}
