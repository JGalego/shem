//! Load OCI runtime bundles and run their processes on shem.
//!
//! `config.json` supplies the [runtime spec]. `dev.shem.vcpu=neural` selects
//! the neural server, with `dev.shem.neural-socket` overriding its socket.
//! Commands starting with `shem:` use the model-session driver instead.
//!
//! [runtime spec]: https://github.com/opencontainers/runtime-spec/blob/main/config.md

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context, bail};
use serde_json::Value;
use shem::container::{Container, Frozen, FrozenHost, RunError};
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
        })
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
/// ignoring the bundle's arguments. A nonzero `kill` (a signal number) stops
/// it at the next slice boundary, with the status a shell reports for that
/// signal.
pub fn run(bundle: &Bundle, stdio: Stdio, kill: &AtomicU32) -> anyhow::Result<u8> {
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
        VcpuKind::Reference => run_on(bundle, stdio, kill, SLICE, |mem| {
            Ok(ReferenceVcpu::new(mem))
        }),
        VcpuKind::Neural(socket) => run_on(bundle, stdio, kill, NEURAL_SLICE, |mem| {
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
    kill: &AtomicU32,
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
    loop {
        match container.run(Some(container.steps + slice)) {
            Ok(status) => return Ok(status),
            Err(RunError::StepLimit(_)) => match kill.load(Ordering::SeqCst) {
                0 => continue,
                signal => return Ok(128u32.saturating_add(signal) as u8),
            },
            Err(err) => return Err(err.into()),
        }
    }
}
