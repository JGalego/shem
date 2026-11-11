//! OCI runtime bundles: what containerd hands a shim to run.
//!
//! A bundle is a directory holding `config.json`, the [runtime spec], and the
//! container's root filesystem. This module reads the parts shem uses (the
//! process's arguments, environment and working directory, the root, and the
//! memory limit) and runs the process as a shem container.
//!
//! [runtime spec]: https://github.com/opencontainers/runtime-spec/blob/main/config.md

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context, bail};
use serde_json::Value;
use shem::container::{Container, RunError};
use shem::image::find_program;
use shem::linux::{RootFs, Stdio};
use shem::reference::ReferenceVcpu;

/// Guest memory when the bundle sets no memory limit.
pub const DEFAULT_MEMORY: u32 = 64 << 20;
/// Instructions between checks for a kill request.
const SLICE: u64 = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bundle {
    pub args: Vec<String>,
    pub env: Vec<String>,
    pub cwd: String,
    pub root: PathBuf,
    pub memory: u32,
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
        Ok(Self {
            args,
            env: strings(&process["env"]),
            cwd: process["cwd"].as_str().unwrap_or("/").to_string(),
            root: dir.join(root),
            memory,
        })
    }
}

/// Run the bundle's process to completion and return its exit status. A
/// nonzero `kill` (a signal number) stops it at the next slice boundary, with
/// the status a shell reports for that signal.
pub fn run(bundle: &Bundle, stdio: Stdio, kill: &AtomicU32) -> anyhow::Result<u8> {
    let rootfs = RootFs::new(&bundle.root).context("the bundle's root filesystem")?;
    let program = find_program(&rootfs, &bundle.cwd, &bundle.env, &bundle.args[0])?;
    let elf = std::fs::read(program)?;
    let vcpu = ReferenceVcpu::new(bundle.memory);
    let mut container = Container::from_elf(vcpu, &elf, &bundle.args, &bundle.env, stdio)?;
    container.set_root(&bundle.root)?;
    if bundle.cwd != "/" {
        container.set_cwd(&bundle.cwd).map_err(|errno| {
            anyhow::anyhow!("working directory {}: errno {}", bundle.cwd, errno.0)
        })?;
    }
    loop {
        match container.run(Some(container.steps + SLICE)) {
            Ok(status) => return Ok(status),
            Err(RunError::StepLimit(_)) => match kill.load(Ordering::SeqCst) {
                0 => continue,
                signal => return Ok(128u32.saturating_add(signal) as u8),
            },
            Err(err) => return Err(err.into()),
        }
    }
}
