//! The namespace holder: a process that holds the container's network
//! namespace, as Kubernetes' pause container does.
//!
//! The guest runs on a vCPU inside the shim, so it has no process of its own,
//! and the shim lives in the host's namespaces. Docker, though, configures a
//! container's network through the task's process (`/proc/PID/ns/net`) before
//! starting it. So when the runtime spec asks for a network namespace, the
//! shim starts a holder in a new one, or in the one at the spec's path, and
//! gives its PID as the task's. The guest has no network device, so it never
//! notices. The holder exits when the shim lets go of it, or exits itself.

use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::process::{Child, Command, Stdio};

use anyhow::Context;

const HOLD: &str = "SHEM_SHIM_HOLD";
const NETNS: &str = "SHEM_SHIM_NETNS";

pub struct Holder {
    child: Child,
}

impl Holder {
    /// Hold the network namespace the bundle's spec asks for, if it asks for
    /// one; otherwise the container shares the host's, and there is nothing
    /// to hold.
    pub fn spawn(bundle: &Path) -> anyhow::Result<Option<Self>> {
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(bundle.join("config.json"))?)?;
        let namespaces = config["linux"]["namespaces"].as_array();
        let Some(network) = namespaces.and_then(|all| all.iter().find(|n| n["type"] == "network"))
        else {
            return Ok(None);
        };
        let mut command = Command::new("/proc/self/exe");
        command
            .env(HOLD, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(path) = network["path"].as_str().filter(|p| !p.is_empty()) {
            command.env(NETNS, path);
        }
        let mut child = command.spawn().context("starting the namespace holder")?;
        let mut ready = [0u8; 1];
        (child.stdout.take().context("the holder's stdout")?)
            .read_exact(&mut ready)
            .context("the namespace holder could not enter its namespace")?;
        Ok(Some(Self { child }))
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Holder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// If this process was started as a holder, be one: enter the namespace, say
/// so on stdout, and wait until the shim closes stdin, which it also does by
/// exiting. Returns only if this process is not a holder.
pub fn run_if_holder() {
    if std::env::var_os(HOLD).is_none() {
        return;
    }
    // SAFETY: setns and unshare take a namespace fd or flags and nothing else
    let entered = match std::env::var_os(NETNS) {
        Some(path) => File::open(path)
            .map(|file| unsafe { libc::setns(file.as_raw_fd(), libc::CLONE_NEWNET) } == 0)
            .unwrap_or(false),
        None => (unsafe { libc::unshare(libc::CLONE_NEWNET) }) == 0,
    };
    if !entered {
        let error = std::io::Error::last_os_error();
        eprintln!("containerd-shim-shem: the holder cannot enter a network namespace: {error}");
        std::process::exit(1);
    }
    let _ = std::io::stdout().write_all(b"r");
    let _ = std::io::stdout().flush();
    let _ = std::io::stdin().read(&mut [0u8; 1]);
    std::process::exit(0);
}
