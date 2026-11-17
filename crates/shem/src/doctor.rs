//! `shem doctor`: what shem needs to run what you ask of it, found or not,
//! with what to do about each thing missing.
//!
//! A check is `ok`, `warn` (an optional part is missing: something shem can do
//! without, or that you may never need) or `fail` (something is there and
//! broken). Only a failure makes `shem doctor` exit with a status of 1.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::neural;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub level: Level,
    pub name: String,
    pub detail: String,
    /// What to do about it, if it is not ok
    pub hint: Option<String>,
}

impl Check {
    fn ok(name: &str, detail: impl Into<String>) -> Self {
        Self {
            level: Level::Ok,
            name: name.into(),
            detail: detail.into(),
            hint: None,
        }
    }

    fn warn(name: &str, detail: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            level: Level::Warn,
            name: name.into(),
            detail: detail.into(),
            hint: Some(hint.into()),
        }
    }

    fn fail(name: &str, detail: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            level: Level::Fail,
            name: name.into(),
            detail: detail.into(),
            hint: Some(hint.into()),
        }
    }
}

/// The executable `name` in the directories of `path`, as a shell would find it.
pub fn find_in_path(name: &str, path: &OsStr) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    std::env::split_paths(path)
        .map(|dir| dir.join(name))
        .find(|candidate| {
            candidate
                .metadata()
                .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
        })
}

/// What `command` prints, if it ends within a few seconds: a program that
/// hangs, as Docker does when its daemon is stuck, must not hang the doctor.
fn output_of(command: &mut Command) -> Option<String> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let started = Instant::now();
    loop {
        if child.try_wait().ok()?.is_some() {
            let out = child.wait_with_output().ok()?;
            return Some(String::from_utf8_lossy(&out.stdout).trim().to_string());
        }
        if started.elapsed() > Duration::from_secs(5) {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Whether a Hugging Face token is where `huggingface_hub` looks for one.
pub fn has_hf_token(env: impl Fn(&str) -> Option<OsString>, home: Option<&Path>) -> bool {
    let given = |name: &str| env(name).is_some_and(|value| !value.is_empty());
    if given("HF_TOKEN") || given("HUGGING_FACE_HUB_TOKEN") {
        return true;
    }
    let hub = env("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| home.map(|home| home.join(".cache/huggingface")));
    hub.is_some_and(|dir| {
        std::fs::read_to_string(dir.join("token")).is_ok_and(|token| !token.trim().is_empty())
    })
}

/// A tool that is installed, with its version where it will say it.
fn tool(name: &str, version: &[&str], why: &str, hint: &str, path: &OsStr) -> Check {
    match find_in_path(name, path) {
        Some(found) => {
            let said = output_of(Command::new(&found).args(version)).unwrap_or_default();
            let said = said
                .lines()
                .find(|line| line.chars().any(|c| c.is_ascii_digit()))
                .unwrap_or("")
                .trim()
                .to_string();
            Check::ok(name, format!("{said} ({})", found.display()))
        }
        None => Check::warn(name, format!("not found: {why}"), hint),
    }
}

/// Every check, for this machine and environment.
pub fn run() -> Vec<Check> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let env = |name: &str| std::env::var_os(name);
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut checks = vec![Check::ok(
        "shem",
        format!(
            "{} on {} {}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH
        ),
    )];

    checks.push(tool(
        "uv",
        &["--version"],
        "models are imported and run with it",
        "install uv: https://docs.astral.sh/uv/",
        &path,
    ));
    checks.push(tool(
        "oras",
        &["version"],
        "pushing and pulling images needs it",
        "install ORAS: https://oras.land/docs/installation (the install script adds it)",
        &path,
    ));
    checks.push(tool(
        "cosign",
        &["version"],
        "`shem sign` and `shem verify` need it",
        "install cosign: https://docs.sigstore.dev/cosign/",
        &path,
    ));

    // Docker, and on Linux the shim that lets it run shem's images
    match find_in_path("docker", &path) {
        Some(docker) => {
            let said = output_of(Command::new(&docker).args([
                "version",
                "--format",
                "{{.Server.Version}}",
            ]));
            match said.filter(|text| !text.is_empty()) {
                Some(version) => checks.push(Check::ok("docker", format!("server {version}"))),
                None => checks.push(Check::warn(
                    "docker",
                    "installed, and its server does not answer",
                    "start Docker, or ignore this if you do not use it",
                )),
            }
        }
        None => checks.push(Check::warn(
            "docker",
            "not found: building with `docker build` and running with `docker run` need it",
            "install Docker: https://docs.docker.com/get-docker/",
        )),
    }
    if cfg!(target_os = "linux") {
        match find_in_path("containerd-shim-shem-v1", &path) {
            Some(shim) => checks.push(Check::ok("containerd shim", shim.display().to_string())),
            None => checks.push(Check::warn(
                "containerd shim",
                "containerd-shim-shem-v1 is not on the PATH: `docker run --runtime io.containerd.shem.v1` needs it",
                "run the install script, or `sudo install` the shim to /usr/local/bin",
            )),
        }
    } else {
        checks.push(Check::warn(
            "containerd shim",
            "Docker runs containers in a VM here, where the shim cannot be",
            "build with Docker, then `docker save IMAGE | shem load && shem run IMAGE`",
        ));
    }

    // The model server: where `shem run` looks for it, and where the shim does
    let socket = neural::default_socket();
    match neural::connect_stream(&socket) {
        Ok(_) => checks.push(Check::ok("model server", format!("listening at {}", socket.display()))),
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => checks.push(Check::fail(
            "model server",
            format!("{err}"),
            "remove the socket, or find out who made it: shem sends nothing to another user's server",
        )),
        Err(_) => checks.push(Check::ok(
            "model server",
            format!("not running; `shem run` of a model starts one at {}", socket.display()),
        )),
    }
    let service = Path::new("/run/shem/neural.sock");
    if cfg!(target_os = "linux") && !service.exists() {
        checks.push(Check::warn(
            "model server for Docker",
            "nothing at /run/shem/neural.sock: model containers under Docker need a server there",
            "run the install script with SHEM_NEURAL_SERVICE=1, or start `sudo shem-neural-server`",
        ));
    }

    checks.push(if has_hf_token(env, home.as_deref()) {
        Check::ok("Hugging Face token", "found")
    } else {
        Check::warn(
            "Hugging Face token",
            "none: gated and private models, and pushes to the Hub, need one",
            "set HF_TOKEN, or `hf auth login`",
        )
    });

    // The store, which every build and pull writes to
    match crate::image::Store::open() {
        Ok(store) => {
            let home = store.home();
            let probe =
                std::fs::create_dir_all(home).and_then(|()| tempfile::tempfile_in(home).map(drop));
            checks.push(match probe {
                Ok(()) => Check::ok("image store", home.display().to_string()),
                Err(err) => Check::fail(
                    "image store",
                    format!("{} is not writable: {err}", home.display()),
                    "set SHEM_HOME to a directory you can write",
                ),
            });
        }
        Err(err) => checks.push(Check::fail(
            "image store",
            format!("{err:#}"),
            "set SHEM_HOME or HOME",
        )),
    }

    // The kernel call that keeps a container's files beneath its root
    #[cfg(target_os = "linux")]
    if let Ok(dir) = tempfile::tempdir()
        && let Ok(root) = crate::linux::RootFs::new(dir.path())
    {
        checks.push(match root.open_beneath(b"/", libc::O_RDONLY | libc::O_DIRECTORY, 0) {
            Some(_) => Check::ok("openat2", "files open beneath a container's root in one kernel step"),
            None => Check::warn(
                "openat2",
                "not in this kernel (before Linux 5.6): a container's files are opened by path, which a host writer could race",
                "upgrade the kernel, or do not share a container's root with other processes",
            ),
        });
    }
    checks
}

/// The checks as lines, and whether none failed.
pub fn report(checks: &[Check]) -> (String, bool) {
    let width = checks.iter().map(|c| c.name.len()).max().unwrap_or(0);
    let mut out = String::new();
    for check in checks {
        let mark = match check.level {
            Level::Ok => "ok  ",
            Level::Warn => "warn",
            Level::Fail => "FAIL",
        };
        out += &format!("{mark}  {:<width$}  {}\n", check.name, check.detail);
        if let Some(hint) = &check.hint {
            out += &format!("      {:<width$}  -> {hint}\n", "");
        }
    }
    (out, checks.iter().all(|c| c.level != Level::Fail))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn a_program_is_found_on_the_path_only_if_it_runs() {
        let dir = tempfile::tempdir().unwrap();
        let (run, text) = (dir.path().join("runs"), dir.path().join("text"));
        std::fs::write(&run, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(&text, "not a program").unwrap();
        let path = std::env::join_paths([dir.path()]).unwrap();
        assert_eq!(find_in_path("runs", &path), Some(run));
        assert_eq!(find_in_path("text", &path), None, "not executable");
        assert_eq!(find_in_path("missing", &path), None);
    }

    #[test]
    fn a_hugging_face_token_is_in_the_environment_or_in_the_hub_cache() {
        let home = tempfile::tempdir().unwrap();
        let none = |_: &str| None;
        assert!(!has_hf_token(none, Some(home.path())));
        assert!(has_hf_token(
            |n| (n == "HF_TOKEN").then(|| "hf_x".into()),
            None
        ));
        assert!(!has_hf_token(
            |n| (n == "HF_TOKEN").then(|| "".into()),
            None
        ));
        let cache = home.path().join(".cache/huggingface");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::write(cache.join("token"), "hf_y\n").unwrap();
        assert!(has_hf_token(none, Some(home.path())));
        let elsewhere = tempfile::tempdir().unwrap();
        std::fs::write(elsewhere.path().join("token"), "").unwrap();
        let hf_home = elsewhere.path().to_path_buf();
        assert!(
            !has_hf_token(
                |n| (n == "HF_HOME").then(|| hf_home.clone().into()),
                Some(home.path())
            ),
            "HF_HOME is where it looks, and the token there is empty"
        );
    }

    #[test]
    fn the_report_says_what_to_do_and_a_failure_is_what_fails_it() {
        let checks = [
            Check::ok("uv", "0.10 (/usr/bin/uv)"),
            Check::warn("oras", "not found", "install it"),
        ];
        let (text, healthy) = report(&checks);
        assert!(healthy, "a missing optional tool is a warning");
        assert!(
            text.contains("ok    uv")
                && text.contains("warn  oras")
                && text.contains("-> install it")
        );
        let (text, healthy) =
            report(&[Check::fail("image store", "not writable", "set SHEM_HOME")]);
        assert!(!healthy && text.starts_with("FAIL"));
    }

    #[test]
    fn a_tool_that_answers_is_reported_with_its_version() {
        let dir = tempfile::tempdir().unwrap();
        let fake = dir.path().join("fake");
        std::fs::write(&fake, "#!/bin/sh\necho 'fake 1.2.3'\n").unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = std::env::join_paths([dir.path()]).unwrap();
        let found = tool("fake", &["--version"], "why", "hint", &path);
        assert_eq!(found.level, Level::Ok);
        assert!(found.detail.starts_with("fake 1.2.3"));
        assert_eq!(tool("absent", &[], "why", "hint", &path).level, Level::Warn);
    }
}
