//! Model images: a language model as layers, and the drivers that run it.
//!
//! `FROM hf.co/ORG/NAME[:REVISION]` imports a Hugging Face checkpoint into
//! `/model` with the neural package's importer (`shem_neural.model.importers`):
//! a manifest, `shem.json`, that lists the model's blocks by type, and one
//! weights file per block. Each weights file becomes its own layer, so images
//! built on the same model share those blobs in stores and registries.
//!
//! An entrypoint that starts with `shem:` names a driver, which the host runs
//! instead of a guest program, like a shell builtin. `shem:generate` runs a
//! session of the image's model, with the rest of the entrypoint as its
//! options and the arguments, or CMD, as the prompt.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};

/// Where a model image keeps its model, relative to the root.
pub const MODEL_DIR: &str = "model";
/// The registry prefix that names a Hugging Face model in `FROM`.
pub const HF_PREFIX: &str = "hf.co/";
pub const DRIVER_PREFIX: &str = "shem:";
/// Overlay filesystems stack at most about 128 layers, so a model with more
/// weights files than this shares layers between neighbouring blocks.
const MAX_MODEL_LAYERS: usize = 96;

/// A Hugging Face model named in `FROM`: the repo and an optional revision.
pub fn hf_reference(from: &str) -> Option<(&str, Option<&str>)> {
    let rest = from.strip_prefix(HF_PREFIX)?;
    Some(match rest.split_once(':') {
        Some((repo, revision)) => (repo, Some(revision)),
        None => (rest, None),
    })
}

/// The neural package's source, which imports and runs models:
/// `SHEM_NEURAL_PROJECT`, or the copy next to the source this binary was
/// built from, if there is one.
pub fn neural_project() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("SHEM_NEURAL_PROJECT") {
        return Some(PathBuf::from(dir));
    }
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../neural");
    checkout.join("pyproject.toml").is_file().then_some(checkout)
}

/// Run a module of the neural package with uv: from its source when there
/// is one, otherwise the release on PyPI that matches this binary.
fn python(module: &str) -> Command {
    let mut command = Command::new("uv");
    command.args(["run", "--quiet"]);
    match neural_project() {
        Some(project) => command.arg("--project").arg(project),
        None => command
            .args(["--no-project", "--with"])
            .arg(concat!("shem-neural==", env!("CARGO_PKG_VERSION"))),
    };
    command.args(["python", "-m", module]);
    command
}

/// Import `repo` at `revision` into `out`, as a manifest and weights files.
pub fn import(repo: &str, revision: Option<&str>, out: &Path) -> anyhow::Result<()> {
    let mut command = python("shem_neural.model.importers");
    command.arg(repo).arg(out);
    if let Some(revision) = revision {
        command.args(["--revision", revision]);
    }
    let status = command
        .status()
        .context("running the model importer, which needs uv")?;
    if !status.success() {
        bail!("importing {repo} failed ({status})");
    }
    Ok(())
}

/// The model's files under `root`, grouped into layers: the directories and
/// small files first, then each weights file alone, or a few together when
/// there would otherwise be too many layers. Paths are relative to `root`.
pub fn layer_groups(root: &Path) -> anyhow::Result<Vec<Vec<PathBuf>>> {
    let dir = Path::new(MODEL_DIR);
    let mut small = vec![dir.to_path_buf(), dir.join("blocks")];
    let mut weights = Vec::new();
    for sub in [dir.to_path_buf(), dir.join("blocks")] {
        let mut entries: Vec<_> = std::fs::read_dir(root.join(&sub))?.collect::<Result<_, _>>()?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            if entry.file_type()?.is_dir() {
                continue;
            }
            let path = sub.join(entry.file_name());
            if path.extension().is_some_and(|e| e == "safetensors") {
                weights.push(path);
            } else {
                small.push(path);
            }
        }
    }
    let per_layer = weights.len().div_ceil(MAX_MODEL_LAYERS - 1).max(1);
    let mut groups = vec![small];
    groups.extend(weights.chunks(per_layer).map(<[PathBuf]>::to_vec));
    Ok(groups)
}

/// Where a model image's primed session lives, relative to the root, once
/// `shem run` or `docker load` has unpacked its state layer.
pub const SESSION_DIR: &str = ".shem";

/// Prime a session of the model at `root` with the driver's options in
/// `entrypoint`, and save it in `out`: what `WARM --until=prompt` freezes.
pub fn warm(entrypoint: &[String], root: &Path, out: &Path) -> anyhow::Result<()> {
    let (driver, options) = entrypoint.split_first().context("no entrypoint")?;
    if driver != "shem:generate" {
        bail!("WARM --until=prompt needs the shem:generate driver, not {driver}");
    }
    let status = python("shem_neural.model.generate")
        .args(options)
        .arg("--warm")
        .arg(out)
        .arg(root.join(MODEL_DIR))
        .status()
        .context("running the model, which needs uv")?;
    if !status.success() {
        bail!("priming the session failed ({status})");
    }
    Ok(())
}

/// Whether `entrypoint` names a driver rather than a guest program.
pub fn is_driver(entrypoint: &[String]) -> bool {
    entrypoint
        .first()
        .is_some_and(|first| first.starts_with(DRIVER_PREFIX))
}

/// Run the driver `entrypoint` for an image whose filesystem is at `root`,
/// with `args` (the run's arguments, or the image's CMD). Returns its status.
pub fn run_driver(entrypoint: &[String], args: &[String], root: &Path) -> anyhow::Result<u8> {
    let (driver, options) = entrypoint.split_first().context("no entrypoint")?;
    let mut command = match driver.as_str() {
        "shem:generate" => python("shem_neural.model.generate"),
        other => bail!("unknown driver {other}: shem has shem:generate"),
    };
    command.args(options);
    let session = root.join(SESSION_DIR);
    if session.join("session.json").is_file() {
        command.arg("--session").arg(session);
    }
    command.arg(root.join(MODEL_DIR)).arg("--").args(args);
    let status = command
        .status()
        .context("running the model, which needs uv")?;
    Ok(status.code().unwrap_or(1) as u8)
}
