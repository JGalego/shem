//! Import language models, group their files into layers and run their drivers.
//!
//! `hf.co/ORG/NAME[:REVISION]` bases are imported into `/model` by the neural
//! package. A `shem:` entrypoint runs a host-side driver rather than a guest
//! executable. See `docs/models.md`.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};

/// Where a model image keeps its model, relative to the root.
pub const MODEL_DIR: &str = "model";
/// The registry prefix that names a Hugging Face model in `FROM`.
pub const HF_PREFIX: &str = "hf.co/";
pub const DRIVER_PREFIX: &str = "shem:";
/// The driver that runs a model's text-generation sessions.
pub const DRIVER: &str = "shem:generate";
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
    checkout
        .join("pyproject.toml")
        .is_file()
        .then_some(checkout)
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
    // Tools' programs run through this shem, on the model's compiled layer
    if let Ok(exe) = std::env::current_exe() {
        command.env("SHEM_BIN", exe);
    }
    command
}

/// The compiled layers shem can build, for `COPY --from=shem:NAME`.
pub const COMPILED: [&str; 1] = ["rv32im"];

/// Compile the layer `name` into `out`: its weights, its description and
/// its model card, as `shem_neural.export` writes them.
pub fn compile(name: &str, out: &Path) -> anyhow::Result<()> {
    if !COMPILED.contains(&name) {
        bail!("shem:{name} is no compiled layer; there is shem:rv32im");
    }
    let status = python("shem_neural.export")
        .arg(out)
        .arg("--version")
        .arg(env!("CARGO_PKG_VERSION"))
        .status()
        .context("compiling the layer, which needs uv")?;
    if !status.success() {
        bail!("compiling shem:{name} failed ({status})");
    }
    Ok(())
}

/// Import `repo` at `revision` into `out`, as a manifest and weights files:
/// a Hugging Face model (in a repo of GGUF files, `revision` names a
/// quantization), or a checkpoint's directory or GGUF file, by its full path.
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

/// Write into `out` what the checkpoint in `dir` needs to run as a model
/// besides its own files, which stay where they are: the manifest, and an
/// index or a chat template if the checkpoint has none.
pub fn import_in_place(dir: &Path, out: &Path) -> anyhow::Result<()> {
    let status = python("shem_neural.model.importers")
        .arg(dir)
        .arg(out)
        .arg("--in-place")
        .status()
        .context("running the model importer, which needs uv")?;
    if !status.success() {
        bail!(
            "importing the checkpoint in {} failed ({status})",
            dir.display()
        );
    }
    Ok(())
}

/// The standard OCI labels for an imported model: where it comes from, and
/// the licence its model card declares, which also travels as the model's
/// `LICENSE` and `NOTICE` files.
pub fn labels(root: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(MODEL_DIR).join("shem.json"))?)?;
    let source = &manifest["source"];
    let mut labels = Vec::new();
    if let Some(repo) = source["repo"].as_str() {
        let url = format!("https://huggingface.co/{repo}");
        labels.push(("org.opencontainers.image.source".into(), url));
    }
    if let Some(spdx) = source["spdx"].as_str() {
        labels.push(("org.opencontainers.image.licenses".into(), spdx.into()));
    }
    Ok(labels)
}

/// The model's files under `root`, grouped into layers: the directory and
/// small files first, then each weights file alone, or a few together when
/// there would otherwise be too many layers. Paths are relative to `root`.
/// Weights files are at the top of the model's directory, or in `blocks/`
/// in imports by earlier versions.
pub fn layer_groups(root: &Path) -> anyhow::Result<Vec<Vec<PathBuf>>> {
    let dir = Path::new(MODEL_DIR);
    let mut small = vec![dir.to_path_buf()];
    let mut weights = Vec::new();
    let mut subs = vec![dir.to_path_buf()];
    if root.join(dir).join("blocks").is_dir() {
        small.push(dir.join("blocks"));
        subs.push(dir.join("blocks"));
    }
    for sub in subs {
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

/// Prime a session of the model in the image at `root` with the driver's
/// options in `entrypoint`, and save it in `out`: what `WARM --until=prompt`
/// freezes. The driver finds the model at `/model` or `/`.
pub fn warm(entrypoint: &[String], root: &Path, out: &Path) -> anyhow::Result<()> {
    let (driver, options) = entrypoint.split_first().context("no entrypoint")?;
    if driver != DRIVER {
        bail!("WARM --until=prompt needs the shem:generate driver, not {driver}");
    }
    let status = python("shem_neural.model.generate")
        .args(options)
        .arg("--warm")
        .arg(out)
        .arg(root)
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
/// with `args` (the run's arguments, or the image's CMD), continuing the
/// session in `session`, or else the image's own, if it has one. Returns
/// the driver's status.
pub fn run_driver(
    entrypoint: &[String],
    args: &[String],
    root: &Path,
    session: Option<&Path>,
) -> anyhow::Result<u8> {
    let (driver, options) = entrypoint.split_first().context("no entrypoint")?;
    let mut command = match driver.as_str() {
        "shem:generate" => python("shem_neural.model.generate"),
        other => bail!("unknown driver {other}: shem has shem:generate"),
    };
    command.args(options);
    let own = root.join(SESSION_DIR);
    let session = session.or(own.join("session.json").is_file().then_some(own.as_path()));
    if let Some(session) = session {
        command.arg("--session").arg(session);
    }
    command.arg(root).arg("--").args(args);
    let status = command
        .status()
        .context("running the model, which needs uv")?;
    Ok(status.code().unwrap_or(1) as u8)
}
