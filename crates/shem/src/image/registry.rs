//! Pushing images to OCI registries and pulling them back.
//!
//! A shem image is a standard OCI image layout, so ORAS, the OCI registry
//! client, copies it as is, whatever its media types; shem runs `oras cp`. The
//! push goes by manifest digest, so nothing depends on how tags are written in
//! the layout. Set `SHEM_ORAS` to use an `oras` that is not on `PATH`.
//!
//! `hf.co/ORG/NAME[:REVISION]` names a Hugging Face repo instead, which
//! keeps an image as its layout's files, at a branch, tag or commit
//! (`latest` is `main`). Pulling one that holds no layout, but a checkpoint,
//! imports the model, as `FROM hf.co/ORG/NAME` does.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, bail};

use super::modelpack::Artifact;
use super::{BuildOptions, Image, build, model, parse_reference};

fn oras() -> Command {
    Command::new(std::env::var_os("SHEM_ORAS").unwrap_or_else(|| "oras".into()))
}

fn run(mut command: Command) -> anyhow::Result<()> {
    let output = command.output().context(
        "running oras, the OCI registry client (https://oras.land); install it or set SHEM_ORAS",
    )?;
    if !output.status.success() {
        bail!(
            "oras failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Push the image in the layout `dir` to `remote`, a registry reference such
/// as `ghcr.io/me/app:1.0`. Returns the manifest digest, which the registry
/// keeps. `plain_http` talks to a registry without TLS, such as a local one.
pub fn push(dir: &Path, remote: &str, plain_http: bool) -> anyhow::Result<String> {
    let digest = Image::layout_digest(dir)?;
    if let Some((repo, revision)) = model::hf_reference(remote) {
        model::push_layout(dir, repo, revision)?;
        return Ok(digest);
    }
    let mut command = oras();
    command
        .args(["cp", "--from-oci-layout"])
        .arg(format!("{}@{digest}", dir.display()));
    if plain_http {
        command.arg("--to-plain-http");
    }
    command.arg(remote);
    run(command)?;
    Ok(digest)
}

/// The registry a reference names, such as ghcr.io in ghcr.io/me/app:1.0.
pub fn registry_host(reference: &str) -> Option<&str> {
    let (first, _) = reference.split_once('/')?;
    (first.contains('.') || first.contains(':') || first == "localhost").then_some(first)
}

/// Whether a registry is local, which speaks plain HTTP.
pub fn is_local(host: &str) -> bool {
    host.starts_with("localhost") || host.starts_with("127.0.0.1")
}

/// Copy `remote`, whatever it is, into a new OCI layout at `dir`, tagged
/// `latest`.
pub fn copy_to_layout(remote: &str, dir: &Path, plain_http: bool) -> anyhow::Result<()> {
    let mut command = oras();
    command.arg("cp");
    if plain_http {
        command.arg("--from-plain-http");
    }
    command
        .arg(remote)
        .arg("--to-oci-layout")
        .arg(format!("{}:latest", dir.display()));
    run(command)
}

/// Pull `remote` into the layout directory `dir`, replacing what is there.
/// A ModelPack artifact becomes its image twin, whose layers are the
/// artifact's own blobs; anything else must be a shem image. Returns the
/// image's manifest digest.
pub fn pull(remote: &str, dir: &Path, plain_http: bool) -> anyhow::Result<String> {
    let parent = dir.parent().context("no directory to pull into")?;
    std::fs::create_dir_all(parent)?;
    let staging = tempfile::tempdir_in(parent)?;
    let layout = staging.path().join("layout");
    match model::hf_reference(remote) {
        Some((repo, revision)) => {
            if !model::fetch_layout(repo, revision, &layout)? {
                import(remote, &staging.path().join("context"), &layout)?;
            }
        }
        None => copy_to_layout(remote, &layout, plain_http)?,
    }
    let layout = match Artifact::read(&layout)? {
        Some(artifact) => {
            let root = tempfile::tempdir()?;
            let twin = staging.path().join("twin");
            artifact
                .twin(remote, root.path())?
                .write_layout(&twin, "latest")?;
            twin
        }
        None => {
            Image::read_layout(&layout).with_context(|| format!("{remote} is not a shem image"))?;
            layout
        }
    };
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::rename(&layout, dir)?;
    Image::layout_digest(dir)
}

/// Import the model of the Hugging Face repo `remote` names, as a Shemfile
/// of `FROM remote` alone builds it, into a layout at `out`. `context` is an
/// empty directory to build in.
fn import(remote: &str, context: &Path, out: &Path) -> anyhow::Result<()> {
    let (name, tag) = parse_reference(remote)?;
    // `latest`, which a reference without a tag means, is no revision to import
    let from = if tag == "latest" { name } else { remote };
    std::fs::create_dir_all(context)?;
    let shemfile = context.join("Shemfile");
    std::fs::write(&shemfile, format!("FROM {from}\n"))?;
    let image = build(&BuildOptions {
        shemfile,
        context: context.to_path_buf(),
        build_args: BTreeMap::new(),
        mem: None,
    })?;
    image.write_layout(out, tag)?;
    Ok(())
}
