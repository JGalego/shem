//! Images for Docker: an image layout as an archive that `docker load` takes,
//! to run with `docker run --runtime io.containerd.shem.v1` (see
//! `docs/containerd.md`).
//!
//! shem's layouts are already standard OCI images for `linux/riscv32`, so the
//! archive is the layout as a tar, with the name Docker gives the image. A
//! warm image's state travels as its last layer, files under `/.shem`, which
//! the containerd shim thaws instead of starting the entrypoint.

use anyhow::bail;

use super::{Image, layer};

/// Where a warm image's state lives in its filesystem.
pub const WARM_DIR: &str = super::STATE_DIR;

impl Image {
    /// The image as an OCI image archive named `name` (such as
    /// `hello:latest`) for `docker load`. The archive is reproducible: the
    /// same image gives the same bytes.
    pub fn docker_archive(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        if self.config.kernel.is_some() {
            bail!("VM images cannot run under Docker yet: the shim runs containers and models");
        }
        let dir = tempfile::tempdir()?;
        let (_, tag) = super::parse_reference(name)?;
        self.write_layout(dir.path(), tag)?;
        let path = dir.path().join("index.json");
        let mut index: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        index["manifests"][0]["annotations"]["io.containerd.image.name"] = docker_name(name)?.into();
        std::fs::write(&path, serde_json::to_vec(&index)?)?;

        let mut files = vec!["oci-layout".to_string(), "index.json".to_string()];
        let mut blobs: Vec<_> = std::fs::read_dir(dir.path().join("blobs/sha256"))?
            .map(|entry| entry.map(|e| format!("blobs/sha256/{}", e.file_name().to_string_lossy())))
            .collect::<Result<_, _>>()?;
        blobs.sort();
        files.extend(blobs);
        let contents: Vec<(String, Vec<u8>)> = files
            .into_iter()
            .map(|name| Ok((name.clone(), std::fs::read(dir.path().join(&name))?)))
            .collect::<anyhow::Result<_>>()?;
        let entries: Vec<(&str, &[u8])> = contents.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        layer::from_files(&entries)
    }
}

/// A reference as Docker spells it in full: `hello` is
/// `docker.io/library/hello:latest`.
fn docker_name(reference: &str) -> anyhow::Result<String> {
    let (name, tag) = super::parse_reference(reference)?;
    let first = name.split('/').next().unwrap_or_default();
    let has_host = name.contains('/')
        && (first.contains('.') || first.contains(':') || first == "localhost");
    let full = if has_host {
        name.to_string()
    } else if name.contains('/') {
        format!("docker.io/{name}")
    } else {
        format!("docker.io/library/{name}")
    };
    Ok(format!("{full}:{tag}"))
}
