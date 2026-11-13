//! Images for Docker: an image layout as an archive that `docker load` takes,
//! to run with `docker run --runtime io.containerd.shem.v1` (see
//! `docs/containerd.md`).
//!
//! shem's layouts are already standard OCI images for `linux/riscv32`, so the
//! archive is the layout as a tar, with the name Docker gives the image. A
//! warm image's state travels as its last layer, files under `/.shem`, which
//! the containerd shim thaws instead of starting the entrypoint.
//!
//! The other way, `shem load` stores the images of an archive `docker save`
//! writes, to run images Docker built where its containerd cannot see the
//! shim, as with Docker Desktop.

use std::io::{Read, Write};

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};

use super::{Image, Index, REF_NAME, Store, layer};

/// Where a warm image's state lives in its filesystem.
pub const WARM_DIR: &str = super::STATE_DIR;

impl Image {
    /// The image as an OCI image archive named `name` (such as
    /// `hello:latest`) for `docker load`. The archive is reproducible: the
    /// same image gives the same bytes.
    pub fn docker_archive(&self, name: &str, also: &[String]) -> anyhow::Result<Vec<u8>> {
        if self.config.kernel.is_some() {
            bail!("VM images cannot run under Docker yet: the shim runs containers and models");
        }
        let dir = tempfile::tempdir()?;
        let (_, tag) = super::parse_reference(name)?;
        self.write_layout_for(dir.path(), tag, also)?;
        let path = dir.path().join("index.json");
        let mut index: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        index["manifests"][0]["annotations"]["io.containerd.image.name"] =
            docker_name(name)?.into();
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
        let entries: Vec<(&str, &[u8])> = contents
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        layer::from_files(&entries)
    }
}

/// A reference as Docker spells it in full: `hello` is
/// `docker.io/library/hello:latest`.
fn docker_name(reference: &str) -> anyhow::Result<String> {
    let (name, tag) = super::parse_reference(reference)?;
    let first = name.split('/').next().unwrap_or_default();
    let has_host =
        name.contains('/') && (first.contains('.') || first.contains(':') || first == "localhost");
    let full = if has_host {
        name.to_string()
    } else if name.contains('/') {
        format!("docker.io/{name}")
    } else {
        format!("docker.io/library/{name}")
    };
    Ok(format!("{full}:{tag}"))
}

/// The short name Docker shows for a reference it spells in full:
/// `docker.io/library/hello:latest` is `hello:latest`.
fn short_name(reference: &str) -> &str {
    reference
        .strip_prefix("docker.io/library/")
        .or_else(|| reference.strip_prefix("docker.io/"))
        .unwrap_or(reference)
}

/// A file being written, and the digest of what was written so far.
struct Hashed {
    file: std::fs::File,
    hasher: Sha256,
}

impl Write for Hashed {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.file.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

impl Store {
    /// Load the images in an OCI image archive, as `docker save` and `shem
    /// save` write it, into the store: each under the name the archive gives
    /// it, or the one image under `tag`. Returns the names. Only the layout
    /// is read from the archive: its index, and blobs that match their
    /// digests.
    pub fn load(&self, archive: impl Read, tag: Option<&str>) -> anyhow::Result<Vec<String>> {
        let tmp = self.root.with_file_name("tmp");
        std::fs::create_dir_all(&tmp)?;
        let staging = tempfile::tempdir_in(&tmp)?;
        let blobs = staging.path().join("blobs/sha256");
        std::fs::create_dir_all(&blobs)?;
        let mut index = None;
        for entry in tar::Archive::new(archive).entries()? {
            let mut entry = entry?;
            if !entry.header().entry_type().is_file() {
                continue;
            }
            let path = entry.path()?.to_string_lossy().into_owned();
            let path = path.trim_start_matches("./");
            if path == "index.json" {
                let mut data = Vec::new();
                entry.read_to_end(&mut data)?;
                index = Some(
                    serde_json::from_slice::<Index>(&data).context("the archive's index.json")?,
                );
            } else if let Some(hex) = path.strip_prefix("blobs/sha256/")
                && hex.len() == 64
                && hex
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            {
                let mut out = Hashed {
                    file: std::fs::File::create(blobs.join(hex))?,
                    hasher: Sha256::new(),
                };
                std::io::copy(&mut entry, &mut out)?;
                let hash = out.hasher.finalize();
                let got: String = hash.iter().map(|b| format!("{b:02x}")).collect();
                if got != hex {
                    bail!("blob sha256:{hex} in the archive does not match its digest");
                }
            }
        }
        let index = index.context("not an OCI image archive: it has no index.json")?;
        if index.manifests.is_empty() {
            bail!("the archive holds no image");
        }
        if tag.is_some() && index.manifests.len() > 1 {
            bail!(
                "the archive holds {} images, and --tag can name only one",
                index.manifests.len()
            );
        }

        // Every image is checked before any is stored
        let mut names = Vec::new();
        for entry in &index.manifests {
            let name = match tag {
                Some(tag) => tag.to_string(),
                None => match entry.annotations.get("io.containerd.image.name") {
                    Some(name) => short_name(name).to_string(),
                    None => bail!("the archive does not name its image: give it one with --tag"),
                },
            };
            super::parse_reference(&name)?;
            let manifest = super::layout_manifest(staging.path(), entry)
                .with_context(|| format!("reading {name} in the archive"))?;
            let config: serde_json::Value =
                serde_json::from_slice(&super::read_blob(staging.path(), &manifest.config)?)?;
            if let Some(arch) = config["architecture"].as_str()
                && arch != "riscv32"
            {
                bail!(
                    "{name} is a linux/{arch} image: build it with --platform linux/riscv32 to run it on shem"
                );
            }
            names.push(name);
        }

        for (i, (entry, name)) in index.manifests.iter().zip(&names).enumerate() {
            let (_, image_tag) = super::parse_reference(name)?;
            // Each image gets its own layout; they share the archive's blobs
            let layout = if i + 1 == index.manifests.len() {
                staging.path().to_path_buf()
            } else {
                let copy = tempfile::tempdir_in(&tmp)?.keep();
                std::fs::create_dir_all(copy.join("blobs/sha256"))?;
                for blob in std::fs::read_dir(&blobs)? {
                    let blob = blob?.path();
                    let to = copy.join("blobs/sha256").join(blob.file_name().unwrap());
                    if std::fs::hard_link(&blob, &to).is_err() {
                        std::fs::copy(&blob, &to)?;
                    }
                }
                copy
            };
            let mut entry = entry.clone();
            entry.annotations.insert(REF_NAME.into(), image_tag.into());
            let single = Index {
                schema_version: 2,
                manifests: vec![entry],
            };
            std::fs::write(layout.join("index.json"), serde_json::to_vec(&single)?)?;
            std::fs::write(
                layout.join("oci-layout"),
                br#"{"imageLayoutVersion":"1.0.0"}"#,
            )?;

            // Replace an image of the same name, as `docker load` does
            let path = self.path(name)?;
            if path.exists() {
                let old = tempfile::tempdir_in(&tmp)?;
                std::fs::rename(&path, old.path().join("image"))?;
            }
            std::fs::create_dir_all(path.parent().context("no parent")?)?;
            std::fs::rename(&layout, &path)?;
        }
        // The staging directory became the last image
        let _ = staging.keep();
        Ok(names)
    }
}
