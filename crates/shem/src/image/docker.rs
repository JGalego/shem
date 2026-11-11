//! Images for Docker: a shem image as a standard OCI image for `linux/riscv32`,
//! which `docker load` takes and `docker run --runtime io.containerd.shem.v1`
//! runs (see `docs/containerd.md`).
//!
//! Docker and containerd only unpack standard layers, so a warm image's state
//! layer becomes an ordinary layer holding its files under `/.shem`, which the
//! containerd shim thaws instead of starting the entrypoint.

use std::collections::BTreeMap;

use anyhow::bail;
use serde_json::json;

use super::{Image, MANIFEST_TYPE, REF_NAME, ROOTFS_TYPE, STATE_TYPE, ZSTD_LEVEL, digest, layer};

/// Where a warm image's state lives in its Docker image's filesystem.
pub const WARM_DIR: &str = ".shem";
const CONFIG_TYPE: &str = "application/vnd.oci.image.config.v1+json";
const LAYER_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+zstd";
const INDEX_TYPE: &str = "application/vnd.oci.image.index.v1+json";

impl Image {
    /// The image as an OCI image archive, a tar of an image layout, named
    /// `name` (such as `hello:latest`) for `docker load`. The archive is
    /// reproducible: the same image gives the same bytes.
    pub fn docker_archive(&self, name: &str) -> anyhow::Result<Vec<u8>> {
        if self.config.kernel.is_some() {
            bail!("VM images cannot run under Docker yet: the shim runs containers");
        }
        let mut blobs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut put = |data: Vec<u8>| -> serde_json::Value {
            let (hash, size) = (digest(&data), data.len());
            blobs.insert(hash.clone(), data);
            json!({ "digest": hash, "size": size })
        };

        let mut layers = Vec::new();
        let mut diff_ids = Vec::new();
        for layer in &self.layers {
            let tar = match layer.media_type.as_str() {
                ROOTFS_TYPE => layer.data.clone(),
                STATE_TYPE => warm_layer(&layer.data)?,
                other => bail!("cannot export a layer of type {other}"),
            };
            diff_ids.push(digest(&tar));
            let mut descriptor = put(zstd::encode_all(&tar[..], ZSTD_LEVEL)?);
            descriptor["mediaType"] = LAYER_TYPE.into();
            layers.push(descriptor);
        }

        let config = &self.config;
        let mut container = json!({
            "Entrypoint": config.entrypoint,
            "Cmd": config.cmd,
            "Env": config.env,
            "Labels": config.labels,
        });
        if !config.workdir.is_empty() {
            container["WorkingDir"] = config.workdir.clone().into();
        }
        let image_config = json!({
            "created": "1970-01-01T00:00:00Z",
            "architecture": "riscv32",
            "os": "linux",
            "config": container,
            "rootfs": { "type": "layers", "diff_ids": diff_ids },
        });
        let mut config_descriptor = put(serde_json::to_vec(&image_config)?);
        config_descriptor["mediaType"] = CONFIG_TYPE.into();
        let manifest = json!({
            "schemaVersion": 2,
            "mediaType": MANIFEST_TYPE,
            "config": config_descriptor,
            "layers": layers,
        });
        let mut manifest_descriptor = put(serde_json::to_vec(&manifest)?);
        let (_, tag) = super::parse_reference(name)?;
        manifest_descriptor["mediaType"] = MANIFEST_TYPE.into();
        manifest_descriptor["platform"] = json!({ "architecture": "riscv32", "os": "linux" });
        manifest_descriptor["annotations"] = json!({
            "io.containerd.image.name": docker_name(name)?,
            REF_NAME: tag,
        });
        let index = json!({
            "schemaVersion": 2,
            "mediaType": INDEX_TYPE,
            "manifests": [manifest_descriptor],
        });

        let mut files: Vec<(String, Vec<u8>)> = vec![
            (
                "oci-layout".into(),
                br#"{"imageLayoutVersion":"1.0.0"}"#.to_vec(),
            ),
            ("index.json".into(), serde_json::to_vec(&index)?),
        ];
        for (hash, data) in blobs {
            files.push((format!("blobs/sha256/{}", &hash["sha256:".len()..]), data));
        }
        let files: Vec<(&str, &[u8])> = files
            .iter()
            .map(|(name, data)| (name.as_str(), data.as_slice()))
            .collect();
        layer::from_files(&files)
    }
}

/// A state layer's files as an ordinary layer, under `WARM_DIR`.
fn warm_layer(state: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_ustar();
    header.set_entry_type(tar::EntryType::Directory);
    header.set_mtime(0);
    header.set_mode(0o700);
    header.set_size(0);
    builder.append_data(&mut header, format!("{WARM_DIR}/"), std::io::empty())?;
    for (name, data) in layer::files(state)? {
        let mut header = tar::Header::new_ustar();
        header.set_mtime(0);
        header.set_mode(0o600);
        header.set_size(data.len() as u64);
        builder.append_data(&mut header, format!("{WARM_DIR}/{name}"), &data[..])?;
    }
    Ok(builder.into_inner()?)
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
