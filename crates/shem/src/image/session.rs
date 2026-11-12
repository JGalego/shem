//! Sessions as artifacts: a model image's session, published apart from the
//! model and linked to it, as ModelPack's issue #239 proposes for model state.
//!
//! A model image keeps the session it was primed with (`WARM`) or saved in
//! (`docker commit`) in its state layer. `publish` makes that layer an
//! artifact of its own, whose `subject` is the model image without it: the
//! image as it was before `WARM`, byte for byte, so a registry lists the
//! sessions of a model among its referrers. The artifact's config says what
//! a runtime needs to decide whether it can load the session: its format,
//! the tokens it has read, its precision, and the model's fingerprint, the
//! digest of its blocks and their weights however its files are laid out.
//! `load` checks that, and unpacks the session for the driver to continue,
//! which changes nothing but speed, on any image of the same model.
//!
//! The media types are shem's own, while #239's are placeholders.

use std::path::Path;

use anyhow::{Context, bail};
use serde_json::Value;

use super::{
    Descriptor, Image, MANIFEST_TYPE, STATE_DIR, STATE_TYPE, ZSTD_LEVEL, digest, layer, put_blob,
};

pub const ARTIFACT_TYPE: &str = "application/vnd.shem.session.v1+json";
const CONFIG_TYPE: &str = "application/vnd.shem.session.config.v1+json";
const LAYER_TYPE: &str = "application/vnd.shem.session.v1.tar+zstd";
/// The format of the session's files, as the driver writes them
const FORMAT: &str = "shem.session/v1";

/// The identity of the model of `image`, unpacked at `root`, which a
/// session is for: the fingerprint its manifest records, or for a model
/// imported before fingerprints, the digest of the manifest itself.
pub fn model_digest(root: &Path, image: &Image) -> anyhow::Result<String> {
    let model = (image.config.model.as_deref())
        .context("not a model image: it has no dev.shem.model")?
        .trim_start_matches('/');
    let manifest = std::fs::read(root.join(model).join("shem.json"))
        .context("reading the model's shem.json")?;
    let parsed: Value = serde_json::from_slice(&manifest)?;
    Ok(match parsed["fingerprint"].as_str() {
        Some(fingerprint) => fingerprint.into(),
        None => digest(&manifest),
    })
}

/// Write the session of the model image `image`, whose filesystem is
/// unpacked at `root`, as an artifact in an OCI layout at `out` tagged
/// `tag`. Its subject is the image without its state, which the layout also
/// holds, untagged, so a push sends both. Returns the artifact's digest.
pub fn publish(image: &Image, root: &Path, out: &Path, tag: &str) -> anyhow::Result<String> {
    let state = (image.layers.iter())
        .rfind(|l| l.media_type == STATE_TYPE)
        .context("the image has no session: prime one with WARM --until=prompt, or docker commit a model container")?;
    let files = layer::files(&state.data)?;
    let meta: Value = serde_json::from_slice(
        files
            .get("session.json")
            .context("the image's state is no model session")?,
    )?;
    if meta["format"] != FORMAT {
        bail!("the image's session is not a {FORMAT} session");
    }

    // The model the session belongs to: the image as it was before WARM
    let model = Image {
        config: image.config.clone(),
        layers: (image.layers.iter())
            .filter(|l| l.media_type != STATE_TYPE)
            .cloned()
            .collect(),
        subject: image.subject.clone(),
    };
    model.write_layout(out, tag)?;
    let subject = Image::layout_descriptor(out)?;

    let blobs = out.join("blobs/sha256");
    let put = |data: &[u8]| -> anyhow::Result<(String, u64)> {
        let digest = digest(data);
        put_blob(&blobs, &digest, data)?;
        Ok((digest, data.len() as u64))
    };
    // The same blob as the image's own state layer
    let tar = layer::under(&state.data, STATE_DIR)?;
    let (layer_digest, layer_size) = put(&zstd::encode_all(&tar[..], ZSTD_LEVEL)?)?;
    let config = serde_json::json!({
        "format": FORMAT,
        "tokens": meta["tokens"].as_array().map_or(0, Vec::len),
        "precision": "float32",
        "model": model_digest(root, image)?,
    });
    let (config_digest, config_size) = put(&serde_json::to_vec(&config)?)?;
    let manifest = serde_json::json!({
        "schemaVersion": 2,
        "mediaType": MANIFEST_TYPE,
        "artifactType": ARTIFACT_TYPE,
        "config": {"mediaType": CONFIG_TYPE, "digest": config_digest, "size": config_size},
        "layers": [{"mediaType": LAYER_TYPE, "digest": layer_digest, "size": layer_size}],
        "subject": Descriptor { media_type: MANIFEST_TYPE.into(), ..subject },
    });
    let (digest, size) = put(&serde_json::to_vec(&manifest)?)?;
    let index = serde_json::json!({
        "schemaVersion": 2,
        "manifests": [{
            "mediaType": MANIFEST_TYPE,
            "artifactType": ARTIFACT_TYPE,
            "digest": digest,
            "size": size,
            "annotations": {"org.opencontainers.image.ref.name": tag},
        }],
    });
    std::fs::write(out.join("index.json"), serde_json::to_vec(&index)?)?;
    Ok(digest)
}

/// Unpack the session artifact in the OCI layout `dir` into `into`, as the
/// driver's `--session` reads it, if it is a session of the model whose
/// manifest has the digest `model`.
pub fn load(dir: &Path, model: &str, into: &Path) -> anyhow::Result<()> {
    let blob = |digest: &str| -> anyhow::Result<Vec<u8>> {
        let hex = digest
            .strip_prefix("sha256:")
            .context("unsupported digest")?;
        let data = std::fs::read(dir.join("blobs/sha256").join(hex))?;
        if super::digest(&data) != digest {
            bail!("blob {digest} does not match its digest");
        }
        Ok(data)
    };
    let index: Value = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
    let entry = index["manifests"][0]["digest"]
        .as_str()
        .context("the layout has no manifest")?;
    let manifest: Value = serde_json::from_slice(&blob(entry)?)?;
    if manifest["artifactType"] != ARTIFACT_TYPE {
        bail!(
            "not a shem session: its type is {}",
            manifest["artifactType"]
        );
    }
    let config: Value = serde_json::from_slice(&blob(
        manifest["config"]["digest"].as_str().unwrap_or_default(),
    )?)?;
    if config["format"] != FORMAT {
        bail!(
            "the session is in {}, and this shem reads {FORMAT}",
            config["format"]
        );
    }
    if config["model"] != model {
        bail!("the session is for another model: its blocks or weights are not this model's");
    }
    let layer = manifest["layers"][0]["digest"]
        .as_str()
        .context("the session has no layer")?;
    let tar = zstd::decode_all(&blob(layer)?[..])?;
    let flat = layer::strip(&tar, STATE_DIR)?.context("the session's layer holds no session")?;
    std::fs::create_dir_all(into)?;
    for (name, data) in layer::files(&flat)? {
        if name.contains('/') {
            bail!("the session holds a file outside it: {name}");
        }
        std::fs::write(into.join(name), data)?;
    }
    Ok(())
}
