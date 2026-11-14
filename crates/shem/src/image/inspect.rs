//! What an image is made of, as `shem inspect` and `shem history` show it:
//! read from its index, manifest and config alone, since layers can be
//! gigabytes of weights.

use std::path::Path;

use anyhow::Context;
use serde_json::{Value, json};

use super::{
    DOCKER_CONFIG_TYPE, Descriptor, History, Image, Index, Manifest, OCI_CONFIG_TYPE,
    layout_manifest, read_blob,
};

/// A step that made the image: a layer, or an instruction such as `ENV`
/// that made none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// The layer it made, if any
    pub layer: Option<Descriptor>,
    /// The instruction, empty where the image does not say
    pub created_by: String,
    pub comment: String,
}

/// The manifest of the layout in `dir`, and its config as JSON: null for a
/// config that is no JSON at all.
fn read(dir: &Path) -> anyhow::Result<(Manifest, Value)> {
    let index: Index = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
    let entry = index.manifests.first().context("no manifest")?;
    let manifest = layout_manifest(dir, entry)?;
    let config = serde_json::from_slice(&read_blob(dir, &manifest.config)?).unwrap_or_default();
    Ok((manifest, config))
}

/// The image in the layout `dir`, known by `references`, as `docker inspect`
/// shows one: its ID, what it runs, its size, its process settings and its
/// layers.
pub fn inspect(dir: &Path, references: &[String]) -> anyhow::Result<Value> {
    let (manifest, config) = read(dir)?;
    let summary = Image::layout_summary(dir)?;
    let process = match manifest.config.media_type.as_str() {
        OCI_CONFIG_TYPE | DOCKER_CONFIG_TYPE => config["config"].clone(),
        // shem's earlier format, or an artifact's, such as ModelPack's
        _ => config.clone(),
    };
    Ok(json!({
        "Id": Image::layout_digest(dir)?,
        "RepoTags": references,
        "Kind": summary.kind.to_string(),
        "Size": summary.size,
        "Created": config.get("created"),
        "Architecture": config.get("architecture"),
        "Os": config.get("os"),
        "Config": process,
        "ConfigMediaType": manifest.config.media_type,
        "Layers": manifest.layers,
        "Subject": manifest.subject,
    }))
}

/// The steps that made the image in the layout `dir`, oldest first. Where
/// its config's history does not account for every layer, each layer is a
/// step that does not say what it was.
pub fn history(dir: &Path) -> anyhow::Result<Vec<Step>> {
    let (manifest, config) = read(dir)?;
    let history: Vec<History> =
        serde_json::from_value(config["history"].clone()).unwrap_or_default();
    let made = history.iter().filter(|h| !h.empty_layer).count();
    if made != manifest.layers.len() {
        return Ok(manifest
            .layers
            .into_iter()
            .map(|layer| Step {
                layer: Some(layer),
                created_by: String::new(),
                comment: String::new(),
            })
            .collect());
    }
    let mut layers = manifest.layers.into_iter();
    Ok(history
        .into_iter()
        .map(|h| Step {
            layer: match h.empty_layer {
                true => None,
                false => layers.next(),
            },
            created_by: h.created_by.unwrap_or_default(),
            comment: h.comment.unwrap_or_default(),
        })
        .collect())
}
