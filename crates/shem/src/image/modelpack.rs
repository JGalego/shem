//! ModelPack artifacts, and the images that run them.
//!
//! [ModelPack](https://github.com/modelpack/model-spec) packages a model as
//! an OCI artifact: its files, one per layer, and a config that describes
//! the model, for an inference server to load. shem runs one through its
//! image twin: an OCI image manifest over the artifact's own layers, plus a
//! layer with what shem needs besides them (`shem.json`, and an index or a
//! chat template if the checkpoint has none), with the artifact as its
//! `subject`. A tar layer goes into the twin as the very same blob, so a
//! registry keeps the weights once for both. A raw layer is a file without
//! a tar around it, which no image can use, so the twin wraps it in one.
//!
//! The spec names its media types and annotations `cncf`; modctl and older
//! artifacts name them `cnai`. shem reads both.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde_json::Value;

use super::{
    Config, Descriptor, Image, Layer, MANIFEST_TYPE, ROOTFS_TYPE, Stored, digest, layer,
    layer_type, model,
};

/// The namespaces of ModelPack's media types: the spec's, and modctl's.
const NAMESPACES: [&str; 2] = ["cncf", "cnai"];

/// A ModelPack artifact in an OCI layout.
pub struct Artifact {
    dir: PathBuf,
    /// The artifact's manifest
    pub descriptor: Descriptor,
    /// Its model config: the `descriptor`, `config` and `modelfs` objects
    pub config: Value,
    layers: Vec<Descriptor>,
}

/// Whether `media_type` is a ModelPack artifact's type or config type.
fn is_modelpack(media_type: &str) -> bool {
    NAMESPACES.iter().any(|ns| {
        media_type == format!("application/vnd.{ns}.model.manifest.v1+json")
            || media_type == format!("application/vnd.{ns}.model.config.v1+json")
    })
}

/// Whether the OCI layout in `dir` holds a ModelPack artifact.
pub fn is_artifact(dir: &Path) -> bool {
    matches!(Artifact::read(dir), Ok(Some(_)))
}

/// A layer annotation, by its name in either namespace, such as `filepath`.
fn annotation<'a>(layer: &'a Descriptor, name: &str) -> Option<&'a str> {
    NAMESPACES
        .iter()
        .find_map(|ns| layer.annotations.get(&format!("org.{ns}.model.{name}")))
        .map(String::as_str)
}

impl Artifact {
    /// The ModelPack artifact the OCI layout in `dir` holds, or None if it
    /// holds something else.
    pub fn read(dir: &Path) -> anyhow::Result<Option<Self>> {
        let index: Value = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
        let descriptor: Descriptor = serde_json::from_value(
            index["manifests"]
                .get(0)
                .context("the layout has no manifest")?
                .clone(),
        )?;
        let manifest: Value = serde_json::from_slice(&blob(dir, &descriptor)?)?;
        let kind = manifest["artifactType"]
            .as_str()
            .or(manifest["config"]["mediaType"].as_str())
            .unwrap_or_default();
        if !is_modelpack(kind) {
            return Ok(None);
        }
        let config: Descriptor = serde_json::from_value(manifest["config"].clone())?;
        let config = serde_json::from_slice(&blob(dir, &config)?)?;
        let layers = serde_json::from_value(manifest["layers"].clone())?;
        Ok(Some(Self {
            dir: dir.to_path_buf(),
            descriptor,
            config,
            layers,
        }))
    }

    /// The artifact's layers as image layers: a tar layer as its own blob,
    /// and a raw one in a tar of its own, at its file path and with its mode.
    fn layers(&self) -> anyhow::Result<Vec<Layer>> {
        let mut layers = Vec::new();
        for descriptor in &self.layers {
            let format = (descriptor.media_type.rsplit_once(".v1."))
                .map(|(_, format)| format)
                .filter(|_| descriptor.media_type.contains(".model."))
                .with_context(|| format!("{} is no ModelPack layer", descriptor.media_type))?;
            let bytes = blob(&self.dir, descriptor)?;
            let data = match format {
                "raw" => {
                    let path = annotation(descriptor, "filepath")
                        .context("a raw layer has no file path")?;
                    let mode = annotation(descriptor, "file.metadata+json")
                        .and_then(|meta| serde_json::from_str::<Value>(meta).ok())
                        .and_then(|meta| meta["mode"].as_u64())
                        .unwrap_or(0o644);
                    layers.push(Layer::new(ROOTFS_TYPE, wrap(path, mode as u32, &bytes)?));
                    continue;
                }
                "tar" => bytes,
                "tar+gzip" => {
                    let mut out = Vec::new();
                    flate2::read::GzDecoder::new(&bytes[..]).read_to_end(&mut out)?;
                    out
                }
                "tar+zstd" => zstd::decode_all(&bytes[..])?,
                other => bail!("a ModelPack layer in {other} format, which shem doesn't read"),
            };
            let hex = &descriptor.digest["sha256:".len()..];
            layers.push(Layer {
                stored: Some(Stored {
                    media_type: layer_type(&descriptor.media_type).into(),
                    digest: descriptor.digest.clone(),
                    size: descriptor.size,
                    path: self.dir.join("blobs/sha256").join(hex),
                }),
                ..Layer::new(ROOTFS_TYPE, data)
            });
        }
        Ok(layers)
    }

    /// The image twin of the artifact, which `name` refers to, with its
    /// filesystem unpacked into `root`.
    pub fn twin(&self, name: &str, root: &Path) -> anyhow::Result<Image> {
        let format = self.config["config"]["format"].as_str().unwrap_or_default();
        if !format.is_empty() && !format.eq_ignore_ascii_case("safetensors") {
            bail!("{name} is a {format} model: shem runs safetensors checkpoints");
        }
        let mut layers = self.layers()?;
        for layer in &layers {
            layer::unpack(&layer.data, root)?;
        }
        // A ModelPack tar holds each file at its path in the model's
        // directory, so the model is at the root, unless it was packed in one
        let dir = ["", model::MODEL_DIR]
            .into_iter()
            .find(|dir| root.join(dir).join("config.json").is_file())
            .with_context(|| {
                format!("{name} has no config.json at its root or in /model: shem runs Hugging Face checkpoints")
            })?;
        let has_weights = std::fs::read_dir(root.join(dir))?
            .filter_map(Result::ok)
            .any(|entry| entry.path().extension().is_some_and(|e| e == "safetensors"));
        if !has_weights {
            bail!("{name} has no safetensors weights: shem runs safetensors checkpoints");
        }

        // What shem needs besides the checkpoint's files, laid over them
        let extra = tempfile::tempdir()?;
        model::import_in_place(&root.join(dir), extra.path())?;
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(extra.path())? {
            let entry = entry?;
            let path = Path::new(dir).join(entry.file_name());
            std::fs::copy(entry.path(), root.join(&path))?;
            paths.push(path);
        }
        layers.push(Layer::new(ROOTFS_TYPE, layer::create(root, &paths, &[])?));

        let config = Config {
            machine: "rv32im".into(),
            entrypoint: vec![model::DRIVER.into()],
            model: Some(format!("/{dir}")),
            labels: self.labels(name),
            ..Config::default()
        };
        Ok(Image {
            config,
            layers,
            subject: Some(Descriptor {
                media_type: MANIFEST_TYPE.into(),
                annotations: BTreeMap::new(),
                platform: None,
                ..self.descriptor.clone()
            }),
        })
    }

    /// The standard OCI labels the model config's descriptor fills in, and
    /// the artifact as the image's base.
    fn labels(&self, name: &str) -> BTreeMap<String, String> {
        let descriptor = &self.config["descriptor"];
        let text = |key: &str| descriptor[key].as_str().filter(|v| !v.is_empty());
        let list = |key: &str, separator: &str| {
            let items: Vec<&str> = descriptor[key]
                .as_array()
                .map(|items| items.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            (!items.is_empty()).then(|| items.join(separator))
        };
        let fields = [
            ("title", text("title").or(text("name")).map(String::from)),
            ("description", text("description").map(String::from)),
            ("version", text("version").map(String::from)),
            ("revision", text("revision").map(String::from)),
            ("vendor", text("vendor").map(String::from)),
            ("created", text("createdAt").map(String::from)),
            ("source", text("sourceURL").map(String::from)),
            ("documentation", text("docURL").map(String::from)),
            ("authors", list("authors", ", ")),
            ("licenses", list("licenses", " AND ")),
            ("base.name", Some(name.into())),
            ("base.digest", Some(self.descriptor.digest.clone())),
        ];
        fields
            .into_iter()
            .filter_map(|(key, value)| Some((format!("org.opencontainers.image.{key}"), value?)))
            .collect()
    }
}

/// The blob `descriptor` names in the layout in `dir`, checked.
fn blob(dir: &Path, descriptor: &Descriptor) -> anyhow::Result<Vec<u8>> {
    let hex = (descriptor.digest.strip_prefix("sha256:")).context("unsupported digest")?;
    let data = std::fs::read(dir.join("blobs/sha256").join(hex))
        .with_context(|| format!("reading blob {}", descriptor.digest))?;
    if digest(&data) != descriptor.digest {
        bail!("blob {} does not match its digest", descriptor.digest);
    }
    Ok(data)
}

/// A tar holding one file, `data`, at `path` with `mode`.
fn wrap(path: &str, mode: u32, data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let path = path.trim_start_matches('/');
    if path.split('/').any(|part| part == "..") {
        bail!("a ModelPack file path climbs out of the model: {path}");
    }
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_ustar();
    header.set_mtime(0);
    header.set_uid(0);
    header.set_gid(0);
    header.set_mode(mode & 0o7777);
    header.set_size(data.len() as u64);
    builder.append_data(&mut header, path, data)?;
    Ok(builder.into_inner()?)
}
