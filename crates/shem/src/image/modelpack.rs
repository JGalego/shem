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
//! The other way, `export` publishes a model image as an artifact: the
//! image's layers that hold the model's files, as they are, with a model
//! config filled in from `shem.json`, and the image as the artifact's
//! `subject`. Its layers are the image's blobs, its files where the image
//! has them, under `model/` for an imported model: compressed with zstd,
//! which the spec allows and modctl 0.2 can't extract. Written uncompressed,
//! they are new blobs, so their files go at the root, as modctl packs them.
//!
//! The spec names its media types and annotations `cncf`; modctl and older
//! artifacts name them `cnai`. shem reads both, and writes the spec's.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde_json::Value;

use super::{
    Config, Descriptor, Image, Layer, MANIFEST_TYPE, ROOTFS_TYPE, Stored, digest, layer,
    layer_type, model, put_blob,
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

/// The spec's names for what `export` writes.
const ARTIFACT_TYPE: &str = "application/vnd.cncf.model.manifest.v1+json";
const CONFIG_TYPE: &str = "application/vnd.cncf.model.config.v1+json";
const FILEPATH: &str = "org.cncf.model.filepath";
/// Top-level directories of an image that hold no model, wherever the model is.
const NOT_MODEL: [&str; 17] = [
    "compiled", ".shem", "usr", "bin", "sbin", "lib", "lib64", "etc", "opt", "var", "tmp", "home",
    "root", "dev", "proc", "sys", "run",
];

/// Write the model of `image`, whose layout is `layout`, as a ModelPack
/// artifact, in an OCI layout at `out` tagged `tag`: the image's layers
/// whose files are all the model's, typed by what they hold, and a model
/// config from the model's manifest, which the image's filesystem, unpacked
/// at `root`, holds. Layers keep their blobs, unless `uncompressed`, which
/// writes each as a plain tar. The image is the artifact's `subject`, so the
/// layout holds it too, untagged, and a push sends both. Returns the
/// artifact's manifest digest.
pub fn export(
    image: &Image,
    layout: &Path,
    root: &Path,
    out: &Path,
    tag: &str,
    uncompressed: bool,
) -> anyhow::Result<String> {
    let subject = Image::layout_descriptor(layout)?;
    let model = (image.config.model.as_deref())
        .context("not a model image: it has no dev.shem.model")?
        .trim_matches('/');
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(root.join(model).join("shem.json"))
            .context("reading the model's shem.json")?,
    )?;
    let index = manifest["weights"]
        .as_str()
        .context("the model's manifest is shem.model/v1: rebuild the image")?;
    let weight_map: Value = serde_json::from_slice(&std::fs::read(root.join(model).join(index))?)?;
    let mut needed: Vec<String> = vec!["shem.json".into(), index.into()];
    needed.extend(
        (weight_map["weight_map"].as_object().into_iter().flatten())
            .filter_map(|(_, f)| f.as_str().map(String::from)),
    );
    for key in ["file", "chat_template"] {
        needed.extend(manifest["tokenizer"][key].as_str().map(String::from));
    }

    let blobs = out.join("blobs/sha256");
    std::fs::create_dir_all(&blobs)?;
    // The image the artifact names as its subject, links to its blobs
    for entry in std::fs::read_dir(layout.join("blobs/sha256"))? {
        let entry = entry?;
        let target = blobs.join(entry.file_name());
        if std::fs::hard_link(entry.path(), &target).is_err() {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    let put = |data: &[u8]| -> anyhow::Result<(String, u64)> {
        let digest = digest(data);
        put_blob(&blobs, &digest, data)?;
        Ok((digest, data.len() as u64))
    };
    let mut layers = Vec::new();
    let mut diff_ids = Vec::new();
    let mut covered = Vec::new();
    for layer in image.layers.iter().filter(|l| l.media_type == ROOTFS_TYPE) {
        let Some(files) = model_files(&layer.data, model)? else {
            continue;
        };
        let kind = if files.iter().all(|f| f.ends_with(".safetensors")) {
            "weight"
        } else if files.iter().all(|f| is_doc(f)) {
            "doc"
        } else {
            "weight.config"
        };
        // Uncompressed, the layer is a new blob anyway, so its files go at the
        // root, as modctl packs them; shared, they stay where the image has them
        let tar = if uncompressed {
            rebase(&layer.data, model)?
        } else {
            layer.data.clone()
        };
        let (format, digest, size) = match &layer.stored {
            Some(stored) if !uncompressed => {
                let target = blobs.join(&stored.digest["sha256:".len()..]);
                if !target.exists() && std::fs::hard_link(&stored.path, &target).is_err() {
                    std::fs::copy(&stored.path, &target)?;
                }
                let format = match stored.media_type.rsplit_once('+') {
                    Some((_, compression)) => format!("tar+{compression}"),
                    None => "tar".into(),
                };
                (format, stored.digest.clone(), stored.size)
            }
            _ => {
                let (digest, size) = put(&tar)?;
                ("tar".into(), digest, size)
            }
        };
        let mut annotations = BTreeMap::new();
        if let [file] = files.as_slice() {
            let path = if uncompressed {
                Path::new(file).to_path_buf()
            } else {
                Path::new(model).join(file)
            };
            annotations.insert(FILEPATH.into(), path.to_string_lossy().into_owned());
        }
        layers.push(Descriptor {
            media_type: format!("application/vnd.cncf.model.{kind}.v1.{format}"),
            digest,
            size,
            annotations,
            platform: None,
        });
        diff_ids.push(super::digest(&tar));
        covered.extend(files);
    }
    if let Some(missing) = needed.iter().find(|f| !covered.contains(f)) {
        bail!(
            "the model's {missing} shares a layer with files outside the model, so no artifact can hold it alone"
        );
    }

    let config = model_config(&manifest, image, diff_ids);
    let (config_digest, config_size) = put(&serde_json::to_vec(&config)?)?;
    let artifact = serde_json::json!({
        "schemaVersion": 2,
        "mediaType": MANIFEST_TYPE,
        "artifactType": ARTIFACT_TYPE,
        "config": {"mediaType": CONFIG_TYPE, "digest": config_digest, "size": config_size},
        "layers": layers,
        "subject": subject,
    });
    let (digest, size) = put(&serde_json::to_vec(&artifact)?)?;
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
    std::fs::write(out.join("oci-layout"), br#"{"imageLayoutVersion":"1.0.0"}"#)?;
    Ok(digest)
}

/// The regular files of a layer, by their paths in the model's directory
/// `model`, if all of them are the model's; None if any is elsewhere, or
/// the layer has no files.
fn model_files(layer: &[u8], model: &str) -> anyhow::Result<Option<Vec<String>>> {
    let mut files = Vec::new();
    for entry in tar::Archive::new(layer).entries()? {
        let entry = entry?;
        let path = entry
            .path()?
            .to_string_lossy()
            .trim_start_matches("./")
            .to_string();
        let path = path.trim_end_matches('/');
        let inside = if model.is_empty() {
            Some(path)
        } else if path == model {
            Some("")
        } else {
            path.strip_prefix(model)
                .and_then(|rest| rest.strip_prefix('/'))
        };
        let Some(inside) = inside else {
            return Ok(None);
        };
        let top = inside.split('/').next().unwrap_or_default();
        if NOT_MODEL.contains(&top) {
            return Ok(None);
        }
        if entry.header().entry_type().is_file() {
            files.push(inside.to_string());
        }
    }
    Ok((!files.is_empty()).then_some(files))
}

/// The layer `tar` with its entries moved from the model's directory
/// `model` to the root, and the entry for the directory itself dropped.
fn rebase(tar: &[u8], model: &str) -> anyhow::Result<Vec<u8>> {
    if model.is_empty() {
        return Ok(tar.to_vec());
    }
    let mut builder = tar::Builder::new(Vec::new());
    for entry in tar::Archive::new(tar).entries()? {
        let mut entry = entry?;
        let path = entry
            .path()?
            .to_string_lossy()
            .trim_start_matches("./")
            .to_string();
        let path = path.trim_end_matches('/');
        let Some(inside) = path
            .strip_prefix(model)
            .and_then(|rest| rest.strip_prefix('/'))
        else {
            continue;
        };
        let mut header = entry.header().clone();
        let mut data = Vec::new();
        entry.read_to_end(&mut data)?;
        builder.append_data(&mut header, inside, &data[..])?;
    }
    Ok(builder.into_inner()?)
}

fn is_doc(file: &str) -> bool {
    let name = file.rsplit('/').next().unwrap_or(file);
    ["LICENSE", "NOTICE", "README"]
        .iter()
        .any(|doc| name.starts_with(doc))
}

/// A model's parameter count as ModelPack writes sizes: a count with at
/// most one decimal and a scale, such as `596m` or `1.5b`.
pub fn param_size(count: u64) -> String {
    let scales = [(1e12, "t"), (1e9, "b"), (1e6, "m"), (1e3, "k")];
    let (scale, suffix) = scales
        .into_iter()
        .find(|(scale, _)| count as f64 >= *scale)
        .unwrap_or((1.0, ""));
    let value = format!("{:.1}", count as f64 / scale);
    format!("{}{suffix}", value.trim_end_matches(".0"))
}

/// The ModelPack config for a model, from its manifest's `source`,
/// `parameters` and `precision`, and the image's labels.
fn model_config(manifest: &Value, image: &Image, diff_ids: Vec<String>) -> Value {
    let source = &manifest["source"];
    let repo = source["repo"].as_str();
    let labels = &image.config.labels;
    let label = |key: &str| {
        labels
            .get(&format!("org.opencontainers.image.{key}"))
            .cloned()
    };
    let mut descriptor = serde_json::Map::new();
    let mut set = |key: &str, value: Option<String>| {
        if let Some(value) = value.filter(|v| !v.is_empty()) {
            descriptor.insert(key.into(), value.into());
        }
    };
    let name = repo
        .and_then(|r| r.rsplit('/').next())
        .map(String::from)
        .or(label("title"));
    set("name", name);
    set("family", source["family"].as_str().map(String::from));
    set(
        "vendor",
        repo.and_then(|r| r.split_once('/'))
            .map(|(org, _)| org.to_string())
            .or(label("vendor")),
    );
    set(
        "revision",
        source["revision"]
            .as_str()
            .map(String::from)
            .or(label("revision")),
    );
    let url = repo.map(|r| format!("https://huggingface.co/{r}"));
    set("sourceURL", url.clone().or(label("source")));
    set("docURL", url.or(label("documentation")));
    set("description", label("description"));
    set("version", label("version"));
    let licenses = (source["spdx"].as_str().map(String::from)).or(label("licenses"));
    if let Some(licenses) = licenses {
        descriptor.insert("licenses".into(), serde_json::json!([licenses]));
    }
    let mut config = serde_json::json!({
        "architecture": "transformer",
        "format": "safetensors",
        "capabilities": {"inputTypes": ["text"], "outputTypes": ["text"]},
    });
    if let Some(count) = manifest["parameters"].as_u64() {
        config["paramSize"] = param_size(count).into();
    }
    let precision: Vec<&str> = (manifest["precision"].as_array().into_iter().flatten())
        .filter_map(Value::as_str)
        .collect();
    if !precision.is_empty() {
        config["precision"] = precision.join(",").into();
    }
    serde_json::json!({
        "descriptor": descriptor,
        "config": config,
        "modelfs": {"type": "layers", "diffIds": diff_ids},
    })
}
