//! Images in an [OCI image layout], with a guest filesystem, runtime config and optional
//! frozen process, VM or model-session state.
//!
//! Layouts use `linux/riscv32`, `dev.shem.*` config labels and a final state
//! layer under `/.shem`. In memory, state files use a [`STATE_TYPE`] layer
//! without that directory prefix. Legacy config and state media types remain
//! readable. See `docs/shemfile.md` for the format and build rules.
//!
//! [OCI image layout]: https://github.com/opencontainers/image-spec/blob/main/image-layout.md

mod build;
mod docker;
mod layer;
pub mod model;
pub mod modelpack;
mod registry;
pub mod session;
pub mod shemfile;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use build::{BuildOptions, CONTAINER_MEM, VM_MEM, build};
pub use docker::WARM_DIR;
pub use layer::unpack;
pub use registry::{copy_to_layout, is_local, pull, push, registry_host};

use crate::container::{Container, Frozen, FrozenHost};
use crate::linux::{RootFs, Stdio};
use crate::vcpu::{MachineState, Vcpu};
use crate::vm::{self, Console, FrozenMachine, Machine};

pub const MANIFEST_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
pub const INDEX_TYPE: &str = "application/vnd.oci.image.index.v1+json";
pub const OCI_CONFIG_TYPE: &str = "application/vnd.oci.image.config.v1+json";
/// The config media type of shem's earlier image format, still read.
pub const CONFIG_TYPE: &str = "application/vnd.shem.config.v1+json";
const LAYER_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+zstd";
/// The media types of OCI image layers: a tar, and a tar compressed with gzip
const TAR_TYPE: &str = "application/vnd.oci.image.layer.v1.tar";
const GZIP_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+gzip";
/// The config media type of images Docker builds without the OCI exporter.
const DOCKER_CONFIG_TYPE: &str = "application/vnd.docker.container.image.v1+json";
/// The prefix of the labels that hold shem's own settings.
const LABEL: &str = "dev.shem.";
/// Where an image's state lives in its filesystem.
pub const STATE_DIR: &str = ".shem";
pub const ROOTFS_TYPE: &str = "application/vnd.oci.image.layer.v1.tar";
pub const STATE_TYPE: &str = "application/vnd.shem.state.v1.tar";
const REF_NAME: &str = "org.opencontainers.image.ref.name";
/// Layers are stored compressed with zstd, their media types suffixed as OCI
/// has it; in memory they are plain tars.
const COMPRESSED: &str = "+zstd";
const ZSTD_LEVEL: i32 = 3;
const DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

/// The labels a config's shem settings become, by field name.
const SETTINGS: [&str; 5] = ["machine", "memory", "kernel", "bootargs", "model"];

impl Config {
    /// The OCI image config: platform, process, and shem's own settings as
    /// labels, with the layers' uncompressed digests.
    fn to_oci(&self, diff_ids: Vec<String>) -> serde_json::Value {
        let mut labels = self.labels.clone();
        let settings = [
            Some(self.machine.clone()),
            self.memory.map(|m| m.to_string()),
            self.kernel.clone(),
            self.bootargs.clone(),
            self.model.clone(),
        ];
        for (name, value) in SETTINGS.iter().zip(settings) {
            if let Some(value) = value.filter(|v| !v.is_empty()) {
                labels.insert(format!("{LABEL}{name}"), value);
            }
        }
        let mut process = serde_json::json!({
            "Entrypoint": self.entrypoint,
            "Cmd": self.cmd,
            "Env": self.env,
            "Labels": labels,
        });
        if !self.workdir.is_empty() {
            process["WorkingDir"] = self.workdir.clone().into();
        }
        serde_json::json!({
            "created": "1970-01-01T00:00:00Z",
            "architecture": "riscv32",
            "os": "linux",
            "config": process,
            "rootfs": { "type": "layers", "diff_ids": diff_ids },
        })
    }

    fn from_oci(oci: &serde_json::Value) -> anyhow::Result<Self> {
        let process = &oci["config"];
        let strings = |value: &serde_json::Value| -> Vec<String> {
            value
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut labels: BTreeMap<String, String> =
            serde_json::from_value(process["Labels"].clone()).unwrap_or_default();
        let mut setting = |name: &str| labels.remove(&format!("{LABEL}{name}"));
        let machine = setting("machine").unwrap_or_else(|| "rv32im".into());
        let memory = setting("memory").map(|m| m.parse()).transpose()?;
        let (kernel, bootargs, model) = (setting("kernel"), setting("bootargs"), setting("model"));
        Ok(Self {
            machine,
            entrypoint: strings(&process["Entrypoint"]),
            cmd: strings(&process["Cmd"]),
            env: strings(&process["Env"]),
            workdir: process["WorkingDir"].as_str().unwrap_or_default().into(),
            labels,
            memory,
            kernel,
            bootargs,
            model,
        })
    }
}

/// How to run an image.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// The instruction set the image's programs and state need.
    pub machine: String,
    pub entrypoint: Vec<String>,
    pub cmd: Vec<String>,
    pub env: Vec<String>,
    pub workdir: String,
    pub labels: BTreeMap<String, String>,
    /// Guest memory for a warm image, which must resume with the memory it
    /// was frozen with.
    pub memory: Option<u32>,
    /// A VM image's kernel, a path in its filesystem; set by `KERNEL`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
    /// The kernel command line, if not the default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootargs: Option<String>,
    /// A model image's model: the directory holding its `shem.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// A layer: its kind (a filesystem layer or shem's state) and its tar.
#[derive(Debug, Clone)]
pub struct Layer {
    pub media_type: String,
    pub data: Vec<u8>,
    /// The blob the layer was read from, which writing the image copies
    /// rather than compressing the tar again: a layer an image shares with
    /// a ModelPack artifact, or with an image Docker built, stays one blob
    /// in stores and registries.
    pub stored: Option<Stored>,
}

impl Layer {
    pub fn new(media_type: &str, data: Vec<u8>) -> Self {
        Self {
            media_type: media_type.into(),
            data,
            stored: None,
        }
    }
}

/// Layers are equal when they hold the same files, however they are stored.
impl PartialEq for Layer {
    fn eq(&self, other: &Self) -> bool {
        self.media_type == other.media_type && self.data == other.data
    }
}

impl Eq for Layer {}

/// A layer's blob in an OCI layout, which holds the layer's tar or that tar
/// compressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stored {
    /// Its OCI media type: a tar, or one compressed with gzip or zstd
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Image {
    pub config: Config,
    pub layers: Vec<Layer>,
    /// The manifest this image's manifest refers to, such as the ModelPack
    /// artifact an image twin runs, which a registry lists it under
    pub subject: Option<Descriptor>,
}

/// An OCI content descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Descriptor {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub annotations: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub platform: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    #[serde(rename = "mediaType")]
    media_type: String,
    config: Descriptor,
    layers: Vec<Descriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    subject: Option<Descriptor>,
}

/// The OCI image layer media type for a layer stored as `media_type`, which
/// may be Docker's or ModelPack's: the same bytes, by their compression.
fn layer_type(media_type: &str) -> &'static str {
    if media_type.ends_with("gzip") {
        GZIP_TYPE
    } else if media_type.ends_with("zstd") {
        LAYER_TYPE
    } else {
        TAR_TYPE
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Index {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    manifests: Vec<Descriptor>,
}

/// A platform as `os/architecture[/variant]`, such as `linux/arm64/v8`, as
/// an OCI descriptor's platform.
fn parse_platform(text: &str) -> anyhow::Result<serde_json::Value> {
    let parts: Vec<&str> = text.split('/').collect();
    let mut platform = match parts.as_slice() {
        [os, architecture] | [os, architecture, _]
            if !os.is_empty() && !architecture.is_empty() =>
        {
            serde_json::json!({ "os": os, "architecture": architecture })
        }
        _ => bail!("{text:?} is no platform: give os/architecture, such as linux/amd64"),
    };
    if let [_, _, variant] = parts.as_slice() {
        platform["variant"] = (*variant).into();
    }
    Ok(platform)
}

pub fn digest(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256:{hex}")
}

/// Write `data`, whose digest is `digest`, into a layout's `blobs` directory.
/// A blob is named by its content, so one already there holds these very
/// bytes, and may be a read-only link to a stored blob, as oras leaves them:
/// it stays.
fn put_blob(blobs: &Path, digest: &str, data: &[u8]) -> anyhow::Result<()> {
    let target = blobs.join(&digest["sha256:".len()..]);
    if !target.exists() {
        std::fs::write(target, data)?;
    }
    Ok(())
}

impl Image {
    /// The files of the state layer, if this is a warm image.
    fn state_files(&self) -> anyhow::Result<Option<BTreeMap<String, Vec<u8>>>> {
        let Some(layer) = self
            .layers
            .iter()
            .rev()
            .find(|l| l.media_type == STATE_TYPE)
        else {
            return Ok(None);
        };
        Ok(Some(layer::files(&layer.data)?))
    }

    /// The frozen container of a warm image, if this is one.
    pub fn warm_state(&self) -> anyhow::Result<Option<Frozen>> {
        let Some(files) = self.state_files()? else {
            return Ok(None);
        };
        let host = files
            .get("container.json")
            .context("state layer has no container.json")?;
        let machine = files
            .get("machine.state")
            .context("state layer has no machine.state")?;
        Ok(Some(Frozen {
            host: serde_json::from_slice::<FrozenHost>(host)?,
            machine: MachineState::decode(machine)?,
        }))
    }

    /// The frozen machine of a warm VM image, if this is one.
    pub fn warm_machine(&self) -> anyhow::Result<Option<vm::Frozen>> {
        let Some(files) = self.state_files()? else {
            return Ok(None);
        };
        let host = files.get("vm.json").context("state layer has no vm.json")?;
        let machine = files
            .get("machine.state")
            .context("state layer has no machine.state")?;
        Ok(Some(vm::Frozen {
            host: serde_json::from_slice::<FrozenMachine>(host)?,
            machine: MachineState::decode(machine)?,
        }))
    }

    /// Unpack the filesystem layers into `root`, in order.
    pub fn unpack_rootfs(&self, root: &Path) -> anyhow::Result<()> {
        for layer in self.layers.iter().filter(|l| l.media_type == ROOTFS_TYPE) {
            unpack(&layer.data, root)?;
        }
        Ok(())
    }

    /// Write the image as an OCI image layout in `dir`, tagged `tag`. Returns
    /// the manifest's digest.
    pub fn write_layout(&self, dir: &Path, tag: &str) -> anyhow::Result<String> {
        self.write_layout_for(dir, tag, &[])
    }

    /// Write the layout as `write_layout` does, listing the image under the
    /// platforms `also` (such as `linux/amd64`) as well as `linux/riscv32`,
    /// in an image index. The guest is RV32 whichever entry a runtime picks,
    /// so this lets runtimes that pick images by the host's platform, such
    /// as Kubernetes with released containerd, find shem images. Returns the
    /// digest of the index, or of the manifest when there are no others.
    pub fn write_layout_for(
        &self,
        dir: &Path,
        tag: &str,
        also: &[String],
    ) -> anyhow::Result<String> {
        let blobs = dir.join("blobs/sha256");
        std::fs::create_dir_all(&blobs)?;
        let put = |media_type: &str, data: &[u8]| -> anyhow::Result<Descriptor> {
            let digest = digest(data);
            put_blob(&blobs, &digest, data)?;
            Ok(Descriptor {
                media_type: media_type.into(),
                digest,
                size: data.len() as u64,
                annotations: BTreeMap::new(),
                platform: None,
            })
        };
        let mut layers = Vec::new();
        let mut diff_ids = Vec::new();
        for layer in &self.layers {
            let tar = match layer.media_type.as_str() {
                STATE_TYPE => layer::under(&layer.data, STATE_DIR)?,
                _ => layer.data.clone(),
            };
            diff_ids.push(digest(&tar));
            layers.push(match &layer.stored {
                Some(stored) => {
                    let target = blobs.join(&stored.digest["sha256:".len()..]);
                    // A link where it can be one: weights are gigabytes
                    if !target.exists() && std::fs::hard_link(&stored.path, &target).is_err() {
                        std::fs::copy(&stored.path, &target).with_context(|| {
                            format!(
                                "copying layer {} from {}",
                                stored.digest,
                                stored.path.display()
                            )
                        })?;
                    }
                    Descriptor {
                        media_type: stored.media_type.clone(),
                        digest: stored.digest.clone(),
                        size: stored.size,
                        annotations: BTreeMap::new(),
                        platform: None,
                    }
                }
                None => put(LAYER_TYPE, &zstd::encode_all(&tar[..], ZSTD_LEVEL)?)?,
            });
        }
        let config = put(
            OCI_CONFIG_TYPE,
            &serde_json::to_vec(&self.config.to_oci(diff_ids))?,
        )?;
        let manifest = Manifest {
            schema_version: 2,
            media_type: MANIFEST_TYPE.into(),
            config,
            layers,
            subject: self.subject.clone(),
        };
        let mut descriptor = put(MANIFEST_TYPE, &serde_json::to_vec(&manifest)?)?;
        descriptor.platform = Some(serde_json::json!({ "architecture": "riscv32", "os": "linux" }));
        if !also.is_empty() {
            let mut entries = vec![descriptor.clone()];
            for platform in also {
                entries.push(Descriptor {
                    platform: Some(parse_platform(platform)?),
                    ..descriptor.clone()
                });
            }
            let index = serde_json::json!({
                "schemaVersion": 2,
                "mediaType": INDEX_TYPE,
                "manifests": entries,
            });
            descriptor = put(INDEX_TYPE, &serde_json::to_vec(&index)?)?;
        }
        descriptor.annotations.insert(REF_NAME.into(), tag.into());
        let index = Index {
            schema_version: 2,
            manifests: vec![descriptor.clone()],
        };
        std::fs::write(dir.join("index.json"), serde_json::to_vec(&index)?)?;
        std::fs::write(dir.join("oci-layout"), br#"{"imageLayoutVersion":"1.0.0"}"#)?;
        Ok(descriptor.digest)
    }

    /// Read the image an OCI image layout holds.
    pub fn read_layout(dir: &Path) -> anyhow::Result<Self> {
        let blob = |descriptor: &Descriptor| -> anyhow::Result<Vec<u8>> {
            let hex = descriptor
                .digest
                .strip_prefix("sha256:")
                .context("unsupported digest")?;
            let data = std::fs::read(dir.join("blobs/sha256").join(hex))?;
            if digest(&data) != descriptor.digest {
                bail!("blob {} does not match its digest", descriptor.digest);
            }
            Ok(data)
        };
        let index: Index = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
        let mut entry = index
            .manifests
            .first()
            .context("the image layout has no manifest")?
            .clone();
        // An index, such as the manifest list BuildKit writes beside its
        // attestations, holds the image for linux/riscv32 among others
        let mut data = blob(&entry)?;
        while let Ok(nested) = serde_json::from_slice::<Index>(&data) {
            let riscv = |d: &&Descriptor| {
                d.platform
                    .as_ref()
                    .is_some_and(|p| p["architecture"] == "riscv32")
            };
            entry = (nested.manifests.iter().find(riscv))
                .or(nested.manifests.first())
                .context("an index in the image layout lists no manifests")?
                .clone();
            data = blob(&entry)?;
        }
        let manifest: Manifest = serde_json::from_slice(&data)?;
        let config = match manifest.config.media_type.as_str() {
            OCI_CONFIG_TYPE | DOCKER_CONFIG_TYPE => {
                Config::from_oci(&serde_json::from_slice(&blob(&manifest.config)?)?)?
            }
            CONFIG_TYPE => serde_json::from_slice(&blob(&manifest.config)?)?,
            other => bail!("not an image: its config is {other}"),
        };
        let mut layers: Vec<Layer> = manifest
            .layers
            .iter()
            .map(|d| {
                let data = blob(d)?;
                let gzip = d.media_type.ends_with("gzip");
                let compressed = d.media_type.ends_with(COMPRESSED) || gzip;
                let data = if gzip {
                    let mut out = Vec::new();
                    std::io::Read::read_to_end(
                        &mut flate2::read::GzDecoder::new(&data[..]),
                        &mut out,
                    )?;
                    out
                } else if compressed {
                    zstd::decode_all(&data[..])?
                } else {
                    data
                };
                if d.media_type.starts_with(STATE_TYPE) {
                    return Ok(Layer::new(STATE_TYPE, data));
                }
                let hex = d
                    .digest
                    .strip_prefix("sha256:")
                    .context("unsupported digest")?;
                Ok(Layer {
                    stored: Some(Stored {
                        media_type: layer_type(&d.media_type).into(),
                        digest: d.digest.clone(),
                        size: d.size,
                        path: dir.join("blobs/sha256").join(hex),
                    }),
                    ..Layer::new(ROOTFS_TYPE, data)
                })
            })
            .collect::<anyhow::Result<_>>()?;
        // A last layer that holds nothing but /.shem is the state
        if let Some(last) = layers.last_mut()
            && last.media_type == ROOTFS_TYPE
            && let Some(state) = layer::strip(&last.data, STATE_DIR)?
        {
            *last = Layer::new(STATE_TYPE, state);
        }
        Ok(Self {
            config,
            layers,
            subject: manifest.subject,
        })
    }

    /// The manifest digest of the image's layout in `dir`.
    pub fn layout_digest(dir: &Path) -> anyhow::Result<String> {
        Ok(Self::layout_descriptor(dir)?.digest)
    }

    /// The descriptor of the manifest (or index) the layout in `dir` holds,
    /// as another manifest's `subject` names it.
    pub fn layout_descriptor(dir: &Path) -> anyhow::Result<Descriptor> {
        let index: Index = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
        let entry = index.manifests.into_iter().next().context("no manifest")?;
        Ok(Descriptor {
            annotations: BTreeMap::new(),
            platform: None,
            ..entry
        })
    }
}

/// Start a container from `image`, its filesystem unpacked into `root`. A warm
/// image resumes where `WARM` froze it, and takes no arguments; otherwise the
/// entrypoint starts, with `args` in place of the image's CMD if given.
/// `vcpu` makes a vCPU with the given memory: a warm image's own size, or
/// `mem` for a cold start.
pub fn start<V: Vcpu + Clone>(
    image: &Image,
    vcpu: impl FnOnce(u32) -> anyhow::Result<V>,
    args: &[String],
    env: &[String],
    stdio: Stdio,
    root: &Path,
    mem: u32,
) -> anyhow::Result<Container<V>> {
    image.unpack_rootfs(root)?;
    let config = &image.config;
    let mut container = match image.warm_state()? {
        Some(frozen) => {
            if !args.is_empty() || !env.is_empty() {
                bail!(
                    "a warm image resumes where it was frozen, so it takes no arguments or environment"
                );
            }
            Container::thaw(vcpu(frozen.machine.mem.len() as u32)?, &frozen, stdio)?
        }
        None => {
            let cmd = if args.is_empty() { &config.cmd } else { args };
            let argv: Vec<String> = config.entrypoint.iter().chain(cmd).cloned().collect();
            let name = argv
                .first()
                .context("the image has no ENTRYPOINT or CMD, and no command was given")?;
            let mut environment = config.env.clone();
            for var in env {
                let key = var.split_once('=').map_or(var.as_str(), |(k, _)| k);
                environment.retain(|v| v.split_once('=').map(|(k, _)| k) != Some(key));
                environment.push(var.clone());
            }
            let program = find_program(&RootFs::new(root)?, &config.workdir, &environment, name)?;
            let elf = std::fs::read(program)?;
            let mut container = Container::from_elf(vcpu(mem)?, &elf, &argv, &environment, stdio)?;
            container.set_root(root)?;
            if !config.workdir.is_empty() {
                container
                    .set_cwd(&config.workdir)
                    .map_err(|e| anyhow::anyhow!("WORKDIR {}: errno {}", config.workdir, e.0))?;
            }
            return Ok(container);
        }
    };
    container.set_root(root)?;
    Ok(container)
}

/// Start a VM image's machine. A warm image resumes where `WARM` froze it;
/// otherwise the kernel boots, its filesystem unpacked into `root` to read it.
/// `vcpu` makes a vCPU with RAM at `vm::RAM_BASE` and the given memory: a warm
/// image's own size, or `mem` for a cold boot.
pub fn start_vm<V: Vcpu>(
    image: &Image,
    vcpu: impl FnOnce(u32) -> anyhow::Result<V>,
    console: Console,
    root: &Path,
    mem: u32,
) -> anyhow::Result<Machine<V>> {
    if let Some(frozen) = image.warm_machine()? {
        return Ok(Machine::thaw(
            vcpu(frozen.machine.mem.len() as u32)?,
            &frozen,
            console,
        )?);
    }
    image.unpack_rootfs(root)?;
    let kernel = read_kernel(&image.config, root)?;
    let bootargs = (image.config.bootargs.as_deref()).unwrap_or(vm::DEFAULT_BOOTARGS);
    Ok(Machine::boot(vcpu(mem)?, &kernel, bootargs, console)?)
}

/// The kernel a VM image's config names, from its unpacked filesystem.
pub fn read_kernel(config: &Config, root: &Path) -> anyhow::Result<Vec<u8>> {
    let path = config
        .kernel
        .as_deref()
        .context("not a VM image: no KERNEL")?;
    let resolved = RootFs::new(root)?
        .resolve(b"/", path.as_bytes(), true)
        .map_err(|e| anyhow::anyhow!("KERNEL {path}: errno {}", e.0))?;
    std::fs::read(&resolved.host).with_context(|| format!("KERNEL {path}"))
}

/// Find the program `name` in `root` the way a shell would: as a path if it has
/// a slash, otherwise on `PATH` from the image's environment.
pub fn find_program(
    root: &RootFs,
    workdir: &str,
    env: &[String],
    name: &str,
) -> anyhow::Result<PathBuf> {
    let candidates: Vec<String> = if name.contains('/') {
        vec![name.to_string()]
    } else {
        let path = env
            .iter()
            .find_map(|var| var.strip_prefix("PATH="))
            .unwrap_or(DEFAULT_PATH);
        path.split(':').map(|dir| format!("{dir}/{name}")).collect()
    };
    let base = if workdir.is_empty() { "/" } else { workdir };
    for candidate in &candidates {
        if let Ok(resolved) = root.resolve(base.as_bytes(), candidate.as_bytes(), true)
            && resolved.host.is_file()
        {
            return Ok(resolved.host);
        }
    }
    bail!("{name}: not found in the image")
}

/// Images built locally, by name and tag, like Docker's image store. Lives in
/// `$SHEM_HOME/images`, or `~/.local/share/shem/images`.
pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn open() -> anyhow::Result<Self> {
        let home = match std::env::var_os("SHEM_HOME") {
            Some(dir) => PathBuf::from(dir),
            None => match std::env::var_os("XDG_DATA_HOME") {
                Some(dir) => PathBuf::from(dir).join("shem"),
                None => PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?)
                    .join(".local/share/shem"),
            },
        };
        Ok(Self {
            root: home.join("images"),
        })
    }

    /// The layout of the image twin of the ModelPack artifact in the layout
    /// `dir`, made once and kept, by the artifact's digest.
    pub fn twin(&self, dir: &Path) -> anyhow::Result<PathBuf> {
        let artifact = modelpack::Artifact::read(dir)?.context("not a ModelPack artifact")?;
        let hex = artifact
            .descriptor
            .digest
            .trim_start_matches("sha256:")
            .to_string();
        let twins = self.root.with_file_name("twins");
        let path = twins.join(hex);
        if !path.join("oci-layout").is_file() {
            std::fs::create_dir_all(&twins)?;
            let staging = tempfile::tempdir_in(&twins)?;
            let root = tempfile::tempdir()?;
            let twin = artifact.twin(&dir.display().to_string(), root.path())?;
            twin.write_layout(&staging.path().join("layout"), "latest")?;
            std::fs::rename(staging.path().join("layout"), &path)?;
        }
        Ok(path)
    }

    /// A directory holding the filesystem of the image whose manifest digest
    /// is `digest`, unpacked once and kept: images are immutable, and a model
    /// is too big to unpack for every run.
    pub fn rootfs(&self, image: &Image, digest: &str) -> anyhow::Result<PathBuf> {
        let hex = digest
            .strip_prefix("sha256:")
            .context("unsupported digest")?;
        let cache = self.root.with_file_name("rootfs");
        let dir = cache.join(hex);
        if !dir.is_dir() {
            std::fs::create_dir_all(&cache)?;
            let staging = tempfile::tempdir_in(&cache)?;
            image.unpack_rootfs(staging.path())?;
            // A model's primed session, where `shem save` puts it for Docker too
            if let (Some(_), Some(files)) = (&image.config.model, image.state_files()?) {
                let dir = staging.path().join(model::SESSION_DIR);
                std::fs::create_dir_all(&dir)?;
                for (name, data) in files {
                    std::fs::write(dir.join(name), data)?;
                }
            }
            std::fs::rename(staging.keep(), &dir)?;
        }
        Ok(dir)
    }

    /// The layout of the image `reference` in the store, pulled from its
    /// registry the first time, as `docker run` does. A ModelPack artifact
    /// is kept as its image twin.
    pub fn fetch(&self, reference: &str) -> anyhow::Result<PathBuf> {
        let path = self.path(reference)?;
        if !path.join("oci-layout").is_file() {
            let Some(host) = registry_host(reference) else {
                bail!("{reference}: no such executable or image (see `shem images`)");
            };
            let digest = pull(reference, &path, is_local(host))?;
            eprintln!("shem: pulled {reference} {digest}");
        }
        Ok(path)
    }

    /// Where the image `reference` (`name` or `name:tag`) lives.
    pub fn path(&self, reference: &str) -> anyhow::Result<PathBuf> {
        let (name, tag) = parse_reference(reference)?;
        Ok(self.root.join(name).join(tag))
    }

    /// Every image in the store, as (reference, manifest digest).
    pub fn list(&self) -> anyhow::Result<Vec<(String, String)>> {
        let mut images = Vec::new();
        self.collect(&self.root, &mut images)?;
        images.sort();
        Ok(images)
    }

    fn collect(&self, dir: &Path, images: &mut Vec<(String, String)>) -> anyhow::Result<()> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(());
        };
        for entry in entries {
            let path = entry?.path();
            if path.join("oci-layout").is_file() {
                let relative = path.strip_prefix(&self.root)?;
                let tag = relative.file_name().unwrap().to_string_lossy();
                let name = relative.parent().unwrap().to_string_lossy();
                images.push((format!("{name}:{tag}"), Image::layout_digest(&path)?));
            } else if path.is_dir() {
                self.collect(&path, images)?;
            }
        }
        Ok(())
    }
}

/// Split `name[:tag]`, checking the name is safe to use as a path. The name
/// may start with a registry host and port, as in `localhost:5000/team/app`.
pub fn parse_reference(reference: &str) -> anyhow::Result<(&str, &str)> {
    let (name, tag) = match reference.rsplit_once(':') {
        Some((name, tag)) if !tag.contains('/') => (name, tag),
        _ => (reference, "latest"),
    };
    let word = |s: &str| {
        !s.is_empty()
            && s != "."
            && s != ".."
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
    };
    let mut parts = name.split('/');
    let first = parts.next().unwrap_or_default();
    let host_ok = match first.split_once(':') {
        Some((host, port)) => {
            word(host) && !port.is_empty() && port.chars().all(|c| c.is_ascii_digit())
        }
        None => word(first),
    };
    if !host_ok || !parts.all(word) || !word(tag) {
        bail!("invalid image reference {reference:?}");
    }
    Ok((name, tag))
}
