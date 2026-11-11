//! Images: a guest filesystem as OCI layers, the configuration to run it, and,
//! for a warm image, the frozen state of a container started from it. A VM
//! image names a kernel in its filesystem instead, and its warm state is a
//! frozen machine that has booted it.
//!
//! An image is stored as an [OCI image layout] whose config and state layer have
//! shem media types, so OCI registries and tools such as ORAS can store it.
//! Builds are reproducible: layers have sorted entries, fixed timestamps and
//! root ownership, and execution is deterministic, so building the same
//! Shemfile twice gives the same digest. See `docs/shemfile.md`.
//!
//! [OCI image layout]: https://github.com/opencontainers/image-spec/blob/main/image-layout.md

mod build;
mod layer;
mod registry;
pub mod shemfile;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use build::{BuildOptions, CONTAINER_MEM, VM_MEM, build};
pub use layer::unpack;
pub use registry::{pull, push};

use crate::container::{Container, Frozen, FrozenHost};
use crate::linux::{RootFs, Stdio};
use crate::vcpu::{MachineState, Vcpu};
use crate::vm::{self, Console, FrozenMachine, Machine};

pub const MANIFEST_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
pub const CONFIG_TYPE: &str = "application/vnd.shem.config.v1+json";
pub const ROOTFS_TYPE: &str = "application/vnd.oci.image.layer.v1.tar";
pub const STATE_TYPE: &str = "application/vnd.shem.state.v1.tar";
const REF_NAME: &str = "org.opencontainers.image.ref.name";
const DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    pub media_type: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Image {
    pub config: Config,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Descriptor {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
    size: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    annotations: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    #[serde(rename = "mediaType")]
    media_type: String,
    config: Descriptor,
    layers: Vec<Descriptor>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Index {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    manifests: Vec<Descriptor>,
}

pub fn digest(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256:{hex}")
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
        let blobs = dir.join("blobs/sha256");
        std::fs::create_dir_all(&blobs)?;
        let put = |media_type: &str, data: &[u8]| -> anyhow::Result<Descriptor> {
            let digest = digest(data);
            std::fs::write(blobs.join(&digest["sha256:".len()..]), data)?;
            Ok(Descriptor {
                media_type: media_type.into(),
                digest,
                size: data.len() as u64,
                annotations: BTreeMap::new(),
            })
        };
        let config = put(CONFIG_TYPE, &serde_json::to_vec(&self.config)?)?;
        let layers = self
            .layers
            .iter()
            .map(|layer| put(&layer.media_type, &layer.data))
            .collect::<anyhow::Result<_>>()?;
        let manifest = Manifest {
            schema_version: 2,
            media_type: MANIFEST_TYPE.into(),
            config,
            layers,
        };
        let mut descriptor = put(MANIFEST_TYPE, &serde_json::to_vec(&manifest)?)?;
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
        let entry = index
            .manifests
            .first()
            .context("the image layout has no manifest")?;
        let manifest: Manifest = serde_json::from_slice(&blob(entry)?)?;
        if manifest.config.media_type != CONFIG_TYPE {
            bail!(
                "not a shem image: its config is {}",
                manifest.config.media_type
            );
        }
        let config = serde_json::from_slice(&blob(&manifest.config)?)?;
        let layers = manifest
            .layers
            .iter()
            .map(|d| {
                Ok(Layer {
                    media_type: d.media_type.clone(),
                    data: blob(d)?,
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self { config, layers })
    }

    /// The manifest digest of the image's layout in `dir`.
    pub fn layout_digest(dir: &Path) -> anyhow::Result<String> {
        let index: Index = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
        Ok(index
            .manifests
            .first()
            .context("no manifest")?
            .digest
            .clone())
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
