//! `shem build`: turn a Shemfile into an image.
//!
//! The builder keeps a working copy of the final stage's root filesystem.
//! `COPY` writes into it and records a layer of what it wrote. `WARM` runs the
//! entrypoint on the reference vCPU until its condition holds, freezes the
//! container into a state layer, and records the files the guest changed as
//! another layer; in a VM image it boots the kernel instead, and freezes the
//! machine. Ordinary Docker stages are built by Docker, only when a
//! `COPY --from` needs their files.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};

use super::shemfile::{self, Instruction, Stage, Until};
use super::{Config, Image, Layer, ROOTFS_TYPE, STATE_TYPE, find_program, layer, model};
use crate::container::{Container, Frozen, Pause, Stop};
use crate::linux::{RootFs, Stdio};
use crate::reference::ReferenceVcpu;
use crate::vm::{self, Console, Machine};

/// The stage name the builder adds to export files from a Docker stage.
const EXPORT_STAGE: &str = "shem-export";
/// Instructions a `WARM` may run before giving up, unless it sets `--max-steps`.
const WARM_MAX_STEPS: u64 = 4_000_000_000;
/// Guest memory for `WARM`, unless the build sets it.
pub const CONTAINER_MEM: u32 = 16 << 20;
pub const VM_MEM: u32 = 64 << 20;

pub struct BuildOptions {
    pub shemfile: PathBuf,
    pub context: PathBuf,
    pub build_args: BTreeMap<String, String>,
    /// Guest memory for `WARM` steps, which becomes the warm image's memory:
    /// by default `CONTAINER_MEM`, or `VM_MEM` for a VM image.
    pub mem: Option<u32>,
}

pub fn build(options: &BuildOptions) -> anyhow::Result<Image> {
    let text = std::fs::read_to_string(&options.shemfile)
        .with_context(|| format!("reading {}", options.shemfile.display()))?;
    let stages = shemfile::parse(&text)?;
    let last = stages.last().context("the Shemfile has no FROM")?;
    let base = shem_base(last, &options.context).context(
        "the last stage must be a shem stage: FROM scratch, or FROM a shem image's directory",
    )?;

    let root = tempfile::tempdir()?;
    let mut builder = Builder {
        options,
        stages: &stages,
        image: Image::default(),
        root: root.path().to_path_buf(),
        vars: options.build_args.clone(),
        frozen: None,
        frozen_vm: None,
        docker: BTreeMap::new(),
    };
    builder.from(base)?;
    for instruction in &last.instructions {
        builder.apply(instruction)?;
    }
    Ok(builder.image)
}

/// The base a shem stage starts from: nothing, an image's layout directory,
/// or a Hugging Face model.
enum Base {
    Scratch,
    Image(PathBuf),
    Model {
        repo: String,
        revision: Option<String>,
    },
}

fn shem_base(stage: &Stage, context: &Path) -> Option<Base> {
    if stage.from == "scratch" {
        return Some(Base::Scratch);
    }
    if let Some((repo, revision)) = model::hf_reference(&stage.from) {
        return Some(Base::Model {
            repo: repo.into(),
            revision: revision.map(String::from),
        });
    }
    let dir = context.join(&stage.from);
    dir.join("oci-layout").is_file().then_some(Base::Image(dir))
}

struct Builder<'a> {
    options: &'a BuildOptions,
    stages: &'a [Stage],
    image: Image,
    root: PathBuf,
    vars: BTreeMap<String, String>,
    frozen: Option<Frozen>,
    frozen_vm: Option<vm::Frozen>,
    /// Docker stages built so far, by name, with their exported files.
    docker: BTreeMap<String, tempfile::TempDir>,
}

impl Builder<'_> {
    fn from(&mut self, base: Base) -> anyhow::Result<()> {
        match base {
            Base::Scratch => {
                self.image.config = Config {
                    machine: "rv32im".into(),
                    ..Config::default()
                };
            }
            Base::Model { repo, revision } => {
                self.image.config = Config {
                    machine: "rv32im".into(),
                    entrypoint: vec!["shem:generate".into()],
                    model: Some(format!("/{}", model::MODEL_DIR)),
                    ..Config::default()
                };
                model::import(
                    &repo,
                    revision.as_deref(),
                    &self.root.join(model::MODEL_DIR),
                )?;
                self.image.config.labels.extend(model::labels(&self.root)?);
                for group in model::layer_groups(&self.root)? {
                    self.add_layer(&group, &[])?;
                }
            }
            Base::Image(dir) => {
                self.image = Image::read_layout(&dir)?;
                self.image.unpack_rootfs(&self.root)?;
                if self.image.config.model.is_some() {
                    // A model's session is rebuilt by its own WARM, if any
                } else if self.image.config.kernel.is_some() {
                    self.frozen_vm = self.image.warm_machine()?;
                } else {
                    self.frozen = self.image.warm_state()?;
                }
            }
        }
        for (key, value) in self
            .image
            .config
            .env
            .iter()
            .filter_map(|var| var.split_once('='))
        {
            self.vars.insert(key.into(), value.into());
        }
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> anyhow::Result<()> {
        match instruction {
            Instruction::Copy {
                from,
                sources,
                dest,
            } => self.copy(from.as_deref(), sources, dest),
            Instruction::Env(pairs) => {
                for (key, value) in pairs {
                    let value = self.expand(value);
                    let config = &mut self.image.config;
                    config
                        .env
                        .retain(|var| var.split_once('=').map(|(k, _)| k) != Some(key));
                    config.env.push(format!("{key}={value}"));
                    self.vars.insert(key.clone(), value);
                }
                Ok(())
            }
            Instruction::Workdir(dir) => {
                let dir = self.guest_path(&self.expand(dir));
                let (resolved, created) = self.ensure_dir(&dir)?;
                self.add_layer(&created, &[])?;
                self.image.config.workdir = resolved;
                Ok(())
            }
            Instruction::Entrypoint(argv) => {
                self.image.config.entrypoint = argv.iter().map(|a| self.expand(a)).collect();
                Ok(())
            }
            Instruction::Cmd(argv) => {
                self.image.config.cmd = argv.iter().map(|a| self.expand(a)).collect();
                Ok(())
            }
            Instruction::Label(pairs) => {
                for (key, value) in pairs {
                    let value = self.expand(value);
                    self.image.config.labels.insert(key.clone(), value);
                }
                Ok(())
            }
            Instruction::Arg { name, default } => {
                if let Some(value) = self.options.build_args.get(name).or(default.as_ref()) {
                    self.vars.insert(name.clone(), value.clone());
                }
                Ok(())
            }
            Instruction::Warm { until, .. } if self.image.config.model.is_some() => {
                if *until != Until::Prompt {
                    bail!("a model image can WARM --until=prompt");
                }
                self.warm_model()
            }
            Instruction::Warm { until, max_steps } => {
                let max_steps = max_steps.unwrap_or(WARM_MAX_STEPS);
                if self.image.config.kernel.is_some() {
                    self.warm_vm(until, max_steps)
                } else {
                    self.warm(until, max_steps)
                }
            }
            Instruction::Kernel { path, cmdline } => {
                if self.frozen.is_some() {
                    bail!("KERNEL after WARM: a warm container cannot become a VM");
                }
                let kernel = self.guest_path(&self.expand(path));
                let cmdline = self.expand(cmdline);
                self.image.config.kernel = Some(kernel);
                self.image.config.bootargs = (!cmdline.is_empty()).then_some(cmdline);
                Ok(())
            }
            Instruction::Other { keyword, .. } if keyword == "RUN" => {
                bail!(
                    "RUN is not allowed in a shem stage: build in a Docker stage and COPY --from it"
                )
            }
            Instruction::Other { keyword, .. } => {
                bail!("{keyword} is not supported in a shem stage")
            }
        }
    }

    /// Substitute `$NAME` and `${NAME}` from build arguments and the environment.
    fn expand(&self, text: &str) -> String {
        let mut out = String::new();
        let mut rest = text;
        while let Some(at) = rest.find('$') {
            out.push_str(&rest[..at]);
            rest = &rest[at + 1..];
            let (name, after) = if let Some(braced) = rest.strip_prefix('{') {
                match braced.find('}') {
                    Some(end) => (&braced[..end], &braced[end + 1..]),
                    None => ("", rest),
                }
            } else {
                let end = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(rest.len());
                (&rest[..end], &rest[end..])
            };
            if name.is_empty() {
                out.push('$');
            } else {
                out.push_str(self.vars.get(name).map_or("", String::as_str));
                rest = after;
            }
        }
        out.push_str(rest);
        out
    }

    fn guest_path(&self, path: &str) -> String {
        if path.starts_with('/') {
            path.to_string()
        } else {
            let workdir = self.image.config.workdir.trim_end_matches('/');
            format!("{workdir}/{path}")
        }
    }

    fn rootfs(&self) -> anyhow::Result<RootFs> {
        Ok(RootFs::new(&self.root)?)
    }

    /// Make sure the guest directory `dir` exists, creating what is missing.
    /// Returns its resolved guest path and the guest paths it created, all
    /// resolved inside the root, so a symlink in the base cannot lead outside.
    fn ensure_dir(&self, dir: &str) -> anyhow::Result<(String, Vec<PathBuf>)> {
        let rootfs = self.rootfs()?;
        let mut current = String::from("/");
        let mut created = Vec::new();
        for part in dir.split('/').filter(|p| !p.is_empty()) {
            current = format!("{}/{part}", current.trim_end_matches('/'));
            let resolved = rootfs
                .resolve(b"/", current.as_bytes(), true)
                .map_err(|e| anyhow::anyhow!("{current}: errno {}", e.0))?;
            match std::fs::metadata(&resolved.host) {
                Ok(meta) if meta.is_dir() => {}
                Ok(_) => bail!("{current} exists and is not a directory"),
                Err(_) => {
                    std::fs::create_dir(&resolved.host)?;
                    created.push(relative(&resolved.guest));
                }
            }
            current = String::from_utf8(resolved.guest)?;
        }
        Ok((current, created))
    }

    fn copy(&mut self, from: Option<&str>, sources: &[String], dest: &str) -> anyhow::Result<()> {
        let sources: Vec<String> = sources.iter().map(|s| self.expand(s)).collect();
        let source_root = match from {
            Some(name) if name.starts_with(model::DRIVER_PREFIX) => self.compiled(name)?,
            Some(stage) => self.docker_export(stage, &sources)?,
            None => self.options.context.clone(),
        };
        let dest = self.guest_path(&self.expand(dest));
        let into_dir = dest.ends_with('/') || sources.len() > 1;
        let mut written = Vec::new();
        for source in &sources {
            let path = inside(&source_root, source)?;
            let meta =
                std::fs::symlink_metadata(&path).with_context(|| format!("COPY {source}"))?;
            if meta.is_dir() {
                let (target, created) = self.ensure_dir(&dest)?;
                written.extend(created);
                self.copy_tree(&path, &target, &mut written)?;
            } else {
                let target = if into_dir {
                    let name = path
                        .file_name()
                        .context("source has no name")?
                        .to_string_lossy();
                    format!("{}/{name}", dest.trim_end_matches('/'))
                } else {
                    dest.clone()
                };
                self.copy_entry(&path, &target, &mut written)?;
            }
        }
        self.add_layer(&written, &[])
    }

    fn copy_tree(
        &self,
        source: &Path,
        target_dir: &str,
        written: &mut Vec<PathBuf>,
    ) -> anyhow::Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(source)?.collect::<Result<_, _>>()?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let name = entry.file_name().to_string_lossy().into_owned();
            let target = format!("{}/{name}", target_dir.trim_end_matches('/'));
            if entry.file_type()?.is_dir() {
                let (resolved, created) = self.ensure_dir(&target)?;
                written.extend(created);
                written.push(relative(resolved.as_bytes()));
                self.copy_tree(&entry.path(), &resolved, written)?;
            } else {
                self.copy_entry(&entry.path(), &target, written)?;
            }
        }
        Ok(())
    }

    /// Copy one file or symlink to the guest path `target`.
    fn copy_entry(
        &self,
        source: &Path,
        target: &str,
        written: &mut Vec<PathBuf>,
    ) -> anyhow::Result<()> {
        let (parent, name) = target.rsplit_once('/').context("bad destination")?;
        let (parent, created) = self.ensure_dir(parent)?;
        written.extend(created);
        let guest = format!("{}/{name}", parent.trim_end_matches('/'));
        let host = self.root.join(relative(guest.as_bytes()));
        if std::fs::symlink_metadata(&host).is_ok_and(|m| !m.is_dir()) {
            std::fs::remove_file(&host)?; // never write through an existing symlink
        }
        let meta = std::fs::symlink_metadata(source)?;
        if meta.is_symlink() {
            std::os::unix::fs::symlink(std::fs::read_link(source)?, &host)?;
        } else {
            std::fs::copy(source, &host)?;
        }
        written.push(relative(guest.as_bytes()));
        Ok(())
    }

    fn add_layer(&mut self, paths: &[PathBuf], deleted: &[PathBuf]) -> anyhow::Result<()> {
        if paths.is_empty() && deleted.is_empty() {
            return Ok(());
        }
        let data = layer::create(&self.root, paths, deleted)?;
        // Filesystem layers go before any state layer, which always comes last.
        let at = self
            .image
            .layers
            .iter()
            .position(|l| l.media_type == STATE_TYPE)
            .unwrap_or(self.image.layers.len());
        self.image.layers.insert(
            at,
            Layer {
                media_type: ROOTFS_TYPE.into(),
                data,
            },
        );
        Ok(())
    }

    fn warm(&mut self, until: &Until, max_steps: u64) -> anyhow::Result<()> {
        let config = self.image.config.clone();
        let before = layer::scan(&self.root)?;
        let stdio = || Stdio {
            stdin: Box::new(std::io::empty()),
            stdout: Box::new(std::io::stderr()),
            stderr: Box::new(std::io::stderr()),
        };
        let mut container = match self.frozen.take() {
            Some(frozen) => {
                let vcpu = ReferenceVcpu::new(frozen.machine.mem.len() as u32);
                Container::thaw(vcpu, &frozen, stdio())?
            }
            None => {
                let argv: Vec<String> = config
                    .entrypoint
                    .iter()
                    .chain(&config.cmd)
                    .cloned()
                    .collect();
                let name = argv
                    .first()
                    .context("WARM needs an ENTRYPOINT or CMD to run")?;
                let program = find_program(&self.rootfs()?, &config.workdir, &config.env, name)?;
                let elf = std::fs::read(program)?;
                let vcpu = ReferenceVcpu::new(self.options.mem.unwrap_or(CONTAINER_MEM));
                Container::from_elf(vcpu, &elf, &argv, &config.env, stdio())?
            }
        };
        container.set_root(&self.root)?;
        if !config.workdir.is_empty() {
            container
                .set_cwd(&config.workdir)
                .map_err(|e| anyhow::anyhow!("WORKDIR {}: errno {}", config.workdir, e.0))?;
        }
        let pause = match *until {
            Until::Syscall(nr) => Pause::Syscall(nr),
            Until::Steps(n) => Pause::Steps(n),
            Until::Ebreak => Pause::Ebreak,
            Until::Output(_) => {
                bail!("WARM --until=output: is for VM images, which have a console")
            }
            Until::Prompt => bail!("WARM --until=prompt is for model images"),
        };
        match container.run_until(Some(max_steps), Some(pause))? {
            Stop::Paused => {}
            Stop::Exited(status) => {
                bail!("WARM: the guest exited with status {status} before its condition held")
            }
        }
        let frozen = container.freeze()?;

        let (changed, deleted) = layer::diff(&before, &layer::scan(&self.root)?);
        self.add_layer(&changed, &deleted)?;
        let host = serde_json::to_vec(&frozen.host)?;
        let machine = frozen.machine.encode();
        let state = layer::from_files(&[("container.json", &host), ("machine.state", &machine)])?;
        self.image.layers.retain(|l| l.media_type != STATE_TYPE);
        self.image.layers.push(Layer {
            media_type: STATE_TYPE.into(),
            data: state,
        });
        self.image.config.memory = Some(frozen.machine.mem.len() as u32);
        self.frozen = Some(frozen);
        Ok(())
    }

    /// `WARM` in a VM image: boot the kernel, or resume the machine an earlier
    /// `WARM` froze, until the condition holds, and freeze the machine. The
    /// console has no input, and its output goes to stderr.
    fn warm_vm(&mut self, until: &Until, max_steps: u64) -> anyhow::Result<()> {
        let pause = match until {
            Until::Steps(n) => vm::Pause::Steps(*n),
            Until::Output(bytes) => vm::Pause::Output(bytes.clone()),
            Until::Syscall(_) | Until::Ebreak | Until::Prompt => {
                bail!("a VM image can WARM --until=steps:N or output:TEXT")
            }
        };
        let (_, input) = std::sync::mpsc::channel();
        let console = Console::new(input, Box::new(std::io::stderr()));
        let mut machine = match self.frozen_vm.take() {
            Some(frozen) => {
                let mem = frozen.machine.mem.len() as u32;
                Machine::thaw(
                    ReferenceVcpu::with_base(mem, vm::RAM_BASE),
                    &frozen,
                    console,
                )?
            }
            None => {
                let config = &self.image.config;
                let kernel = super::read_kernel(config, &self.root)?;
                let bootargs = config.bootargs.as_deref().unwrap_or(vm::DEFAULT_BOOTARGS);
                let mem = self.options.mem.unwrap_or(VM_MEM);
                let vcpu = ReferenceVcpu::with_base(mem, vm::RAM_BASE);
                Machine::boot(vcpu, &kernel, bootargs, console)?
            }
        };
        match machine.run_until(Some(max_steps), Some(pause))? {
            vm::Stop::Paused => {}
            vm::Stop::Halted(halt) => {
                bail!("WARM: the machine stopped ({halt:?}) before its condition held")
            }
        }
        let frozen = machine.freeze();
        let host = serde_json::to_vec(&frozen.host)?;
        let state = frozen.machine.encode();
        let state = layer::from_files(&[("machine.state", &state), ("vm.json", &host)])?;
        self.image.layers.retain(|l| l.media_type != STATE_TYPE);
        self.image.layers.push(Layer {
            media_type: STATE_TYPE.into(),
            data: state,
        });
        self.image.config.memory = Some(frozen.machine.mem.len() as u32);
        self.frozen_vm = Some(frozen);
        Ok(())
    }

    /// The compiled layer `shem:NAME`, which `COPY --from` copies: built from
    /// its specification by shem, once per build.
    fn compiled(&mut self, from: &str) -> anyhow::Result<PathBuf> {
        if let Some(dir) = self.docker.get(from) {
            return Ok(dir.path().to_path_buf());
        }
        let dir = tempfile::tempdir()?;
        model::compile(&from[model::DRIVER_PREFIX.len()..], dir.path())?;
        let path = dir.path().to_path_buf();
        self.docker.insert(from.to_string(), dir);
        Ok(path)
    }

    /// `WARM --until=prompt` in a model image: prime a session with the
    /// entrypoint's options, such as its system prompt, and freeze every
    /// block's state into the state layer.
    fn warm_model(&mut self) -> anyhow::Result<()> {
        let out = tempfile::tempdir()?;
        model::warm(&self.image.config.entrypoint, &self.root, out.path())?;
        let mut files = Vec::new();
        for name in ["session.json", "session.safetensors"] {
            files.push((name, std::fs::read(out.path().join(name))?));
        }
        let files: Vec<(&str, &[u8])> = files.iter().map(|(n, d)| (*n, d.as_slice())).collect();
        let state = layer::from_files(&files)?;
        self.image.layers.retain(|l| l.media_type != STATE_TYPE);
        self.image.layers.push(Layer {
            media_type: STATE_TYPE.into(),
            data: state,
        });
        Ok(())
    }

    /// The files `sources` of the Docker stage `name`, exported to a directory
    /// where each sits at its own path. Docker builds the stage with an extra
    /// export stage that copies just those paths, so only they leave Docker.
    fn docker_export(&mut self, name: &str, sources: &[String]) -> anyhow::Result<PathBuf> {
        let key = format!("{name}\0{}", sources.join("\0"));
        if let Some(dir) = self.docker.get(&key) {
            return Ok(dir.path().to_path_buf());
        }
        let context = &self.options.context;
        let docker_stages: Vec<&Stage> = self
            .stages
            .iter()
            .filter(|stage| shem_base(stage, context).is_none())
            .collect();
        if !docker_stages
            .iter()
            .any(|s| s.name.as_deref() == Some(name))
        {
            bail!("COPY --from={name}: no Docker stage has that name");
        }
        let mut text: Vec<String> = docker_stages.iter().flat_map(|s| s.lines.clone()).collect();
        text.push(format!("FROM scratch AS {EXPORT_STAGE}"));
        for source in sources {
            let source = format!("/{}", source.trim_start_matches('/'));
            text.push(format!("COPY --from={name} {source} {source}"));
        }
        let dockerfile = tempfile::NamedTempFile::new()?;
        std::fs::write(dockerfile.path(), text.join("\n") + "\n")?;
        let output = tempfile::tempdir()?;
        let mut command = Command::new("docker");
        command
            .args(["build", "--quiet", "--file"])
            .arg(dockerfile.path())
            .args(["--target", EXPORT_STAGE, "--output"])
            .arg(format!("type=local,dest={}", output.path().display()));
        for (key, value) in &self.options.build_args {
            command.arg("--build-arg").arg(format!("{key}={value}"));
        }
        let built = command.arg(context).output().context("running docker")?;
        if !built.status.success() {
            bail!(
                "docker build of stage {name} failed:\n{}",
                String::from_utf8_lossy(&built.stderr)
            );
        }
        let path = output.path().to_path_buf();
        self.docker.insert(key, output);
        Ok(path)
    }
}

/// A guest path as a path relative to the root.
fn relative(guest: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(guest).trim_start_matches('/'))
}

/// `path` under `base`, refusing anything that climbs out of it.
fn inside(base: &Path, path: &str) -> anyhow::Result<PathBuf> {
    let relative = Path::new(path.trim_start_matches('/'));
    if relative
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        bail!("COPY {path}: sources must stay inside the build context");
    }
    Ok(base.join(relative))
}
