use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use shem::container::Container;
use shem::image::{self, BuildOptions, Image, Store};
use shem::linux::Stdio;
use shem::neural::NeuralVcpu;
use shem::reference::ReferenceVcpu;
use shem::vcpu::Vcpu;
use shem::vm::{self, Console, Halt, Machine};

/// Exit status for errors in shem itself rather than the guest, as with `docker run`.
const HOST_ERROR: u8 = 125;

#[derive(Parser)]
#[command(
    name = "shem",
    version,
    about = "Run containers and VMs inside a model's weights"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a container from an image, or from a static RV32 Linux executable
    Run(RunArgs),
    /// Build an image from a Shemfile
    Build(BuildArgs),
    /// List the images in the local store
    Images,
    /// Push an image to an OCI registry (uses oras), or to a Hugging Face
    /// repo as hf.co/ORG/NAME[:BRANCH]
    Push(PushArgs),
    /// Pull an image from an OCI registry into the store (uses oras), or
    /// from a Hugging Face repo as hf.co/ORG/NAME[:REVISION]
    Pull(PullArgs),
    /// Write an image as an OCI archive for `docker load`, to run with
    /// `docker run --runtime io.containerd.shem.v1`
    Save(SaveArgs),
    /// Load images from an OCI archive, such as `docker save` writes, into
    /// the store: to run images Docker built where the shim is not, as with
    /// Docker Desktop
    Load(LoadArgs),
    /// Serve a model image as OpenAI's chat completions API, at
    /// http://HOST:PORT/v1, until interrupted
    Serve(ServeArgs),
    /// Serve the images in Hugging Face repos as a read-only OCI registry,
    /// at HOST:PORT/ORG/NAME:TAG, until interrupted
    Registry(RegistryArgs),
    /// Boot an operating system kernel in VM mode, with its console on stdio
    Boot(BootArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum VcpuKind {
    /// The reference interpreter
    Reference,
    /// The neural executor, served over --neural-socket
    Neural,
}

#[derive(Args)]
struct VcpuArgs {
    /// Which vCPU runs the guest
    #[arg(long = "vcpu", value_enum, default_value = "reference")]
    kind: VcpuKind,
    /// Unix socket of a running neural vCPU server (python -m shem_neural.server)
    #[arg(long, value_name = "PATH", env = "SHEM_NEURAL_SOCKET")]
    neural_socket: Option<PathBuf>,
    /// A compiled CPU layer for the neural vCPU, such as a model image's
    /// model/compiled/rv32im, instead of the server's own weights
    #[arg(long, value_name = "PATH")]
    neural_weights: Option<PathBuf>,
}

impl VcpuArgs {
    fn socket(&self) -> anyhow::Result<PathBuf> {
        (self.neural_socket.clone())
            .context("--vcpu neural needs --neural-socket or SHEM_NEURAL_SOCKET")
    }
}

/// A neural vCPU with memory at `base`, on the server at `socket`.
fn neural(
    socket: &Path,
    mem: u32,
    base: u32,
    weights: Option<&Path>,
) -> anyhow::Result<NeuralVcpu> {
    NeuralVcpu::connect_with(socket, mem, base, weights)
        .with_context(|| format!("connecting to {}", socket.display()))
}

#[derive(Args)]
struct RunArgs {
    /// Guest memory size, such as 16M; the default is 16M, or 64M for a VM
    /// image, and a warm image brings its own
    #[arg(long, value_parser = parse_size)]
    mem: Option<u32>,
    /// Stop the guest after this many instructions
    #[arg(long)]
    max_steps: Option<u64>,
    /// Set an environment variable in the guest (KEY=VALUE)
    #[arg(short, long = "env", value_name = "KEY=VALUE")]
    env: Vec<String>,
    /// Print each system call the guest makes to stderr
    #[arg(long)]
    trace: bool,
    /// Print how many instructions the guest retired
    #[arg(long)]
    stats: bool,
    #[command(flatten)]
    vcpu: VcpuArgs,
    /// Host directory to use as an executable's root filesystem
    #[arg(long, value_name = "DIR")]
    root: Option<PathBuf>,
    /// For a model image, continue this session: one `shem push --state`
    /// published, in a registry or as an OCI layout directory
    #[arg(long, value_name = "REF")]
    state: Option<String>,
    /// An image (name[:tag] in the store, or an OCI layout directory), or an executable
    target: String,
    /// Arguments: for an image they replace its CMD
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

#[derive(Args)]
struct BuildArgs {
    /// Name and tag for the image, such as hello:latest
    #[arg(short, long)]
    tag: Option<String>,
    /// The Shemfile, if not CONTEXT/Shemfile
    #[arg(short, long)]
    file: Option<PathBuf>,
    /// Set a build argument (KEY=VALUE)
    #[arg(long = "build-arg", value_name = "KEY=VALUE")]
    build_args: Vec<String>,
    /// Guest memory for WARM steps, which a warm image keeps; the default is
    /// 16M, or 64M for a VM image
    #[arg(long, value_parser = parse_size)]
    mem: Option<u32>,
    /// Write the image as an OCI layout to this directory instead of the store
    #[arg(short, long, value_name = "DIR")]
    output: Option<PathBuf>,
    /// The build context: COPY sources are relative to it
    #[arg(default_value = ".")]
    context: PathBuf,
}

#[derive(Args)]
struct PushArgs {
    /// Talk to the registry over plain HTTP, as to a local test registry
    #[arg(long)]
    plain_http: bool,
    /// Also list the image under these platforms, such as linux/amd64, for
    /// runtimes that pick images by the host's platform (see docs/kubernetes.md)
    #[arg(long, value_delimiter = ',', conflicts_with_all = ["modelpack", "state"])]
    platform: Vec<String>,
    /// Push a model image's model as a ModelPack artifact, whose layers are
    /// the image's own and whose subject is the image (see docs/models.md)
    #[arg(long)]
    modelpack: bool,
    /// With --modelpack, write the layers as plain tars, as the spec
    /// recommends and modctl 0.2 needs, instead of sharing the image's blobs
    #[arg(long, requires = "modelpack")]
    uncompressed: bool,
    /// Push a model image's session as an artifact of its own, whose subject
    /// is the image without it, for `shem run --state` (see docs/models.md)
    #[arg(long, conflicts_with = "modelpack")]
    state: bool,
    /// The image: name[:tag] in the store, or an OCI layout directory
    image: String,
    /// Where to push it, such as ghcr.io/me/app:1.0, or hf.co/me/app to push
    /// it to the main branch of a Hugging Face repo
    remote: String,
}

#[derive(Args)]
struct SaveArgs {
    /// Write to this file instead of stdout
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// The name Docker gives the image (default: the image's own)
    #[arg(short, long)]
    tag: Option<String>,
    /// Also list the image under these platforms, such as linux/amd64, for
    /// runtimes that pick images by the host's platform (see docs/kubernetes.md)
    #[arg(long, value_delimiter = ',')]
    platform: Vec<String>,
    /// The image: name[:tag] in the store, or an OCI layout directory
    image: String,
}

#[derive(Args)]
struct LoadArgs {
    /// Read the archive from this file instead of stdin
    #[arg(short, long, value_name = "FILE")]
    input: Option<PathBuf>,
    /// The name to store the image under, if not the one in the archive
    #[arg(short, long)]
    tag: Option<String>,
}

#[derive(Args)]
struct ServeArgs {
    /// The port to serve at
    #[arg(short, long, default_value_t = 8000)]
    port: u16,
    /// The address to listen at: this host's own by default, or 0.0.0.0
    /// for every network it is on
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// A model image: name[:tag] in the store, or an OCI layout directory
    image: String,
}

#[derive(Args)]
struct RegistryArgs {
    /// The port to serve at
    #[arg(short, long, default_value_t = 5000)]
    port: u16,
    /// The address to listen at: this host's own by default. Repos are read
    /// with this host's Hugging Face token, so whoever reaches the registry
    /// can pull the private ones it can read
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

#[derive(Args)]
struct PullArgs {
    /// Talk to the registry over plain HTTP, as to a local test registry
    #[arg(long)]
    plain_http: bool,
    /// The name to store it under, if not the remote reference
    #[arg(short, long)]
    tag: Option<String>,
    /// The image to pull, such as ghcr.io/me/app:1.0, or hf.co/me/app, which
    /// imports the model if the repo holds a checkpoint and no image
    remote: String,
}

#[derive(Args)]
struct BootArgs {
    /// Guest memory size, such as 64M
    #[arg(long, default_value = "64M", value_parser = parse_size)]
    mem: u32,
    /// The kernel command line
    #[arg(long, default_value = vm::DEFAULT_BOOTARGS)]
    append: String,
    /// Stop the machine after this many instructions
    #[arg(long)]
    max_steps: Option<u64>,
    /// Print how many instructions the machine retired
    #[arg(long)]
    stats: bool,
    #[command(flatten)]
    vcpu: VcpuArgs,
    /// A raw RV32 kernel image, such as a no-MMU Linux Image, loaded at 0x80000000
    kernel: PathBuf,
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Run(args) => run(args),
        Command::Build(args) => build(args),
        Command::Images => images(),
        Command::Push(args) => push(args),
        Command::Pull(args) => pull(args),
        Command::Save(args) => save(args),
        Command::Load(args) => load(args),
        Command::Serve(args) => serve(args),
        Command::Registry(args) => image::model::serve_registry(&args.host, args.port),
        Command::Boot(args) => boot(args),
    };
    match result {
        Ok(status) => ExitCode::from(status),
        Err(err) => {
            eprintln!("shem: {err:#}");
            ExitCode::from(HOST_ERROR)
        }
    }
}

fn run(args: RunArgs) -> anyhow::Result<u8> {
    match args.vcpu.kind {
        VcpuKind::Reference => launch(&args, |mem, base| Ok(ReferenceVcpu::with_base(mem, base))),
        VcpuKind::Neural => {
            let socket = args.vcpu.socket()?;
            let weights = args.vcpu.neural_weights.clone();
            launch(&args, |mem, base| {
                neural(&socket, mem, base, weights.as_deref())
            })
        }
    }
}

/// Run an executable or an image. `vcpu` makes a vCPU with the given memory
/// size and base address.
fn launch<V: Vcpu + Clone>(
    args: &RunArgs,
    vcpu: impl FnOnce(u32, u32) -> anyhow::Result<V>,
) -> anyhow::Result<u8> {
    let stdio = Stdio {
        stdin: Box::new(std::io::stdin()),
        stdout: Box::new(std::io::stdout()),
        stderr: Box::new(std::io::stderr()),
    };
    // An image's filesystem is unpacked fresh for each run, and removed after.
    let mut rootfs = None;
    let mut container = if Path::new(&args.target).is_file() {
        if args.state.is_some() {
            bail!("--state is for model images, not executables");
        }
        let elf =
            std::fs::read(&args.target).with_context(|| format!("reading {}", args.target))?;
        let argv: Vec<String> = std::iter::once(args.target.clone())
            .chain(args.args.iter().cloned())
            .collect();
        let mem = args.mem.unwrap_or(image::CONTAINER_MEM);
        let mut container = Container::from_elf(vcpu(mem, 0)?, &elf, &argv, &args.env, stdio)
            .with_context(|| format!("loading {}", args.target))?;
        if let Some(root) = &args.root {
            container
                .set_root(root)
                .with_context(|| format!("using {} as the root filesystem", root.display()))?;
        }
        container
    } else {
        if args.root.is_some() {
            bail!("--root is for executables: an image brings its own filesystem");
        }
        let image = load_image(&args.target)?;
        if image::model::is_driver(&image.config.entrypoint) {
            return run_model(args, &image);
        }
        if args.state.is_some() {
            bail!(
                "--state is for model images: a program or VM image resumes the state it was frozen with"
            );
        }
        let dir = tempfile::tempdir()?;
        if image.config.kernel.is_some() {
            return run_vm(args, &image, dir.path(), |mem| vcpu(mem, vm::RAM_BASE));
        }
        let container = image::start(
            &image,
            |mem| vcpu(mem, 0),
            &args.args,
            &args.env,
            stdio,
            dir.path(),
            args.mem.unwrap_or(image::CONTAINER_MEM),
        )
        .with_context(|| format!("starting {}", args.target))?;
        rootfs = Some(dir);
        container
    };
    container.trace = args.trace;
    let terminal = Terminal::attach(&mut container.linux);
    let status = container.run(args.max_steps);
    drop(terminal);
    if args.stats {
        eprintln!("shem: {} instructions retired", container.steps);
    }
    for (nr, count) in &container.linux.unsupported {
        eprintln!("shem: guest made unsupported system call {nr} ({count}x)");
    }
    for (pid, exit) in &container.faults {
        eprintln!(
            "shem: process {pid} stopped: {:?} at pc {:#010x}",
            exit.reason, exit.pc
        );
    }
    drop(rootfs);
    Ok(status?)
}

/// The host terminal, while a container uses it: its settings, restored when
/// the container is done, since the guest may change them (a shell turns off
/// echo and line buffering to edit lines itself).
struct Terminal {
    saved: libc::termios,
}

impl Terminal {
    /// Give the container the host's terminal if stdin is one, and send the
    /// signals typed at it (Ctrl-C, Ctrl-\, Ctrl-Z) to the guest instead of
    /// to shem.
    fn attach(linux: &mut shem::linux::Linux) -> Option<Self> {
        // SAFETY: isatty, tcgetattr and signal take a descriptor, a buffer
        // the size of termios, and a handler that only stores an integer
        unsafe {
            if libc::isatty(0) != 1 {
                return None;
            }
            let mut saved: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(0, &mut saved) != 0 {
                return None;
            }
            extern "C" fn typed(signal: libc::c_int) {
                shem::container::TERMINAL_SIGNAL
                    .store(signal as u32, std::sync::atomic::Ordering::SeqCst);
            }
            for signal in [libc::SIGINT, libc::SIGQUIT, libc::SIGTSTP] {
                libc::signal(signal, typed as *const () as libc::sighandler_t);
            }
            linux.terminal = Some(0);
            Some(Self { saved })
        }
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        // SAFETY: restoring the settings tcgetattr returned
        unsafe { libc::tcsetattr(0, libc::TCSANOW, &self.saved) };
    }
}

/// An image from an OCI layout directory, or from the store by reference,
/// pulled from its registry the first time. A ModelPack artifact, in a
/// layout or a registry, is its image twin.
fn load_image(target: &str) -> anyhow::Result<Image> {
    let dir = Path::new(target);
    if dir.join("oci-layout").is_file() {
        return Image::read_layout(&image_dir(target)?);
    }
    Image::read_layout(&Store::open()?.fetch(target)?)
}

fn build(args: BuildArgs) -> anyhow::Result<u8> {
    let mut build_args = BTreeMap::new();
    for pair in &args.build_args {
        let (key, value) = pair
            .split_once('=')
            .context("expected --build-arg KEY=VALUE")?;
        build_args.insert(key.to_string(), value.to_string());
    }
    let options = BuildOptions {
        shemfile: args.file.unwrap_or_else(|| args.context.join("Shemfile")),
        context: args.context,
        build_args,
        mem: args.mem,
    };
    let built = image::build(&options)?;
    let (tag, dir) = match (args.tag, args.output) {
        (tag, Some(dir)) => (tag.unwrap_or_else(|| "latest".into()), dir),
        (Some(tag), None) => {
            let dir = Store::open()?.path(&tag)?;
            (tag, dir)
        }
        (None, None) => bail!("give the image a --tag, or an --output directory"),
    };
    if dir.exists() {
        let is_image = dir.join("oci-layout").is_file();
        let is_empty = std::fs::read_dir(&dir)?.next().is_none();
        if !is_image && !is_empty {
            bail!(
                "{} exists and is not an image; not replacing it",
                dir.display()
            );
        }
        std::fs::remove_dir_all(&dir)?;
    }
    let (_, tag_name) = image::parse_reference(&tag)?;
    let digest = built.write_layout(&dir, tag_name)?;
    eprintln!("shem: built {tag} {digest}");
    Ok(0)
}

/// Run a VM image: boot its kernel, or resume its booted machine, with the
/// console on stdio.
fn run_vm<V: Vcpu>(
    args: &RunArgs,
    image: &Image,
    root: &Path,
    vcpu: impl FnOnce(u32) -> anyhow::Result<V>,
) -> anyhow::Result<u8> {
    if !args.args.is_empty() || !args.env.is_empty() {
        bail!("a VM image takes no arguments or environment: KERNEL sets its command line");
    }
    let mem = args.mem.unwrap_or(image::VM_MEM);
    let machine = image::start_vm(image, vcpu, stdio_console(), root, mem)
        .with_context(|| format!("starting {}", args.target))?;
    finish_vm(machine, args.max_steps, args.stats)
}

/// Run an image whose entrypoint is a driver, such as a model's
/// `shem:generate`, on its filesystem, unpacked once and cached.
fn run_model(args: &RunArgs, image: &Image) -> anyhow::Result<u8> {
    let digest = Image::layout_digest(&image_dir(&args.target)?)?;
    let root = Store::open()?.rootfs(image, &digest)?;
    let input = if args.args.is_empty() {
        &image.config.cmd
    } else {
        &args.args
    };
    // A published session: fetched, checked against this model, and unpacked
    let session = tempfile::tempdir()?;
    if let Some(state) = &args.state {
        let fetched = tempfile::tempdir()?;
        let layout = if Path::new(state).join("oci-layout").is_file() {
            PathBuf::from(state)
        } else {
            let host = image::registry_host(state).with_context(|| {
                format!("{state}: no such session layout or registry reference")
            })?;
            image::copy_to_layout(state, fetched.path(), image::is_local(host))?;
            fetched.path().to_path_buf()
        };
        let model = image::session::model_digest(&root, image)?;
        image::session::load(&layout, &model, session.path())?;
    }
    let session = args.state.as_ref().map(|_| session.path());
    image::model::run_driver(&image.config.entrypoint, input, &root, session)
}

/// A console on this process's stdin and stdout.
fn stdio_console() -> Console {
    let (keys, input) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::Read;
        for byte in std::io::stdin().lock().bytes().map_while(Result::ok) {
            if keys.send(byte).is_err() {
                break;
            }
        }
    });
    Console::new(input, Box::new(std::io::stdout()))
}

fn boot(args: BootArgs) -> anyhow::Result<u8> {
    let kernel = std::fs::read(&args.kernel)
        .with_context(|| format!("reading {}", args.kernel.display()))?;
    match args.vcpu.kind {
        VcpuKind::Reference => {
            let vcpu = ReferenceVcpu::with_base(args.mem, vm::RAM_BASE);
            boot_on(vcpu, &kernel, &args)
        }
        VcpuKind::Neural => {
            let weights = args.vcpu.neural_weights.as_deref();
            let vcpu = neural(&args.vcpu.socket()?, args.mem, vm::RAM_BASE, weights)?;
            boot_on(vcpu, &kernel, &args)
        }
    }
}

fn boot_on<V: Vcpu>(vcpu: V, kernel: &[u8], args: &BootArgs) -> anyhow::Result<u8> {
    let machine = Machine::boot(vcpu, kernel, &args.append, stdio_console())?;
    finish_vm(machine, args.max_steps, args.stats)
}

fn finish_vm<V: Vcpu>(
    mut machine: Machine<V>,
    max_steps: Option<u64>,
    stats: bool,
) -> anyhow::Result<u8> {
    let halt = machine.run(max_steps);
    if stats {
        eprintln!("shem: {} instructions retired", machine.steps);
    }
    match halt? {
        Halt::Poweroff => Ok(0),
        Halt::Reboot => {
            eprintln!("shem: the guest asked to reboot; stopping instead");
            Ok(0)
        }
    }
}

fn image_dir(target: &str) -> anyhow::Result<PathBuf> {
    let dir = Path::new(target);
    if dir.join("oci-layout").is_file() {
        if image::modelpack::is_artifact(dir) {
            return Store::open()?.twin(dir);
        }
        return Ok(dir.to_path_buf());
    }
    let path = Store::open()?.path(target)?;
    if !path.join("oci-layout").is_file() {
        bail!("{target}: no such image (see `shem images`)");
    }
    Ok(path)
}

fn push(args: PushArgs) -> anyhow::Result<u8> {
    let staged = tempfile::tempdir()?;
    let dir = if args.state {
        let dir = image_dir(&args.image)?;
        let image = Image::read_layout(&dir)?;
        let root = Store::open()?.rootfs(&image, &Image::layout_digest(&dir)?)?;
        let (_, tag) = image::parse_reference(&args.remote)?;
        let out = staged.path().join("session");
        image::session::publish(&image, &root, &out, tag)?;
        out
    } else if args.modelpack {
        let dir = image_dir(&args.image)?;
        let image = Image::read_layout(&dir)?;
        let root = Store::open()?.rootfs(&image, &Image::layout_digest(&dir)?)?;
        let (_, tag) = image::parse_reference(&args.remote)?;
        let out = staged.path().join("artifact");
        image::modelpack::export(&image, &dir, &root, &out, tag, args.uncompressed)?;
        out
    } else if args.platform.is_empty() {
        image_dir(&args.image)?
    } else {
        let (_, tag) = image::parse_reference(&args.remote)?;
        load_image(&args.image)?.write_layout_for(staged.path(), tag, &args.platform)?;
        staged.path().to_path_buf()
    };
    let digest = image::push(&dir, &args.remote, args.plain_http)?;
    eprintln!("shem: pushed {} {digest}", args.remote);
    Ok(0)
}

fn save(args: SaveArgs) -> anyhow::Result<u8> {
    use std::io::{IsTerminal, Write};
    let image = load_image(&args.image)?;
    let name = args.tag.as_deref().unwrap_or(&args.image);
    let archive = image.docker_archive(name, &args.platform)?;
    match &args.output {
        Some(path) => std::fs::write(path, archive)?,
        None if std::io::stdout().is_terminal() => {
            bail!("not writing an archive to a terminal: pipe it to `docker load` or give --output")
        }
        None => std::io::stdout().lock().write_all(&archive)?,
    }
    Ok(0)
}

fn load(args: LoadArgs) -> anyhow::Result<u8> {
    use std::io::IsTerminal;
    let store = Store::open()?;
    let tag = args.tag.as_deref();
    let names = match &args.input {
        Some(path) => {
            let file =
                std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            store.load(std::io::BufReader::new(file), tag)?
        }
        None if std::io::stdin().is_terminal() => {
            bail!("no archive on stdin: pipe `docker save IMAGE` to it, or give --input")
        }
        None => store.load(std::io::stdin().lock(), tag)?,
    };
    for name in names {
        println!("Loaded image: {name}");
    }
    Ok(0)
}

fn serve(args: ServeArgs) -> anyhow::Result<u8> {
    let image = load_image(&args.image)?;
    if !image::model::is_driver(&image.config.entrypoint) {
        bail!(
            "{} is no model image: shem serve serves a model",
            args.image
        );
    }
    let digest = Image::layout_digest(&image_dir(&args.image)?)?;
    let root = Store::open()?.rootfs(&image, &digest)?;
    let address = match args.host.contains(':') {
        true => format!("[{}]:{}", args.host, args.port),
        false => format!("{}:{}", args.host, args.port),
    };
    image::model::serve(&image.config.entrypoint, &root, &address)
}

fn pull(args: PullArgs) -> anyhow::Result<u8> {
    let local = args.tag.as_deref().unwrap_or(&args.remote);
    let dir = Store::open()?.path(local)?;
    let digest = image::pull(&args.remote, &dir, args.plain_http)?;
    eprintln!("shem: pulled {} as {local} {digest}", args.remote);
    Ok(0)
}

/// List the store as `docker images` does: a column each for the reference,
/// the short manifest digest, what the image runs and its stored size. On a
/// terminal, and unless NO_COLOR is set, the header is bold and the tag and
/// ID dim.
fn images() -> anyhow::Result<u8> {
    use std::io::IsTerminal;
    let store = Store::open()?;
    let mut rows = Vec::new();
    for (reference, digest) in store.list()? {
        let summary = Image::layout_summary(&store.path(&reference)?)?;
        let id = digest.trim_start_matches("sha256:");
        rows.push((reference, id[..12.min(id.len())].to_string(), summary));
    }
    let style = std::io::stdout().is_terminal()
        && std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty());
    let paint = |code: &str, text: &str| match style {
        true => format!("\x1b[{code}m{text}\x1b[0m"),
        false => text.to_string(),
    };
    let width = rows
        .iter()
        .map(|(reference, ..)| reference.chars().count())
        .chain(["IMAGE".len()])
        .max()
        .unwrap_or_default();
    let header = format!(
        "{:<width$}   {:<12}   {:<7}   {:>8}",
        "IMAGE", "ID", "KIND", "SIZE"
    );
    println!("{}", paint("1", &header));
    for (reference, id, summary) in rows {
        let pad = " ".repeat(width - reference.chars().count());
        let (name, tag) = image::parse_reference(&reference)?;
        let reference = format!("{name}{}", paint("2", &format!(":{tag}")));
        println!(
            "{reference}{pad}   {}   {:<7}   {:>8}",
            paint("2", &id),
            summary.kind,
            human_size(summary.size),
        );
    }
    Ok(0)
}

/// `bytes` in decimal units, as `docker images` shows sizes: 435kB, 1.81MB.
fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1000.0 && unit < UNITS.len() - 1 {
        size /= 1000.0;
        unit += 1;
    }
    match unit {
        0 => format!("{bytes}B"),
        _ if size < 10.0 => format!("{size:.2}{}", UNITS[unit]),
        _ if size < 100.0 => format!("{size:.1}{}", UNITS[unit]),
        _ => format!("{size:.0}{}", UNITS[unit]),
    }
}

fn parse_size(text: &str) -> anyhow::Result<u32> {
    let (digits, unit) = match text.char_indices().last() {
        Some((i, 'K' | 'k')) => (&text[..i], 1 << 10),
        Some((i, 'M' | 'm')) => (&text[..i], 1 << 20),
        Some((i, 'G' | 'g')) => (&text[..i], 1 << 30),
        _ => (text, 1),
    };
    let count = digits
        .parse::<u64>()
        .context("expected a size such as 16M")?;
    match count.checked_mul(unit).map(u32::try_from) {
        Some(Ok(bytes)) if bytes > 0 => Ok(bytes),
        _ => bail!("guest memory must be more than 0 and less than 4G (the RV32 address space)"),
    }
}
