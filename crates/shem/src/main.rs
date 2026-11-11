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
}

#[derive(Clone, Copy, ValueEnum)]
enum VcpuKind {
    /// The reference interpreter
    Reference,
    /// The neural executor, served over --neural-socket
    Neural,
}

#[derive(Args)]
struct RunArgs {
    /// Guest memory size, such as 16M (a warm image brings its own)
    #[arg(long, default_value = "16M", value_parser = parse_size)]
    mem: u32,
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
    /// Which vCPU runs the guest
    #[arg(long, value_enum, default_value = "reference")]
    vcpu: VcpuKind,
    /// Unix socket of a running neural vCPU server (python -m shem_neural.server)
    #[arg(long, value_name = "PATH", env = "SHEM_NEURAL_SOCKET")]
    neural_socket: Option<PathBuf>,
    /// Host directory to use as an executable's root filesystem
    #[arg(long, value_name = "DIR")]
    root: Option<PathBuf>,
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
    /// Guest memory for WARM steps, which a warm image keeps
    #[arg(long, default_value = "16M", value_parser = parse_size)]
    mem: u32,
    /// Write the image as an OCI layout to this directory instead of the store
    #[arg(short, long, value_name = "DIR")]
    output: Option<PathBuf>,
    /// The build context: COPY sources are relative to it
    #[arg(default_value = ".")]
    context: PathBuf,
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Run(args) => run(args),
        Command::Build(args) => build(args),
        Command::Images => images(),
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
    match args.vcpu {
        VcpuKind::Reference => launch(&args, |mem| Ok(ReferenceVcpu::new(mem))),
        VcpuKind::Neural => {
            let socket = (args.neural_socket.clone())
                .context("--vcpu neural needs --neural-socket or SHEM_NEURAL_SOCKET")?;
            launch(&args, |mem| {
                NeuralVcpu::connect(&socket, mem)
                    .with_context(|| format!("connecting to {}", socket.display()))
            })
        }
    }
}

fn launch<V: Vcpu + Clone>(
    args: &RunArgs,
    vcpu: impl FnOnce(u32) -> anyhow::Result<V>,
) -> anyhow::Result<u8> {
    let stdio = Stdio {
        stdin: Box::new(std::io::stdin()),
        stdout: Box::new(std::io::stdout()),
        stderr: Box::new(std::io::stderr()),
    };
    // An image's filesystem is unpacked fresh for each run, and removed after.
    let mut rootfs = None;
    let mut container = if Path::new(&args.target).is_file() {
        let elf =
            std::fs::read(&args.target).with_context(|| format!("reading {}", args.target))?;
        let argv: Vec<String> = std::iter::once(args.target.clone())
            .chain(args.args.iter().cloned())
            .collect();
        let mut container = Container::from_elf(vcpu(args.mem)?, &elf, &argv, &args.env, stdio)
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
        let dir = tempfile::tempdir()?;
        let container = image::start(
            &image,
            vcpu,
            &args.args,
            &args.env,
            stdio,
            dir.path(),
            args.mem,
        )
        .with_context(|| format!("starting {}", args.target))?;
        rootfs = Some(dir);
        container
    };
    container.trace = args.trace;
    let status = container.run(args.max_steps);
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

/// An image from an OCI layout directory, or from the store by reference.
fn load_image(target: &str) -> anyhow::Result<Image> {
    let dir = Path::new(target);
    if dir.join("oci-layout").is_file() {
        return Image::read_layout(dir);
    }
    let path = Store::open()?.path(target)?;
    if !path.join("oci-layout").is_file() {
        bail!("{target}: no such executable or image (see `shem images`)");
    }
    Image::read_layout(&path)
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
    let digest = built.write_layout(&dir, &tag)?;
    eprintln!("shem: built {tag} {digest}");
    Ok(0)
}

fn images() -> anyhow::Result<u8> {
    for (reference, digest) in Store::open()?.list()? {
        println!("{reference:<40} {}", &digest[..19]);
    }
    Ok(0)
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
