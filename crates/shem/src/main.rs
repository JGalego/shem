use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, bail};
use clap::{Parser, Subcommand, ValueEnum};
use shem::container::Container;
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

#[derive(Clone, Copy, ValueEnum)]
enum VcpuKind {
    /// The reference interpreter
    Reference,
    /// The neural executor, served over --neural-socket
    Neural,
}

#[derive(Subcommand)]
enum Command {
    /// Run a static RV32IM Linux executable as a container
    Run {
        /// Guest memory size, such as 16M
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
        /// Host directory to use as the guest's root filesystem
        #[arg(long, value_name = "DIR")]
        root: Option<PathBuf>,
        /// Executable to run, as a host path
        program: PathBuf,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(status) => ExitCode::from(status),
        Err(err) => {
            eprintln!("shem: {err:#}");
            ExitCode::from(HOST_ERROR)
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<u8> {
    let Command::Run {
        mem,
        vcpu,
        neural_socket,
        ..
    } = &cli.command;
    match vcpu {
        VcpuKind::Reference => run_on(ReferenceVcpu::new(*mem), cli.command),
        VcpuKind::Neural => {
            let socket = neural_socket
                .as_ref()
                .context("--vcpu neural needs --neural-socket or SHEM_NEURAL_SOCKET")?;
            let vcpu = NeuralVcpu::connect(socket, *mem)
                .with_context(|| format!("connecting to {}", socket.display()))?;
            run_on(vcpu, cli.command)
        }
    }
}

fn run_on<V: Vcpu + Clone>(vcpu: V, command: Command) -> anyhow::Result<u8> {
    let Command::Run {
        max_steps,
        env,
        trace,
        stats,
        root,
        program,
        args,
        ..
    } = command;
    let elf = std::fs::read(&program).with_context(|| format!("reading {}", program.display()))?;
    let argv: Vec<String> = std::iter::once(program.display().to_string())
        .chain(args)
        .collect();
    let stdio = Stdio {
        stdin: Box::new(std::io::stdin()),
        stdout: Box::new(std::io::stdout()),
        stderr: Box::new(std::io::stderr()),
    };
    let mut container = Container::from_elf(vcpu, &elf, &argv, &env, stdio)
        .with_context(|| format!("loading {}", program.display()))?;
    container.trace = trace;
    if let Some(root) = root {
        container
            .set_root(&root)
            .with_context(|| format!("using {} as the root filesystem", root.display()))?;
    }
    let status = container.run(max_steps);
    if stats {
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
    Ok(status?)
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
