//! The neural vCPU against the reference vCPU.
//!
//! These tests start the neural server from `neural/` (Python and PyTorch, run
//! through uv), and the ISA suite needs the riscv-tests binaries, so they are
//! ignored by default; `just neural-tests` sets everything up and runs them.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::machine::*;
use common::*;
use shem::container::{Container, RunError};
use shem::linux;
use shem::neural::NeuralVcpu;
use shem::reference::ReferenceVcpu;
use shem::vcpu::{MachineState, Vcpu};
use shem::vm::{self, Console, DEFAULT_BOOTARGS, Halt, Machine, Pause, RAM_BASE, Stop, VmError};

const MEM: u32 = 1 << 20;
const MAX_STEPS: u64 = 1_000_000;

/// A neural vCPU server for the length of a test.
struct Server {
    child: Child,
    socket: PathBuf,
    _dir: tempfile::TempDir,
}

impl Server {
    fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("neural.sock");
        let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../neural");
        let child = Command::new("uv")
            .args(["run", "--quiet", "--project"])
            .arg(&project)
            .args(["python", "-m", "shem_neural.server", "--socket"])
            .arg(&socket)
            .stderr(Stdio::null())
            .spawn()
            .expect("starting the neural server needs uv on PATH");
        let started = Instant::now();
        while !socket.exists() {
            assert!(
                started.elapsed() < Duration::from_secs(300),
                "the neural server did not start"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
        Server {
            child,
            socket,
            _dir: dir,
        }
    }

    fn vcpu(&self) -> NeuralVcpu {
        NeuralVcpu::connect(&self.socket, MEM).unwrap()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // SIGTERM rather than SIGKILL: uv forwards it to the Python process.
        unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
        let _ = self.child.wait();
    }
}

fn stdio(stdout: &SharedBuf) -> linux::Stdio {
    linux::Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(stdout.clone()),
        stderr: Box::new(std::io::sink()),
    }
}

struct Outcome {
    result: Result<u8, RunError>,
    steps: u64,
    stdout: Vec<u8>,
    state: MachineState,
}

fn run_container<V: Vcpu + Clone>(vcpu: V, elf: &[u8], argv: &[&str]) -> Outcome {
    let stdout = SharedBuf::default();
    let argv: Vec<String> = argv.iter().map(|a| a.to_string()).collect();
    let mut container = Container::from_elf(vcpu, elf, &argv, &[], stdio(&stdout)).unwrap();
    let result = container.run(Some(MAX_STEPS));
    Outcome {
        result,
        steps: container.steps,
        stdout: stdout.contents(),
        state: container.init().vcpu.snapshot(),
    }
}

/// Assert two machine states are identical, naming the first difference.
fn assert_same(what: &str, reference: &MachineState, neural: &MachineState) {
    assert_eq!(reference.pc, neural.pc, "{what}: pc");
    assert_eq!(reference.regs, neural.regs, "{what}: registers");
    assert_eq!(
        reference.instret, neural.instret,
        "{what}: instructions retired"
    );
    if let Some(at) = (0..reference.mem.len()).find(|&i| reference.mem[i] != neural.mem[i]) {
        panic!("{what}: memory differs first at {at:#x}");
    }
}

#[test]
#[ignore = "needs uv, PyTorch and riscv-tests: run `just neural-tests`"]
fn riscv_tests_run_identically_on_both_vcpus() {
    let server = Server::start();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/riscv-tests");
    let mut tests: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("build riscv-tests first: just neural-tests")
        .map(|entry| entry.unwrap().path())
        .collect();
    tests.sort();
    assert!(tests.len() >= 60);
    for path in &tests {
        let name = path.file_name().unwrap().to_string_lossy();
        let elf = std::fs::read(path).unwrap();
        let reference = run_container(ReferenceVcpu::new(MEM), &elf, &[&name]);
        let neural = run_container(server.vcpu(), &elf, &[&name]);
        assert_eq!(reference.result, Ok(0), "{name} on the reference vCPU");
        assert_eq!(neural.result, reference.result, "{name}: exit status");
        assert_eq!(neural.steps, reference.steps, "{name}: instructions");
        assert_same(&name, &reference.state, &neural.state);
    }
}

/// xorshift32, so the programs are the same on every run.
struct Random(u32);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// A random straight-line program over every RV32IM register and immediate
/// ALU instruction, starting from random register values.
fn random_program(random: &mut Random) -> Vec<u32> {
    let mut code = Vec::new();
    for reg in 1..16 {
        code.extend(li(reg, random.next()));
    }
    for _ in 0..60 {
        let (rd, rs1, rs2) = (random.below(16), random.below(16), random.below(16));
        let imm = random.below(4096) as i32 - 2048;
        code.push(match random.below(6) {
            // R-type: base ALU (funct7 0 or 0x20) and the M extension (funct7 1)
            0..=2 => {
                let f3 = random.below(8);
                match random.below(3) {
                    0 => r(0x01, f3, rd, rs1, rs2),
                    1 if f3 == 0 || f3 == 5 => r(0x20, f3, rd, rs1, rs2),
                    _ => r(0x00, f3, rd, rs1, rs2),
                }
            }
            // I-type ALU, with valid shift encodings
            3 | 4 => match random.below(8) {
                1 => i(0x13, 1, rd, rs1, (random.below(32)) as i32),
                5 => i(
                    0x13,
                    5,
                    rd,
                    rs1,
                    (random.below(32) | random.below(2) << 10) as i32,
                ),
                f3 => i(0x13, f3, rd, rs1, imm),
            },
            _ if random.below(2) == 0 => lui(rd, random.next()),
            _ => auipc(rd, random.next()),
        });
    }
    code.push(EBREAK);
    code
}

#[test]
#[ignore = "needs uv and PyTorch: run `just neural-tests`"]
fn random_programs_run_identically_on_both_vcpus() {
    let server = Server::start();
    let mut random = Random(0x5348_454d);
    for n in 0..20 {
        let code = bytes(&random_program(&mut random));
        let mut reference = ReferenceVcpu::new(MEM);
        let mut neural = server.vcpu();
        reference.write_mem(0, &code).unwrap();
        neural.write_mem(0, &code).unwrap();
        assert_eq!(
            neural.run(10_000),
            reference.run(10_000),
            "program {n}: exit"
        );
        assert_same(
            &format!("program {n}"),
            &reference.snapshot(),
            &neural.snapshot(),
        );
    }
}

#[test]
#[ignore = "needs uv and PyTorch: run `just neural-tests`"]
fn reference_snapshots_resume_on_the_neural_vcpu() {
    let server = Server::start();
    let code = bytes(&program![
        addi(A0, ZERO, 0),
        addi(T0, ZERO, 100),
        add(A0, A0, T0),
        addi(T0, T0, -1),
        bne(T0, ZERO, -8),
        sw(A0, ZERO, 0x800),
        EBREAK,
    ]);
    let mut reference = ReferenceVcpu::new(MEM);
    reference.write_mem(0, &code).unwrap();
    reference.run(137);
    let midway = reference.snapshot();

    let mut neural = server.vcpu();
    neural.restore(&midway).unwrap();
    assert_eq!(neural.run(10_000), reference.run(10_000));
    assert_same("resumed run", &reference.snapshot(), &neural.snapshot());
    assert_eq!(neural.reg(A0 as usize), 5050);
}

#[test]
#[ignore = "needs uv and PyTorch: run `just neural-tests`"]
fn containers_fork_and_pipe_on_the_neural_vcpu() {
    let server = Server::start();
    let msg = b"hello from the neural vcpu\n";
    let hello = elf(
        &program![
            addi(A0, ZERO, 1),
            li(A1, DATA),
            addi(A2, ZERO, msg.len() as i32),
            addi(A7, ZERO, 64),
            ECALL,
            addi(A0, ZERO, 7),
            addi(A7, ZERO, 93),
            ECALL,
        ],
        msg,
    );
    let out = run_container(server.vcpu(), &hello, &["hello"]);
    assert_eq!((out.result, out.stdout.as_slice()), (Ok(7), &msg[..]));

    // A child writes a byte into a pipe; the parent blocks reading it, then
    // reaps the child and exits with the byte plus the child's status.
    let pipeline = elf(
        &program![
            addi(A0, SP, -8),
            addi(A1, ZERO, 0),
            addi(A7, ZERO, 59), // pipe2
            ECALL,
            addi(A0, ZERO, 17),
            addi(A1, ZERO, 0),
            addi(A7, ZERO, 220), // clone
            ECALL,
            beq(A0, ZERO, 72), // the child jumps to its part
            lw(A0, SP, -8),    // parent: read(fds[0], sp - 16, 1)
            addi(A1, SP, -16),
            addi(A2, ZERO, 1),
            addi(A7, ZERO, 63),
            ECALL,
            addi(A0, ZERO, -1), // wait4(-1, sp - 20, 0, 0)
            addi(A1, SP, -20),
            addi(A2, ZERO, 0),
            addi(A3, ZERO, 0),
            addi(A7, ZERO, 260),
            ECALL,
            lbu(A0, SP, -16), // exit(byte + (status >> 8))
            lw(T0, SP, -20),
            srai(T0, T0, 8),
            add(A0, A0, T0),
            addi(A7, ZERO, 93),
            ECALL,
            addi(T0, ZERO, 60), // child: write(fds[1], "<", 1); exit(5)
            sb(T0, SP, -12),
            lw(A0, SP, -4),
            addi(A1, SP, -12),
            addi(A2, ZERO, 1),
            addi(A7, ZERO, 64),
            ECALL,
            addi(A0, ZERO, 5),
            addi(A7, ZERO, 93),
            ECALL,
        ],
        &[],
    );
    let reference = run_container(ReferenceVcpu::new(MEM), &pipeline, &["pipeline"]);
    let neural = run_container(server.vcpu(), &pipeline, &["pipeline"]);
    assert_eq!(reference.result, Ok(65));
    assert_eq!(neural.result, reference.result);
    assert_eq!(neural.steps, reference.steps);
    assert_same("pipeline", &reference.state, &neural.state);
}

#[test]
#[ignore = "needs uv and PyTorch: run `just neural-tests`"]
fn warm_images_built_on_the_reference_resume_on_the_neural_vcpu() {
    // write(1, "ready\n"), then echo one read of stdin, then exit(0)
    let mut data = b"ready\n".to_vec();
    data.resize(16 + 64, 0);
    let buf = DATA + 16;
    let echo = elf(
        &program![
            addi(A0, ZERO, 1),
            li(A1, DATA),
            addi(A2, ZERO, 6),
            addi(A7, ZERO, 64),
            ECALL,
            addi(A0, ZERO, 0),
            li(A1, buf),
            addi(A2, ZERO, 64),
            addi(A7, ZERO, 63),
            ECALL,
            addi(A2, A0, 0),
            addi(A0, ZERO, 1),
            li(A1, buf),
            addi(A7, ZERO, 64),
            ECALL,
            addi(A0, ZERO, 0),
            addi(A7, ZERO, 93),
            ECALL,
        ],
        &data,
    );
    let context = tempfile::tempdir().unwrap();
    std::fs::write(context.path().join("echo"), echo).unwrap();
    let shemfile =
        "FROM scratch\nCOPY echo /echo\nENTRYPOINT [\"/echo\"]\nWARM --until=syscall:read\n";
    std::fs::write(context.path().join("Shemfile"), shemfile).unwrap();
    let image = shem::image::build(&shem::image::BuildOptions {
        shemfile: context.path().join("Shemfile"),
        context: context.path().to_path_buf(),
        build_args: Default::default(),
        mem: Some(MEM),
    })
    .unwrap();

    let server = Server::start();
    let root = tempfile::tempdir().unwrap();
    let stdout = SharedBuf::default();
    let stdio = linux::Stdio {
        stdin: Box::new(std::io::Cursor::new(b"resumed on the model\n".to_vec())),
        stdout: Box::new(stdout.clone()),
        stderr: Box::new(std::io::sink()),
    };
    let make = |mem| Ok(NeuralVcpu::connect(&server.socket, mem)?);
    let mut container =
        shem::image::start(&image, make, &[], &[], stdio, root.path(), MEM).unwrap();
    assert_eq!(container.run(Some(MAX_STEPS)), Ok(0));
    assert_eq!(stdout.contents(), b"resumed on the model\n");
}

/// Boot `kernel` in VM mode on `vcpu` and run it until it stops. Returns how
/// it stopped, its console output, and the machine frozen at the end.
fn run_machine<V: Vcpu>(vcpu: V, kernel: &[u8]) -> (Result<Halt, VmError>, Vec<u8>, vm::Frozen) {
    let output = SharedBuf::default();
    let (_keys, input) = std::sync::mpsc::channel();
    let console = Console::new(input, Box::new(output.clone()));
    let mut machine = Machine::boot(vcpu, kernel, DEFAULT_BOOTARGS, console).unwrap();
    let halt = machine.run(Some(MAX_STEPS));
    (halt, output.contents(), machine.freeze())
}

#[test]
#[ignore = "needs uv and PyTorch: run `just neural-tests`"]
fn vm_mode_runs_identically_on_both_vcpus() {
    let server = Server::start();
    // An ecall into the trap handler and back, timer interrupts while
    // spinning, and wfi skipping idle time, with RAM at 0x80000000
    let ecall = with_handler(
        |_| program![addi(A0, ZERO, 0x41 - 11), ECALL, putc(A0), poweroff()],
        program![
            csr(2, T2, MCAUSE, ZERO),
            add(A0, A0, T2),
            csr(2, T2, MEPC, ZERO),
            addi(T2, T2, 4),
            csr(1, ZERO, MEPC, T2),
            MRET,
        ],
    );
    let kernels = [
        ("traps", ecall),
        ("timer", timer_program(10, program![beq(A2, ZERO, 0)])),
        (
            "idle",
            timer_program(2000, program![WFI, beq(A2, ZERO, -4)]),
        ),
    ];
    for (name, kernel) in kernels {
        let reference = run_machine(ReferenceVcpu::with_base(RAM, RAM_BASE), &kernel);
        let vcpu = NeuralVcpu::connect_with_base(&server.socket, RAM, RAM_BASE).unwrap();
        let neural = run_machine(vcpu, &kernel);
        assert_eq!(reference.0, Ok(Halt::Poweroff), "{name}");
        assert_eq!(neural.0, reference.0, "{name}");
        assert_eq!(neural.1, reference.1, "{name}: console output");
        assert_eq!(neural.2.host, reference.2.host, "{name}: the host's half");
        assert_same(name, &reference.2.machine, &neural.2.machine);
    }
}

/// Thaw Linux, frozen at its prompt, on `vcpu`, type a command, and run until
/// the next prompt. Returns the console output and the machine frozen there.
fn resume_linux<V: Vcpu>(vcpu: V, frozen: &vm::Frozen) -> (Vec<u8>, vm::Frozen) {
    let output = SharedBuf::default();
    let (keys, input) = std::sync::mpsc::channel();
    b"echo $((6 * 7))\n"
        .iter()
        .for_each(|b| keys.send(*b).unwrap());
    drop(keys); // so that wfi does not wait for more
    let console = Console::new(input, Box::new(output.clone()));
    let mut machine = Machine::thaw(vcpu, frozen, console).unwrap();
    let limit = Some(frozen.host.steps + 1_000_000);
    let prompt = Some(Pause::Output(b"~ # ".to_vec()));
    assert_eq!(machine.run_until(limit, prompt), Ok(Stop::Paused));
    (output.contents(), machine.freeze())
}

#[test]
#[ignore = "needs uv, PyTorch and a Linux image: run `just neural-tests`"]
fn booted_linux_runs_identically_on_both_vcpus() {
    const LINUX_RAM: u32 = 64 << 20;
    let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/linux/Image");
    let kernel = std::fs::read(&image).expect("download the image first: just linux-image");

    // Boot to the prompt on the reference vCPU, as a VM image's WARM does
    let (_keys, input) = std::sync::mpsc::channel();
    let console = Console::new(input, Box::new(std::io::sink()));
    let vcpu = ReferenceVcpu::with_base(LINUX_RAM, RAM_BASE);
    let mut machine = Machine::boot(vcpu, &kernel, DEFAULT_BOOTARGS, console).unwrap();
    let prompt = Some(Pause::Output(b"~ # ".to_vec()));
    assert_eq!(
        machine.run_until(Some(2_000_000_000), prompt),
        Ok(Stop::Paused)
    );
    let frozen = machine.freeze();

    // Then run a shell command on both vCPUs from there: about 180,000
    // instructions of kernel and BusyBox
    let server = Server::start();
    let reference = resume_linux(ReferenceVcpu::with_base(LINUX_RAM, RAM_BASE), &frozen);
    let vcpu = NeuralVcpu::connect_with_base(&server.socket, LINUX_RAM, RAM_BASE).unwrap();
    let neural = resume_linux(vcpu, &frozen);
    let text = String::from_utf8_lossy(&reference.0).into_owned();
    assert!(text.contains("\n42"), "{text}");
    assert_eq!(
        String::from_utf8_lossy(&neural.0),
        String::from_utf8_lossy(&reference.0)
    );
    assert_eq!(neural.1.host, reference.1.host);
    assert_same("linux", &reference.1.machine, &neural.1.machine);
}
