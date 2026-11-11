//! VM mode: the host as the machine around an RV32IM vCPU.

mod common;

use std::sync::mpsc::{Sender, channel};
use std::time::{Duration, Instant};

use common::*;
use shem::reference::ReferenceVcpu;
use shem::vm::{self, Console, DEFAULT_BOOTARGS, Halt, Machine, RAM_BASE, VmError};

const RAM: u32 = 64 << 10;
const MTVEC: u32 = 0x305;
const MSTATUS: u32 = 0x300;
const MIE: u32 = 0x304;
const MEPC: u32 = 0x341;
const MCAUSE: u32 = 0x342;
const MTVAL: u32 = 0x343;
const MSCRATCH: u32 = 0x340;
const MISA: u32 = 0x301;
const UART: u32 = 0x1000_0000;
const MTIMECMP: u32 = 0x1100_4000;
const MTIME: u32 = 0x1100_bff8;
const SYSCON: u32 = 0x1110_0000;

/// Print the low byte of `reg` on the UART.
fn putc(reg: u32) -> [u32; 3] {
    let [upper, low] = li(T1, UART);
    [upper, low, sb(reg, T1, 0)]
}

/// Power the machine off through the syscon.
fn poweroff() -> Vec<u32> {
    program![li(T1, SYSCON), li(T2, 0x5555), sw(T2, T1, 0)]
}

/// `main`, given the handler's offset from the start, followed by `handler`;
/// main installs the handler as the trap vector first.
fn with_handler(main: impl Fn(i32) -> Vec<u32>, handler: Vec<u32>) -> Vec<u8> {
    let install = |offset: i32| {
        let mut code = program![auipc(T0, 0), addi(T0, T0, offset), csr(1, ZERO, MTVEC, T0)];
        code.extend(main(offset));
        code
    };
    let offset = install(0).len() as i32 * 4;
    bytes(&[install(offset), handler].concat())
}

fn boot(kernel: &[u8]) -> (Machine<ReferenceVcpu>, SharedBuf, Sender<u8>) {
    let output = SharedBuf::default();
    let (keys, input) = channel();
    let console = Console::new(input, Box::new(output.clone()));
    let vcpu = ReferenceVcpu::with_base(RAM, RAM_BASE);
    let machine = Machine::boot(vcpu, kernel, DEFAULT_BOOTARGS, console).unwrap();
    (machine, output, keys)
}

fn run(kernel: &[u8]) -> (Result<Halt, VmError>, String, u64) {
    let (mut machine, output, _keys) = boot(kernel);
    let halt = machine.run(Some(1_000_000));
    let output = output.contents();
    let text = String::from_utf8(output.clone()).unwrap_or_else(|_| format!("{output:?}"));
    (halt, text, machine.steps)
}

#[test]
fn csrs_read_back_and_the_uart_prints() {
    let code = program![
        li(A0, 0x41),
        csr(1, ZERO, MSCRATCH, A0), // mscratch = 'A'
        csr(2, A1, MSCRATCH, ZERO), // a1 = mscratch
        putc(A1),
        csr(2, A2, MISA, ZERO), // misa: RV32 with I, M and A
        srai(A2, A2, 30),
        addi(A2, A2, 0x30), // '1' for XLEN 32
        putc(A2),
        poweroff(),
    ];
    let (halt, out, _) = run(&bytes(&code));
    assert_eq!((halt, out.as_str()), (Ok(Halt::Poweroff), "A1"));
}

#[test]
fn ecall_traps_to_the_vector_and_mret_returns() {
    // a0 starts at 'A' - 11, and the handler adds mcause, 11 for an ecall
    // from machine mode, then returns past the ecall.
    let main = |_| program![addi(A0, ZERO, 0x41 - 11), ECALL, putc(A0), poweroff()];
    let handler = program![
        csr(2, T2, MCAUSE, ZERO),
        add(A0, A0, T2),
        csr(2, T2, MEPC, ZERO),
        addi(T2, T2, 4),
        csr(1, ZERO, MEPC, T2),
        MRET,
    ];
    assert_eq!(run(&with_handler(main, handler)).1, "A");
}

#[test]
fn illegal_instructions_and_bad_accesses_trap_with_details() {
    // The handler prints mcause as a digit and stops.
    let handler = program![
        csr(2, A0, MCAUSE, ZERO),
        addi(A0, A0, 0x30),
        putc(A0),
        poweroff()
    ];
    let illegal = with_handler(|_| program![0xffff_ffff], handler.clone());
    assert_eq!(run(&illegal).1, "2");
    let nowhere = with_handler(
        |_| program![li(A1, 0x4000_0000), lw(A2, A1, 0)],
        handler.clone(),
    );
    assert_eq!(run(&nowhere).1, "5", "load access fault");

    let mtval = program![
        csr(2, A0, MTVAL, ZERO), // the illegal instruction's word
        srai(A0, A0, 24),
        addi(A0, A0, 0x40), // 0x12 + 0x40 = 'R'
        putc(A0),
        poweroff(),
    ];
    let word = 0x1234_5678 | 0x7f; // opcode 0x7f is reserved
    assert_eq!(run(&with_handler(|_| program![word], mtval)).1, "R");
}

/// Arm the timer `ticks` ahead, enable its interrupt, then run `wait` until
/// the handler sets a2. (a1 starts out holding the device tree's address.)
fn timer_program(ticks: i32, wait: Vec<u32>) -> Vec<u8> {
    let main = move |_| {
        program![
            li(T1, MTIME),
            lw(T2, T1, 0),
            addi(T2, T2, ticks),
            li(T1, MTIMECMP),
            sw(T2, T1, 0),
            sw(ZERO, T1, 4),
            addi(T2, ZERO, 0x80),
            csr(1, ZERO, MIE, T2),    // mie.MTIE
            csr(6, ZERO, MSTATUS, 8), // mstatus.MIE
        ]
        .into_iter()
        .chain(wait.clone())
        .chain(program![addi(A2, A2, 0x40), putc(A2), poweroff()])
        .collect()
    };
    let handler = program![
        csr(2, A2, MCAUSE, ZERO), // 0x80000007
        andi_(A2, A2, 0xf),       // 7
        csr(1, ZERO, MIE, ZERO),  // no more timer interrupts
        MRET,
    ];
    with_handler(main, handler)
}

fn andi_(rd: u32, rs1: u32, imm: i32) -> u32 {
    i(0x13, 7, rd, rs1, imm)
}

#[test]
fn timer_interrupts_arrive_and_wfi_skips_ahead() {
    // Spin until the handler sets a1.
    let spin = timer_program(50, program![beq(A2, ZERO, 0)]);
    let (halt, out, steps) = run(&spin);
    assert_eq!((halt, out.as_str()), (Ok(Halt::Poweroff), "G")); // '@' + 7
    assert!(steps > 50 * 100, "50 ticks of 100 instructions each passed");

    // Wait a virtual minute: wfi skips the idle time instead of running it.
    let idle = timer_program(2000, program![WFI, beq(A2, ZERO, -4)]);
    let (_, out, steps) = run(&idle);
    assert_eq!(out, "G");
    assert!(steps < 100, "only {steps} instructions ran");
}

#[test]
fn the_device_tree_describes_this_machine() {
    let tree = vm::device_tree(0x0100_0000, DEFAULT_BOOTARGS);
    assert_eq!(&tree[..4], &0xd00d_feedu32.to_be_bytes());
    assert_eq!(
        u32::from_be_bytes(tree[4..8].try_into().unwrap()) as usize,
        tree.len()
    );
    let text = String::from_utf8_lossy(&tree);
    for needle in [
        "rv32ima",
        "riscv,none",
        "ns16850",
        "riscv,clint0",
        "syscon-poweroff",
        "console=ttyS0",
    ] {
        assert!(text.contains(needle), "{needle}");
    }
}

fn wait_for(output: &SharedBuf, needle: &str, since: usize) -> usize {
    let started = Instant::now();
    loop {
        let text = String::from_utf8_lossy(&output.contents()).into_owned();
        if let Some(at) = text[since.min(text.len())..].find(needle) {
            return since + at + needle.len();
        }
        assert!(
            started.elapsed() < Duration::from_secs(300),
            "never saw {needle:?}; the console said:\n{text}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
#[ignore = "needs a Linux image: run `just vm-tests`"]
fn linux_boots_to_a_shell_and_powers_off() {
    let image = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/linux/Image");
    let kernel = std::fs::read(&image).expect("download the image first: just vm-tests");
    let output = SharedBuf::default();
    let (keys, input) = channel();
    let console = Console::new(input, Box::new(output.clone()));
    let vcpu = ReferenceVcpu::with_base(64 << 20, RAM_BASE);
    let mut machine = Machine::boot(vcpu, &kernel, DEFAULT_BOOTARGS, console).unwrap();
    let running = std::thread::spawn(move || machine.run(Some(20_000_000_000)));

    let type_line = |line: &str| {
        line.bytes()
            .chain(*b"\n")
            .for_each(|b| keys.send(b).unwrap())
    };
    let mut seen = wait_for(&output, "~ # ", 0);
    type_line("uname -sm");
    seen = wait_for(&output, "Linux riscv32", seen);
    type_line("echo $((6 * 7))");
    seen = wait_for(&output, "42", seen);
    wait_for(&output, "~ # ", seen);
    type_line("poweroff");
    assert_eq!(running.join().unwrap(), Ok(Halt::Poweroff));
}
