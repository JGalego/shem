mod common;

use common::*;
use shem::container::{Container, RunError};
use shem::elf::{ElfError, PAGE_SIZE};
use shem::linux::Stdio;
use shem::reference::ReferenceVcpu;
use shem::vcpu::ExitReason;

const MEM: u32 = 4 << 20;
const SYS_READ: i32 = 63;
const SYS_WRITE: i32 = 64;
const SYS_EXIT: i32 = 93;
const SYS_BRK: i32 = 214;

struct Run {
    result: Result<u8, RunError>,
    stdout: Vec<u8>,
    container: Container<ReferenceVcpu>,
}

fn start(
    elf: &[u8],
    argv: &[&str],
    stdin: &[u8],
) -> Result<(Container<ReferenceVcpu>, SharedBuf), shem::container::StartError> {
    let stdout = SharedBuf::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::Cursor::new(stdin.to_vec())),
        stdout: Box::new(stdout.clone()),
        stderr: Box::new(std::io::sink()),
    };
    let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    let container = Container::from_elf(ReferenceVcpu::new(MEM), elf, &argv, &[], stdio)?;
    Ok((container, stdout))
}

fn run_with(code: Vec<u32>, data: &[u8], argv: &[&str], stdin: &[u8]) -> Run {
    let (mut container, stdout) = start(&elf(&code, data), argv, stdin).unwrap();
    let result = container.run(Some(1_000_000));
    Run {
        result,
        stdout: stdout.contents(),
        container,
    }
}

fn run(code: Vec<u32>) -> Run {
    run_with(code, &[], &["prog"], &[])
}

/// exit(a0)
fn exit_a0() -> [u32; 2] {
    [addi(A7, ZERO, SYS_EXIT), ECALL]
}

#[test]
fn hello_world() {
    let msg = b"hello, shem\n";
    let out = run_with(
        program![
            addi(A0, ZERO, 1),
            li(A1, DATA),
            addi(A2, ZERO, msg.len() as i32),
            addi(A7, ZERO, SYS_WRITE),
            ECALL,
            addi(A0, ZERO, 7),
            exit_a0(),
        ],
        msg,
        &["prog"],
        &[],
    );
    assert_eq!(out.result, Ok(7));
    assert_eq!(out.stdout, msg);
}

#[test]
fn argc_and_argv_are_on_the_stack() {
    let out = run_with(
        program![
            lw(A1, SP, 8), // argv[1]
            addi(A0, ZERO, 1),
            addi(A2, ZERO, 3),
            addi(A7, ZERO, SYS_WRITE),
            ECALL,
            lw(A0, SP, 0), // argc
            exit_a0(),
        ],
        &[],
        &["prog", "xyz", "q"],
        &[],
    );
    assert_eq!(out.result, Ok(3));
    assert_eq!(out.stdout, b"xyz");
}

#[test]
fn echoes_stdin() {
    let out = run_with(
        program![
            addi(SP, SP, -16),
            addi(A0, ZERO, 0),
            addi(A1, SP, 0),
            addi(A2, ZERO, 16),
            addi(A7, ZERO, SYS_READ),
            ECALL,
            addi(A2, A0, 0),
            addi(A0, ZERO, 1),
            addi(A1, SP, 0),
            addi(A7, ZERO, SYS_WRITE),
            ECALL,
            addi(A0, ZERO, 0),
            exit_a0(),
        ],
        &[],
        &["prog"],
        b"ping",
    );
    assert_eq!(out.result, Ok(0));
    assert_eq!(out.stdout, b"ping");
}

#[test]
fn brk_grows_the_heap() {
    let out = run(program![
        addi(A0, ZERO, 0),
        addi(A7, ZERO, SYS_BRK),
        ECALL, // a0 = current break
        addi(S0, A0, 0),
        addi(A0, S0, 256),
        ECALL, // break += 256
        addi(T0, ZERO, 99),
        sw(T0, S0, 0),
        lw(A0, S0, 0),
        exit_a0(),
    ]);
    assert_eq!(out.result, Ok(99));
    // The whole file fits in one page, so the heap starts on the next one.
    assert_eq!(out.container.linux.brk(), BASE + PAGE_SIZE + 256);
}

#[test]
fn unsupported_syscalls_return_enosys() {
    let out = run(program![
        addi(A7, ZERO, 999),
        ECALL,
        sub(A0, ZERO, A0),
        exit_a0()
    ]);
    assert_eq!(out.result, Ok(38));
    assert_eq!(out.container.linux.unsupported.get(&999), Some(&1));
}

#[test]
fn bad_guest_pointers_return_efault() {
    let out = run(program![
        addi(A0, ZERO, 1),
        li(A1, 0xffff_fff0),
        addi(A2, ZERO, 4),
        addi(A7, ZERO, SYS_WRITE),
        ECALL,
        sub(A0, ZERO, A0),
        exit_a0(),
    ]);
    assert_eq!(out.result, Ok(14));
}

#[test]
fn faults_stop_the_container() {
    let out = run(vec![0]);
    let Err(RunError::Fault(exit)) = out.result else {
        panic!("expected a fault")
    };
    assert_eq!(exit.reason, ExitReason::IllegalInstruction(0));
}

#[test]
fn step_limit_stops_runaway_guests() {
    let out = run(vec![jal(ZERO, 0)]);
    assert_eq!(out.result, Err(RunError::StepLimit(1_000_000)));
    assert_eq!(out.container.steps, 1_000_000);
}

#[test]
fn rejects_executables_it_cannot_run() {
    let start_err = |elf: &[u8]| start(elf, &["prog"], &[]).err().map(|e| e.to_string());
    let compressed = elf_with_flags(&[ECALL], &[], 0x1);
    assert_eq!(
        start_err(&compressed),
        Some(ElfError::Compressed.to_string())
    );
    let hard_float = elf_with_flags(&[ECALL], &[], 0x4);
    assert_eq!(start_err(&hard_float), Some(ElfError::FloatAbi.to_string()));
    assert_eq!(start_err(b"#!/bin/sh"), Some(ElfError::NotElf.to_string()));
}
