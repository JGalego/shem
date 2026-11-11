mod common;

use common::*;
use shem::container::{Container, RunError, STACK_SIZE};
use shem::elf::{ElfError, PAGE_SIZE};
use shem::reference::ReferenceVcpu;
use shem::vcpu::{ExitReason, Vcpu};

const MEM: u32 = 4 << 20;
const SYS_READ: i32 = 63;
const SYS_WRITE: i32 = 64;
const SYS_EXIT: i32 = 93;
const SYS_BRK: i32 = 214;
const SYS_MMAP: i32 = 222;

struct Run {
    result: Result<u8, RunError>,
    stdout: Vec<u8>,
    container: Container<ReferenceVcpu>,
}

fn run_with(code: Vec<u32>, data: &[u8], argv: &[&str], stdin: &[u8]) -> Run {
    let (mut container, stdout) = start(&elf(&code, data), argv, stdin, MEM).unwrap();
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
    assert_eq!(
        out.container.init().process().mm.brk(),
        BASE + PAGE_SIZE + 256
    );
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
    let start_err = |elf: &[u8]| start(elf, &["prog"], &[], MEM).err().map(|e| e.to_string());
    let compressed = elf_with_flags(&[ECALL], &[], 0x1);
    assert_eq!(
        start_err(&compressed),
        Some(ElfError::Compressed.to_string())
    );
    let hard_float = elf_with_flags(&[ECALL], &[], 0x4);
    assert_eq!(start_err(&hard_float), Some(ElfError::FloatAbi.to_string()));
    assert_eq!(start_err(b"#!/bin/sh"), Some(ElfError::NotElf.to_string()));
}

#[test]
fn mmap_returns_zeroed_pages_below_the_stack() {
    let out = run(program![
        addi(A0, ZERO, 0),     // anywhere
        li(A1, 2 * PAGE_SIZE), // length
        addi(A2, ZERO, 3),     // PROT_READ | PROT_WRITE
        addi(A3, ZERO, 0x22),  // MAP_PRIVATE | MAP_ANONYMOUS
        addi(A4, ZERO, -1),    // no file
        addi(A7, ZERO, SYS_MMAP),
        ECALL,
        addi(S0, A0, 0),
        lw(A0, S0, 4),
        exit_a0(),
    ]);
    assert_eq!(out.result, Ok(0));
    let addr = out.container.init().vcpu().reg(S0 as usize);
    assert_eq!(addr, MEM - STACK_SIZE - 2 * PAGE_SIZE);
}

#[test]
fn atomics_are_emulated_by_the_host() {
    let out = run_with(
        program![
            li(T0, DATA), // a word holding 40
            addi(T1, ZERO, 2),
            amoadd_w(A1, T1, T0),    // a1 = 40, word = 42
            lr_w(A2, T0),            // a2 = 42, and reserve the word
            sc_w(A3, T1, T0),        // succeeds: a3 = 0, word = 2
            sc_w(A4, T1, T0),        // the reservation is gone: a4 = 1
            amomaxu_w(ZERO, A1, T0), // word = max(2, 40)
            lw(A0, T0, 0),
            exit_a0(),
        ],
        &40u32.to_le_bytes(),
        &["prog"],
        &[],
    );
    assert_eq!(out.result, Ok(40));
    let regs = [A1, A2, A3, A4].map(|r| out.container.init().vcpu().reg(r as usize));
    assert_eq!(regs, [40, 42, 0, 1]);
}

#[test]
fn misaligned_atomics_fault() {
    let out = run(program![
        li(T0, DATA + 2),
        amoadd_w(A1, ZERO, T0),
        exit_a0()
    ]);
    let Err(RunError::Fault(exit)) = out.result else {
        panic!("expected a fault")
    };
    assert_eq!(exit.reason, ExitReason::MemoryFault(DATA + 2));
}

const SYS_CLONE: i32 = 220;
const SYS_WAIT4: i32 = 260;
const SYS_PIPE2: i32 = 59;
const SIGCHLD: i32 = 17;

#[test]
fn fork_and_wait_report_the_child_status() {
    let out = run(program![
        addi(A0, ZERO, SIGCHLD),
        addi(A1, ZERO, 0),
        addi(A7, ZERO, SYS_CLONE),
        ECALL,
        bne(A0, ZERO, 16), // the parent skips the child's exit(7)
        addi(A0, ZERO, 7),
        exit_a0(),
        addi(A0, ZERO, -1), // parent: wait4(-1, sp - 16, 0, 0)
        addi(A1, SP, -16),
        addi(A2, ZERO, 0),
        addi(A3, ZERO, 0),
        addi(A7, ZERO, SYS_WAIT4),
        ECALL,
        lw(A0, SP, -16),
        srai(A0, A0, 8),
        addi(A0, A0, 10),
        exit_a0(),
    ]);
    assert_eq!(out.result, Ok(17));
    assert!(out.container.task(2).is_none(), "the child was reaped");
}

#[test]
fn a_read_blocks_until_another_process_writes() {
    let out = run(program![
        addi(A0, SP, -8), // pipe2(sp - 8, 0)
        addi(A1, ZERO, 0),
        addi(A7, ZERO, SYS_PIPE2),
        ECALL,
        addi(A0, ZERO, SIGCHLD), // fork
        addi(A1, ZERO, 0),
        addi(A7, ZERO, SYS_CLONE),
        ECALL,
        beq(A0, ZERO, 36), // the child jumps ahead
        lw(A0, SP, -8),    // parent: read(fds[0], sp - 16, 1), then exit with the byte
        addi(A1, SP, -16),
        addi(A2, ZERO, 1),
        addi(A7, ZERO, SYS_READ),
        ECALL,
        lbu(A0, SP, -16),
        exit_a0(),
        addi(T0, ZERO, 65), // child: write(fds[1], "A", 1), then exit(0)
        sb(T0, SP, -12),
        lw(A0, SP, -4),
        addi(A1, SP, -12),
        addi(A2, ZERO, 1),
        addi(A7, ZERO, SYS_WRITE),
        ECALL,
        addi(A0, ZERO, 0),
        exit_a0(),
    ]);
    assert_eq!(out.result, Ok(65));
}

#[test]
fn processes_that_all_block_are_a_deadlock() {
    let out = run(program![
        addi(A0, SP, -8),
        addi(A1, ZERO, 0),
        addi(A7, ZERO, SYS_PIPE2),
        ECALL,
        lw(A0, SP, -8), // read from a pipe whose only writer is this process
        addi(A1, SP, -16),
        addi(A2, ZERO, 1),
        addi(A7, ZERO, SYS_READ),
        ECALL,
        exit_a0(),
    ]);
    assert_eq!(out.result, Err(RunError::Deadlock));
}

const SYS_KILL: i32 = 129;
const SYS_RT_SIGACTION: i32 = 134;
const SIGUSR1: i32 = 10;

#[test]
fn signal_handlers_run_and_return_through_sigreturn() {
    let handler = 84; // byte offset of the handler from the entry point
    let out = run_with(
        program![
            auipc(T0, 0), // sigaction(SIGUSR1, {handler, 0, 0}, NULL, 8)
            addi(T0, T0, handler),
            sw(T0, SP, -16),
            sw(ZERO, SP, -12),
            sw(ZERO, SP, -8),
            sw(ZERO, SP, -4),
            addi(A0, ZERO, SIGUSR1),
            addi(A1, SP, -16),
            addi(A2, ZERO, 0),
            addi(A3, ZERO, 8),
            addi(A7, ZERO, SYS_RT_SIGACTION),
            ECALL,
            addi(A0, ZERO, 1), // kill(1, SIGUSR1): signal ourselves
            addi(A1, ZERO, SIGUSR1),
            addi(A7, ZERO, SYS_KILL),
            ECALL,
            li(T2, DATA), // exit with the word the handler stored
            lw(A0, T2, 0),
            exit_a0(),
            addi(T1, ZERO, 42), // the handler: store 42, then return
            li(T2, DATA),
            sw(T1, T2, 0),
            jalr(ZERO, RA, 0),
        ],
        &[0; 4],
        &["prog"],
        &[],
    );
    assert_eq!(out.result, Ok(42));
}
