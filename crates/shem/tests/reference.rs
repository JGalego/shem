mod common;

use common::*;
use shem::reference::ReferenceVcpu;
use shem::vcpu::{Exit, ExitReason, MachineState, SnapshotError, Vcpu};

const MEM: u32 = 64 << 10;

fn load(code: &[u32]) -> ReferenceVcpu {
    let mut vcpu = ReferenceVcpu::new(MEM);
    vcpu.write_mem(0, &bytes(code)).unwrap();
    vcpu
}

/// Run `code` followed by an EBREAK, which the tests use as "halt".
fn run(code: Vec<u32>) -> (ReferenceVcpu, Exit) {
    let mut vcpu = load(&[code, vec![EBREAK]].concat());
    let exit = vcpu.run(10_000);
    assert_eq!(
        exit.reason,
        ExitReason::Ebreak,
        "program did not reach its final EBREAK"
    );
    (vcpu, exit)
}

fn reg(vcpu: &ReferenceVcpu, index: u32) -> u32 {
    vcpu.reg(index as usize)
}

#[test]
fn arithmetic_and_x0() {
    let (vcpu, _) = run(program![
        li(A0, 40),
        addi(A1, ZERO, 2),
        add(A2, A0, A1),
        sub(A3, A1, A0),
        addi(ZERO, ZERO, 5),
        li(T0, 0x8000_0000),
        srai(T1, T0, 4),
        sra(S0, T0, A1),
    ]);
    assert_eq!(reg(&vcpu, A2), 42);
    assert_eq!(reg(&vcpu, A3), (-38i32) as u32);
    assert_eq!(reg(&vcpu, ZERO), 0);
    assert_eq!(reg(&vcpu, T1), 0xf800_0000);
    assert_eq!(reg(&vcpu, S0), 0xe000_0000);
}

#[test]
fn lui_and_auipc() {
    let (vcpu, _) = run(program![
        lui(A0, 0x1234_5000),
        addi(ZERO, ZERO, 0),
        auipc(A1, 0x1000)
    ]);
    assert_eq!(reg(&vcpu, A0), 0x1234_5000);
    assert_eq!(reg(&vcpu, A1), 0x1008);
}

#[test]
fn loop_sums_one_to_ten() {
    let (vcpu, exit) = run(program![
        addi(A0, ZERO, 0),
        addi(T0, ZERO, 10),
        add(A0, A0, T0), // loop:
        addi(T0, T0, -1),
        bne(T0, ZERO, -8),
    ]);
    assert_eq!(reg(&vcpu, A0), 55);
    assert_eq!(exit.steps, 2 + 3 * 10);
}

#[test]
fn loads_sign_and_zero_extend() {
    let (vcpu, _) = run(program![
        li(T0, 0x1000),
        addi(T1, ZERO, -1),
        sb(T1, T0, 0),
        lb(A0, T0, 0),
        lbu(A1, T0, 0),
        sw(T1, T0, -4),
        lw(A2, T0, -4),
    ]);
    assert_eq!(reg(&vcpu, A0), 0xffff_ffff);
    assert_eq!(reg(&vcpu, A1), 0xff);
    assert_eq!(reg(&vcpu, A2), 0xffff_ffff);
}

#[test]
fn m_extension_including_edge_cases() {
    type Op = fn(u32, u32, u32) -> u32;
    let min = 0x8000_0000;
    let cases: [(Op, u32, u32, u32); 14] = [
        (mul, 7, -3i32 as u32, -21i32 as u32),
        (mulh, -1i32 as u32, -1i32 as u32, 0),
        (mulh, min, min, 0x4000_0000),
        (mulhu, u32::MAX, u32::MAX, 0xffff_fffe),
        (mulhsu, -1i32 as u32, u32::MAX, u32::MAX),
        (div, -7i32 as u32, 2, -3i32 as u32),
        (rem, -7i32 as u32, 2, -1i32 as u32),
        (divu, 7, 2, 3),
        (remu, 7, 2, 1),
        (div, 5, 0, u32::MAX),
        (divu, 5, 0, u32::MAX),
        (rem, 5, 0, 5),
        (div, min, -1i32 as u32, min),
        (rem, min, -1i32 as u32, 0),
    ];
    for (op, x, y, expected) in cases {
        let (vcpu, _) = run(program![li(A0, x), li(A1, y), op(A2, A0, A1)]);
        assert_eq!(reg(&vcpu, A2), expected, "{x:#x} op {y:#x}");
    }
}

#[test]
fn jal_and_jalr_call_and_return() {
    let mut vcpu = load(&[
        jal(RA, 8),         // 0: call 8
        EBREAK,             // 4: halt after return
        addi(A0, ZERO, 42), // 8
        jalr(ZERO, RA, 0),  // 12: return
    ]);
    let exit = vcpu.run(100);
    assert_eq!((exit.reason, exit.pc), (ExitReason::Ebreak, 4));
    assert_eq!(reg(&vcpu, A0), 42);
    assert_eq!(reg(&vcpu, RA), 4);
}

#[test]
fn jumps_backwards() {
    let mut vcpu = load(&[
        jal(ZERO, 12),     // 0: forward to 12
        addi(A0, ZERO, 1), // 4
        EBREAK,            // 8
        jal(ZERO, -8),     // 12: back to 4
    ]);
    let exit = vcpu.run(100);
    assert_eq!((exit.reason, exit.pc), (ExitReason::Ebreak, 8));
    assert_eq!(reg(&vcpu, A0), 1);
}

#[test]
fn branch_not_taken_falls_through() {
    let (vcpu, _) = run(program![
        addi(T0, ZERO, 1),
        beq(T0, ZERO, 8),
        addi(A0, ZERO, 7)
    ]);
    assert_eq!(reg(&vcpu, A0), 7);
}

#[test]
fn ecall_exits_without_retiring() {
    let mut vcpu = load(&[addi(A7, ZERO, 93), ECALL]);
    let exit = vcpu.run(100);
    assert_eq!(
        exit,
        Exit {
            reason: ExitReason::Ecall,
            pc: 4,
            steps: 1
        }
    );
    assert_eq!(vcpu.instret(), 1);
}

#[test]
fn budget_exit_is_resumable() {
    let code = program![
        addi(A0, ZERO, 0),
        addi(T0, ZERO, 10),
        add(A0, A0, T0),
        addi(T0, T0, -1),
        bne(T0, ZERO, -8),
        EBREAK,
    ];
    let mut vcpu = load(&code);
    assert_eq!(vcpu.run(5).reason, ExitReason::Budget);
    assert_eq!(vcpu.run(1000).reason, ExitReason::Ebreak);
    assert_eq!(reg(&vcpu, A0), 55);
    assert_eq!(vcpu.instret(), 32);
}

#[test]
fn illegal_instructions_exit_with_the_instruction_word() {
    let csrrs_cycle = 0xc000_2573;
    for word in [0x0000_0000, csrrs_cycle, sub(A0, A0, A0) | 1 << 26] {
        let mut vcpu = load(&[word]);
        let exit = vcpu.run(10);
        assert_eq!(
            exit,
            Exit {
                reason: ExitReason::IllegalInstruction(word),
                pc: 0,
                steps: 0
            }
        );
    }
}

#[test]
fn out_of_range_access_faults_before_any_change() {
    let mut vcpu = load(&program![li(T0, MEM), addi(A0, ZERO, 1), lw(A0, T0, 0)]);
    let exit = vcpu.run(10);
    assert_eq!(exit.reason, ExitReason::MemoryFault(MEM));
    assert_eq!(exit.pc, 12);
    assert_eq!(reg(&vcpu, A0), 1);
}

#[test]
fn snapshot_round_trips_through_bytes() {
    let mut vcpu = load(&program![
        addi(T0, ZERO, 10),
        addi(T0, T0, -1),
        bne(T0, ZERO, -4),
        EBREAK
    ]);
    vcpu.run(7);
    let snapshot = vcpu.snapshot();
    assert_eq!(
        MachineState::decode(&snapshot.encode()),
        Ok(snapshot.clone())
    );

    let first = vcpu.run(1000);
    let after_first = vcpu.snapshot();
    vcpu.restore(&snapshot).unwrap();
    let second = vcpu.run(1000);
    assert_eq!(first, second);
    assert_eq!(vcpu.snapshot(), after_first);
}

#[test]
fn snapshots_only_restore_onto_the_same_memory_size() {
    let snapshot = ReferenceVcpu::new(MEM).snapshot();
    let err = ReferenceVcpu::new(MEM * 2).restore(&snapshot).unwrap_err();
    assert!(matches!(err, SnapshotError::MemorySize { .. }));
    assert_eq!(MachineState::decode(b"nope"), Err(SnapshotError::Truncated));
}
