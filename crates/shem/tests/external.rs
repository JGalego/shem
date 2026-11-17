//! The reference interpreter against an emulator that is not shem's: random
//! RV32IM programs, run here and on QEMU's user-mode emulator, whose final
//! registers must be the same. The riscv-tests say that each instruction does
//! what it should in the cases they chose; this says the same for the ones a
//! random program picks, on someone else's implementation of the ISA.
//!
//! The comparison needs `qemu-riscv32` (the `qemu-user` package), so it is
//! ignored by default and CI runs it; `SHEM_QEMU` names another binary.

mod common;

use std::path::PathBuf;
use std::process::Command;

use common::*;

/// A small, fast generator whose sequences are the same on every machine.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u32) as usize]
    }

    /// A value that is often one an implementation gets wrong.
    fn value(&mut self) -> u32 {
        match self.below(10) {
            0 => 0,
            1 => 1,
            2 => u32::MAX,
            3 => 0x7fff_ffff,
            4 => 0x8000_0000,
            5 => self.below(64),
            6 => self.below(64).wrapping_neg(),
            _ => self.next() ^ self.next() << 16,
        }
    }
}

/// The registers programs compute in: everything but x0, sp (2) and gp (3), which holds the data's address.
const REGS: [u32; 28] = [
    1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30,
];
const GP: u32 = 3;
const DATA_BYTES: usize = 256;

fn s_type(f3: u32, rs2: u32, rs1: u32, imm: u32) -> u32 {
    (imm >> 5 & 0x7f) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (imm & 0x1f) << 7 | 0x23
}

fn b_type(f3: u32, rs1: u32, rs2: u32, offset: u32) -> u32 {
    (offset >> 12 & 1) << 31
        | (offset >> 5 & 0x3f) << 25
        | rs2 << 20
        | rs1 << 15
        | f3 << 12
        | (offset >> 1 & 0xf) << 8
        | (offset >> 11 & 1) << 7
        | 0x63
}

/// One random instruction that cannot fault: no division by anything but
/// whatever the registers hold (which RISC-V defines for zero and overflow),
/// loads and stores inside the data, branches over the next instruction.
fn instruction(rng: &mut Rng) -> u32 {
    let rd = rng.pick(&REGS);
    // Sources are mostly registers the program has computed, and sometimes zero
    let source = |rng: &mut Rng| {
        if rng.below(8) == 0 {
            0
        } else {
            rng.pick(&REGS)
        }
    };
    let (rs1, rs2) = (source(rng), source(rng));
    match rng.below(100) {
        0..=39 => {
            // The base ISA's register operations, and the M extension's
            let (funct7, funct3) = rng.pick(&[
                (0x00, 0),
                (0x20, 0),
                (0x00, 1),
                (0x00, 2),
                (0x00, 3),
                (0x00, 4),
                (0x00, 5),
                (0x20, 5),
                (0x00, 6),
                (0x00, 7),
                (0x01, 0),
                (0x01, 1),
                (0x01, 2),
                (0x01, 3),
                (0x01, 4),
                (0x01, 5),
                (0x01, 6),
                (0x01, 7),
            ]);
            r(funct7, funct3, rd, rs1, rs2)
        }
        40..=64 => {
            let funct3 = rng.pick(&[0, 2, 3, 4, 6, 7]);
            i(0x13, funct3, rd, rs1, rng.value() as i32 >> rng.below(21))
        }
        65..=74 => {
            let (funct3, high) = rng.pick(&[(1, 0), (5, 0), (5, 0x400)]);
            i(0x13, funct3, rd, rs1, (high | rng.below(32)) as i32)
        }
        75..=79 => {
            let upper = rng.next() & 0xffff_f000;
            if rng.below(2) == 0 {
                upper | rd << 7 | 0x37
            } else {
                upper | rd << 7 | 0x17
            }
        }
        80..=88 => {
            let (funct3, size) = rng.pick(&[(0, 1), (1, 2), (2, 4), (4, 1), (5, 2)]);
            i(
                0x03,
                funct3,
                rd,
                GP,
                (rng.below((DATA_BYTES as u32 - 4) / size) * size) as i32,
            )
        }
        89..=94 => {
            let (funct3, size) = rng.pick(&[(0, 1), (1, 2), (2, 4)]);
            s_type(
                funct3,
                rs2,
                GP,
                rng.below((DATA_BYTES as u32 - 4) / size) * size,
            )
        }
        _ => b_type(rng.pick(&[0, 1, 4, 5, 6, 7]), rs1, rs2, 8),
    }
}

/// A program of `ops` random instructions over registers it fills with
/// random values, then stores every register for `write` to give back.
fn program(seed: u64, ops: usize) -> Vec<u8> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut code: Vec<u32> = Vec::new();
    code.extend(li(GP, DATA));
    for &reg in &REGS {
        code.extend(li(reg, rng.value()));
    }
    for _ in 0..ops {
        let word = instruction(&mut rng);
        code.push(word);
        // A branch skips the instruction after it, which must be one that is not itself a branch
        if word & 0x7f == 0x63 {
            let (rd, rs1) = (rng.pick(&REGS), rng.pick(&REGS));
            code.push(r(0x00, 0, rd, rs1, rs1));
        }
    }
    for (slot, &reg) in REGS.iter().enumerate() {
        code.push(sw(reg, GP, 4 * slot as i32));
    }
    code.extend([addi(A0, ZERO, 1)]);
    code.extend(li(A1, DATA));
    code.extend([
        addi(A2, ZERO, (4 * REGS.len()) as i32),
        addi(A7, ZERO, 64),
        ECALL,
    ]);
    code.extend([addi(A0, ZERO, 0), addi(A7, ZERO, 93), ECALL]);
    let data: Vec<u8> = (0..DATA_BYTES).map(|_| rng.next() as u8).collect();
    elf(&code, &data)
}

/// What the program writes when shem runs it.
fn on_shem(program: &[u8]) -> Vec<u8> {
    let (mut container, stdout) = start(program, &["p"], &[], 1 << 20).unwrap();
    assert_eq!(container.run(Some(1_000_000)), Ok(0));
    stdout.contents()
}

fn qemu() -> Option<PathBuf> {
    let name = std::env::var("SHEM_QEMU").unwrap_or_else(|_| "qemu-riscv32".into());
    let found = Command::new(&name).arg("--version").output().ok()?;
    found.status.success().then(|| PathBuf::from(name))
}

/// What the program writes when QEMU runs it.
fn on_qemu(qemu: &std::path::Path, program: &[u8], dir: &std::path::Path) -> Vec<u8> {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join("program");
    std::fs::write(&path, program).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    let out = Command::new(qemu).arg(&path).output().unwrap();
    assert!(
        out.status.success(),
        "qemu: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

#[test]
fn random_programs_end_with_every_register_stored_on_shem() {
    // The generator, on its own: every program runs to its end and says what all the registers hold
    for seed in 0..20 {
        let said = on_shem(&program(seed, 150));
        assert_eq!(said.len(), 4 * REGS.len(), "seed {seed}");
    }
    // And the same program gives the same answer
    assert_eq!(on_shem(&program(7, 150)), on_shem(&program(7, 150)));
    assert_ne!(on_shem(&program(7, 150)), on_shem(&program(8, 150)));
}

#[test]
#[ignore = "needs qemu-riscv32 (the qemu-user package), which CI installs"]
fn the_reference_vcpu_agrees_with_qemu_on_random_programs() {
    let qemu = qemu().expect("qemu-riscv32 (or SHEM_QEMU) is needed");
    let dir = tempfile::tempdir().unwrap();
    let count: u64 = std::env::var("SHEM_RANDOM_PROGRAMS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(300);
    for seed in 0..count {
        let program = program(seed, 200);
        let (ours, theirs) = (on_shem(&program), on_qemu(&qemu, &program, dir.path()));
        if ours != theirs {
            let word = |bytes: &[u8], at: usize| {
                u32::from_le_bytes(bytes[4 * at..4 * at + 4].try_into().unwrap())
            };
            let at = (0..REGS.len())
                .find(|&slot| word(&ours, slot) != word(&theirs, slot))
                .unwrap_or(0);
            panic!(
                "seed {seed}: x{} is {:#010x} on shem and {:#010x} on QEMU",
                REGS[at],
                word(&ours, at),
                word(&theirs, at)
            );
        }
    }
}
