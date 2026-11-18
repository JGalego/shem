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

/// The compressed instructions a random program may use: the register and
/// immediate ones of `data/compressed.txt`, which cannot fault, and leave
/// gp (x3, the data's address) and sp (whose value is the host's) alone.
fn compressed_pool() -> Vec<u16> {
    const FORMS: [&str; 14] = [
        "c.addi ", "c.li ", "c.lui ", "c.slli ", "c.srli ", "c.srai ", "c.andi ", "c.mv ",
        "c.add ", "c.sub ", "c.xor ", "c.or ", "c.and ", "c.nop",
    ];
    include_str!("data/compressed.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let half = u16::from_str_radix(fields.next()?, 16).ok()?;
            let text = line.splitn(3, ' ').nth(2)?;
            // gp holds the data's address, and sp is where QEMU puts the stack and shem does not
            let uses_gp_or_sp = text
                .split([' ', ','])
                .any(|token| token == "x3" || token == "x2");
            (FORMS.iter().any(|form| text.starts_with(form)) && !uses_gp_or_sp).then_some(half)
        })
        .collect()
}

/// As `program`, with compressed instructions among the others, and 32-bit ones on any halfword.
fn program_c(seed: u64, ops: usize) -> Vec<u8> {
    let pool = compressed_pool();
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut halves: Vec<u16> = Vec::new();
    let full = |code: &mut Vec<u16>, word: u32| code.extend([word as u16, (word >> 16) as u16]);
    for word in li(GP, DATA) {
        full(&mut halves, word);
    }
    for &reg in &REGS {
        for word in li(reg, rng.value()) {
            full(&mut halves, word);
        }
    }
    for _ in 0..ops {
        if rng.below(2) == 0 {
            halves.push(rng.pick(&pool));
            continue;
        }
        let word = instruction(&mut rng);
        full(&mut halves, word);
        if word & 0x7f == 0x63 {
            let (rd, rs1) = (rng.pick(&REGS), rng.pick(&REGS));
            full(&mut halves, r(0x00, 0, rd, rs1, rs1));
        }
    }
    let mut tail: Vec<u32> = Vec::new();
    for (slot, &reg) in REGS.iter().enumerate() {
        tail.push(sw(reg, GP, 4 * slot as i32));
    }
    tail.extend([addi(A0, ZERO, 1)]);
    tail.extend(li(A1, DATA));
    tail.extend([
        addi(A2, ZERO, (4 * REGS.len()) as i32),
        addi(A7, ZERO, 64),
        ECALL,
    ]);
    tail.extend([addi(A0, ZERO, 0), addi(A7, ZERO, 93), ECALL]);
    for word in tail {
        full(&mut halves, word);
    }
    if halves.len() & 1 == 1 {
        halves.push(0x0001); // c.nop, to fill the last word
    }
    let code: Vec<u32> = halves
        .chunks(2)
        .map(|pair| u32::from(pair[0]) | u32::from(pair[1]) << 16)
        .collect();
    let data: Vec<u8> = (0..DATA_BYTES).map(|_| rng.next() as u8).collect();
    elf_with_flags(&code, &data, 0x1)
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
    let path = dir.join("program");
    std::fs::write(&path, program).unwrap();
    shem::sys::fs::set_mode(&path, 0o755).unwrap();
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
fn random_compressed_programs_end_with_every_register_stored_on_shem() {
    assert!(compressed_pool().len() > 1000);
    for seed in 0..20 {
        assert_eq!(
            on_shem(&program_c(seed, 150)).len(),
            4 * REGS.len(),
            "seed {seed}"
        );
    }
    assert_ne!(on_shem(&program_c(7, 150)), on_shem(&program_c(8, 150)));
}

#[test]
#[ignore = "needs qemu-riscv32 (the qemu-user package), which CI installs"]
fn the_reference_vcpu_agrees_with_qemu_on_random_compressed_programs() {
    let qemu = qemu().expect("qemu-riscv32 (or SHEM_QEMU) is needed");
    let dir = tempfile::tempdir().unwrap();
    let count: u64 = std::env::var("SHEM_RANDOM_PROGRAMS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(300);
    for seed in 0..count {
        let program = program_c(seed, 200);
        let (ours, theirs) = (on_shem(&program), on_qemu(&qemu, &program, dir.path()));
        assert_eq!(ours, theirs, "seed {seed}: the registers differ");
    }
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

// ---- Spike ------------------------------------------------------------------
//
// Spike is the RISC-V ISA's own simulator. It runs bare-metal programs, which
// stop by writing to `tohost` and say what they computed in a signature, a
// range of memory that it writes to a file. A program here is the same random
// instructions as above, between a prologue that fills the registers and an
// epilogue that stores them with the data, and ends by writing `tohost`; the
// reference vCPU runs it too, at Spike's address, and the two ranges must be
// the same.

const SPIKE_BASE: u32 = 0x8000_0000;
/// The signature: the data the program may have written, then every register.
const SIGNATURE_BYTES: usize = DATA_BYTES + 4 * REGS.len();

struct Bare {
    /// What is in memory at `SPIKE_BASE`: the code, `tohost`, the data and the room for the registers
    image: Vec<u8>,
    /// The program as an ELF file with the symbols Spike looks for
    elf: Vec<u8>,
    /// Where the signature is
    signature: u32,
}

/// The instructions of a bare program, given where `tohost` and the data are.
fn bare_code(seed: u64, ops: usize, compressed: bool, tohost: u32, data: u32) -> (Vec<u16>, Rng) {
    let pool = compressed_pool();
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut halves: Vec<u16> = Vec::new();
    let full = |code: &mut Vec<u16>, word: u32| code.extend([word as u16, (word >> 16) as u16]);
    for word in li(GP, data) {
        full(&mut halves, word);
    }
    for &reg in &REGS {
        for word in li(reg, rng.value()) {
            full(&mut halves, word);
        }
    }
    for _ in 0..ops {
        if compressed && rng.below(2) == 0 {
            halves.push(rng.pick(&pool));
            continue;
        }
        let word = instruction(&mut rng);
        full(&mut halves, word);
        if word & 0x7f == 0x63 {
            let (rd, rs1) = (rng.pick(&REGS), rng.pick(&REGS));
            full(&mut halves, r(0x00, 0, rd, rs1, rs1));
        }
    }
    for (slot, &reg) in REGS.iter().enumerate() {
        full(&mut halves, sw(reg, GP, (DATA_BYTES + 4 * slot) as i32));
    }
    for word in li(T0, tohost) {
        full(&mut halves, word);
    }
    // tohost = 1: exit with status 0
    for word in [addi(T1, ZERO, 1), sw(T1, T0, 0), EBREAK] {
        full(&mut halves, word);
    }
    (halves, rng)
}

fn bare_program(seed: u64, ops: usize, compressed: bool) -> Bare {
    // The addresses do not change how long the code is
    let (halves, _) = bare_code(seed, ops, compressed, 0, 0);
    let code_len = 2 * halves.len() as u32;
    let tohost = SPIKE_BASE + code_len.next_multiple_of(64);
    let data = tohost + 64;
    let (halves, mut rng) = bare_code(seed, ops, compressed, tohost, data);
    let mut image: Vec<u8> = halves.iter().flat_map(|half| half.to_le_bytes()).collect();
    image.resize((data - SPIKE_BASE) as usize, 0);
    image.extend((0..DATA_BYTES).map(|_| rng.next() as u8));
    image.resize(image.len() + 4 * REGS.len(), 0);
    let symbols = [
        ("tohost", tohost),
        ("fromhost", tohost + 8),
        ("begin_signature", data),
        ("end_signature", data + SIGNATURE_BYTES as u32),
    ];
    Bare {
        elf: bare_elf(&image, &symbols, compressed),
        image,
        signature: data,
    }
}

/// An ELF file that loads `image` at `SPIKE_BASE` and starts there, with a
/// symbol table of `symbols`, which is where Spike finds `tohost`.
fn bare_elf(image: &[u8], symbols: &[(&str, u32)], compressed: bool) -> Vec<u8> {
    const TEXT_AT: u32 = 0x100;
    let mut strtab = vec![0u8];
    let mut symtab = vec![0u8; 16]; // the null symbol
    for (name, value) in symbols {
        symtab.extend((strtab.len() as u32).to_le_bytes());
        strtab.extend(name.bytes().chain([0]));
        symtab.extend(value.to_le_bytes());
        symtab.extend(0u32.to_le_bytes()); // size
        symtab.extend([0x10, 0]); // global, no type
        symtab.extend(1u16.to_le_bytes()); // in .text
    }
    let shstrtab = b"\0.text\0.symtab\0.strtab\0.shstrtab\0".to_vec();
    let (text, symtab_at) = (TEXT_AT, TEXT_AT + image.len() as u32);
    let strtab_at = symtab_at + symtab.len() as u32;
    let shstrtab_at = strtab_at + strtab.len() as u32;
    let sections_at = (shstrtab_at + shstrtab.len() as u32).next_multiple_of(4);

    let mut out = b"\x7fELF\x01\x01\x01".to_vec();
    out.resize(16, 0);
    for half in [2u16, 243] {
        out.extend(half.to_le_bytes());
    }
    let flags = u32::from(compressed);
    for word in [1, SPIKE_BASE, 52, sections_at, flags] {
        out.extend(word.to_le_bytes()); // version, entry, phoff, shoff, flags
    }
    for half in [52u16, 32, 1, 40, 5, 4] {
        out.extend(half.to_le_bytes()); // ehsize, phentsize, phnum, shentsize, shnum, shstrndx
    }
    let size = image.len() as u32;
    for word in [1, TEXT_AT, SPIKE_BASE, SPIKE_BASE, size, size, 7, 0x1000] {
        out.extend(word.to_le_bytes());
    }
    out.resize(TEXT_AT as usize, 0);
    out.extend(image);
    out.extend(&symtab);
    out.extend(&strtab);
    out.extend(&shstrtab);
    out.resize(sections_at as usize, 0);
    // name, type, flags, addr, offset, size, link, info, addralign, entsize
    let section = |out: &mut Vec<u8>, fields: [u32; 10]| {
        for field in fields {
            out.extend(field.to_le_bytes());
        }
    };
    section(&mut out, [0; 10]);
    section(&mut out, [1, 1, 6, SPIKE_BASE, text, size, 0, 0, 4, 0]);
    section(
        &mut out,
        [7, 2, 0, 0, symtab_at, symtab.len() as u32, 3, 1, 4, 16],
    );
    section(
        &mut out,
        [15, 3, 0, 0, strtab_at, strtab.len() as u32, 0, 0, 1, 0],
    );
    section(
        &mut out,
        [23, 3, 0, 0, shstrtab_at, shstrtab.len() as u32, 0, 0, 1, 0],
    );
    out
}

/// The signature the reference vCPU leaves, as words.
fn signature_on_reference(bare: &Bare, compressed: bool) -> Vec<u32> {
    use shem::vcpu::{ExitReason, Vcpu};
    let mut vcpu = shem::reference::ReferenceVcpu::with_base(1 << 20, SPIKE_BASE);
    assert!(vcpu.set_compressed(compressed));
    vcpu.write_mem(SPIKE_BASE, &bare.image).unwrap();
    vcpu.set_pc(SPIKE_BASE);
    let exit = vcpu.run(1_000_000);
    assert_eq!(exit.reason, ExitReason::Ebreak, "{exit:?}");
    let mut bytes = vec![0u8; SIGNATURE_BYTES];
    vcpu.read_mem(bare.signature, &mut bytes).unwrap();
    bytes
        .chunks(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn spike() -> Option<PathBuf> {
    let name = std::env::var("SHEM_SPIKE").unwrap_or_else(|_| "spike".into());
    // Spike has no --version: its usage text, which it prints and exits with 1, says it is there
    let found = Command::new(&name).arg("--help").output().ok()?;
    String::from_utf8_lossy(&found.stderr)
        .contains("Spike RISC-V ISA Simulator")
        .then(|| PathBuf::from(name))
}

/// The signature Spike writes for the program.
fn signature_on_spike(
    spike: &std::path::Path,
    bare: &Bare,
    compressed: bool,
    dir: &std::path::Path,
) -> Vec<u32> {
    let (elf, signature) = (dir.join("program.elf"), dir.join("signature.txt"));
    std::fs::write(&elf, &bare.elf).unwrap();
    let isa = if compressed { "rv32imc" } else { "rv32im" };
    let out = Command::new(spike)
        .arg(format!("--isa={isa}"))
        .arg(format!("+signature={}", signature.display()))
        .arg("+signature-granularity=4")
        .arg(&elf)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "spike: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read_to_string(&signature)
        .unwrap()
        .lines()
        .map(|line| u32::from_str_radix(line.trim(), 16).unwrap())
        .collect()
}

#[test]
fn bare_programs_run_to_their_end_and_say_what_they_computed_on_shem() {
    for compressed in [false, true] {
        for seed in 0..20 {
            let bare = bare_program(seed, 150, compressed);
            assert_eq!(bare.elf[36] & 1, u8::from(compressed));
            let signature = signature_on_reference(&bare, compressed);
            assert_eq!(signature.len(), SIGNATURE_BYTES / 4);
        }
        let (one, other) = (
            bare_program(1, 150, compressed),
            bare_program(2, 150, compressed),
        );
        assert_ne!(
            signature_on_reference(&one, compressed),
            signature_on_reference(&other, compressed)
        );
    }
}

fn agree_with_spike(compressed: bool) {
    let spike = spike().expect("spike (or SHEM_SPIKE) is needed");
    let dir = tempfile::tempdir().unwrap();
    let count: u64 = std::env::var("SHEM_RANDOM_PROGRAMS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(300);
    for seed in 0..count {
        let bare = bare_program(seed, 200, compressed);
        let (ours, theirs) = (
            signature_on_reference(&bare, compressed),
            signature_on_spike(&spike, &bare, compressed, dir.path()),
        );
        assert_eq!(
            theirs.len(),
            ours.len(),
            "seed {seed}: the signature's size"
        );
        if let Some(at) = (0..ours.len()).find(|&at| ours[at] != theirs[at]) {
            let what = if at < DATA_BYTES / 4 {
                format!("data word {at}")
            } else {
                format!("x{}", REGS[at - DATA_BYTES / 4])
            };
            panic!(
                "seed {seed} (compressed: {compressed}): {what} is {:#010x} on shem and {:#010x} on Spike",
                ours[at], theirs[at]
            );
        }
    }
}

#[test]
#[ignore = "needs spike (riscv-isa-sim), which CI builds"]
fn the_reference_vcpu_agrees_with_spike_on_random_programs() {
    agree_with_spike(false);
}

#[test]
#[ignore = "needs spike (riscv-isa-sim), which CI builds"]
fn the_reference_vcpu_agrees_with_spike_on_random_compressed_programs() {
    agree_with_spike(true);
}
