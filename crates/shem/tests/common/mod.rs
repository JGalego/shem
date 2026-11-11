//! A tiny RV32IM assembler and ELF writer, so tests don't need a cross-compiler.
// Each test binary compiles this module separately and uses a different part of it.
#![allow(dead_code, unused_imports, unused_macros)]

use std::io::Write;
use std::sync::{Arc, Mutex};

use shem::container::{Container, StartError};
use shem::linux::Stdio;
use shem::reference::ReferenceVcpu;

pub const ZERO: u32 = 0;
pub const RA: u32 = 1;
pub const SP: u32 = 2;
pub const T0: u32 = 5;
pub const T1: u32 = 6;
pub const T2: u32 = 7;
pub const S0: u32 = 8;
pub const A0: u32 = 10;
pub const A1: u32 = 11;
pub const A2: u32 = 12;
pub const A3: u32 = 13;
pub const A4: u32 = 14;
pub const A7: u32 = 17;

pub const ECALL: u32 = 0x0000_0073;
pub const EBREAK: u32 = 0x0010_0073;

pub fn r(f7: u32, f3: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | 0x33
}

pub fn i(op: u32, f3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
    (imm as u32 & 0xfff) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
}

fn s(f3: u32, rs2: u32, rs1: u32, imm: i32) -> u32 {
    let imm = imm as u32;
    (imm >> 5 & 0x7f) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (imm & 0x1f) << 7 | 0x23
}

fn b(f3: u32, rs1: u32, rs2: u32, offset: i32) -> u32 {
    let o = offset as u32;
    (o >> 12 & 1) << 31
        | (o >> 5 & 0x3f) << 25
        | rs2 << 20
        | rs1 << 15
        | f3 << 12
        | (o >> 1 & 0xf) << 8
        | (o >> 11 & 1) << 7
        | 0x63
}

/// Define one encoder function per instruction: `name(operands) = encoding`.
macro_rules! encoders {
    ($($name:ident($($arg:ident: $ty:ty),*) = $body:expr;)*) => {
        $(pub fn $name($($arg: $ty),*) -> u32 { $body })*
    };
}

encoders! {
    add(rd: u32, rs1: u32, rs2: u32) = r(0x00, 0, rd, rs1, rs2);
    sub(rd: u32, rs1: u32, rs2: u32) = r(0x20, 0, rd, rs1, rs2);
    sra(rd: u32, rs1: u32, rs2: u32) = r(0x20, 5, rd, rs1, rs2);
    mul(rd: u32, rs1: u32, rs2: u32) = r(0x01, 0, rd, rs1, rs2);
    mulh(rd: u32, rs1: u32, rs2: u32) = r(0x01, 1, rd, rs1, rs2);
    mulhsu(rd: u32, rs1: u32, rs2: u32) = r(0x01, 2, rd, rs1, rs2);
    mulhu(rd: u32, rs1: u32, rs2: u32) = r(0x01, 3, rd, rs1, rs2);
    div(rd: u32, rs1: u32, rs2: u32) = r(0x01, 4, rd, rs1, rs2);
    divu(rd: u32, rs1: u32, rs2: u32) = r(0x01, 5, rd, rs1, rs2);
    rem(rd: u32, rs1: u32, rs2: u32) = r(0x01, 6, rd, rs1, rs2);
    remu(rd: u32, rs1: u32, rs2: u32) = r(0x01, 7, rd, rs1, rs2);
    addi(rd: u32, rs1: u32, imm: i32) = i(0x13, 0, rd, rs1, imm);
    srai(rd: u32, rs1: u32, shamt: u32) = i(0x13, 5, rd, rs1, (0x400 | shamt) as i32);
    lb(rd: u32, rs1: u32, imm: i32) = i(0x03, 0, rd, rs1, imm);
    lw(rd: u32, rs1: u32, imm: i32) = i(0x03, 2, rd, rs1, imm);
    lbu(rd: u32, rs1: u32, imm: i32) = i(0x03, 4, rd, rs1, imm);
    jalr(rd: u32, rs1: u32, imm: i32) = i(0x67, 0, rd, rs1, imm);
    sb(rs2: u32, rs1: u32, imm: i32) = s(0, rs2, rs1, imm);
    sw(rs2: u32, rs1: u32, imm: i32) = s(2, rs2, rs1, imm);
    beq(rs1: u32, rs2: u32, offset: i32) = b(0, rs1, rs2, offset);
    bne(rs1: u32, rs2: u32, offset: i32) = b(1, rs1, rs2, offset);
    lui(rd: u32, upper: u32) = upper & 0xffff_f000 | rd << 7 | 0x37;
    auipc(rd: u32, upper: u32) = upper & 0xffff_f000 | rd << 7 | 0x17;
    lr_w(rd: u32, rs1: u32) = amo(0x02, rd, rs1, 0);
    sc_w(rd: u32, rs2: u32, rs1: u32) = amo(0x03, rd, rs1, rs2);
    amoadd_w(rd: u32, rs2: u32, rs1: u32) = amo(0x00, rd, rs1, rs2);
    amomaxu_w(rd: u32, rs2: u32, rs1: u32) = amo(0x1c, rd, rs1, rs2);
}

fn amo(funct5: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    funct5 << 27 | rs2 << 20 | rs1 << 15 | 2 << 12 | rd << 7 | 0x2f
}

pub fn jal(rd: u32, offset: i32) -> u32 {
    let o = offset as u32;
    (o >> 20 & 1) << 31
        | (o >> 1 & 0x3ff) << 21
        | (o >> 11 & 1) << 20
        | (o >> 12 & 0xff) << 12
        | rd << 7
        | 0x6f
}

/// Load any 32-bit constant: LUI for the upper bits, ADDI for the sign-extended lower 12.
pub fn li(rd: u32, value: u32) -> [u32; 2] {
    let low = ((value & 0xfff) ^ 0x800) as i32 - 0x800;
    [lui(rd, value.wrapping_sub(low as u32)), addi(rd, rd, low)]
}

pub trait Emit {
    fn emit(self, out: &mut Vec<u32>);
}

impl Emit for u32 {
    fn emit(self, out: &mut Vec<u32>) {
        out.push(self);
    }
}

impl<const N: usize> Emit for [u32; N] {
    fn emit(self, out: &mut Vec<u32>) {
        out.extend(self);
    }
}

/// Build a list of instruction words from instructions and multi-word pseudo-instructions.
macro_rules! program {
    ($($item:expr),* $(,)?) => {{
        let mut words: Vec<u32> = Vec::new();
        $($crate::common::Emit::emit($item, &mut words);)*
        words
    }};
}
pub(crate) use program;

pub fn bytes(code: &[u32]) -> Vec<u8> {
    code.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Where `elf` loads the file, and where the data section ends up.
pub const BASE: u32 = 0x1_0000;
pub const DATA: u32 = BASE + 52 + 32;

/// A static RV32 executable with one RWX segment: headers, then `data`, then `code`.
pub fn elf(code: &[u32], data: &[u8]) -> Vec<u8> {
    elf_with_flags(code, data, 0)
}

pub fn elf_with_flags(code: &[u32], data: &[u8], flags: u32) -> Vec<u8> {
    let mut body = data.to_vec();
    body.resize(data.len().next_multiple_of(4), 0);
    let entry = DATA + body.len() as u32;
    body.extend(bytes(code));
    let filesz = 52 + 32 + body.len() as u32;

    let mut out = b"\x7fELF\x01\x01\x01".to_vec();
    out.resize(16, 0);
    for half in [2u16, 243] {
        out.extend(half.to_le_bytes()); // e_type = ET_EXEC, e_machine = EM_RISCV
    }
    for word in [1, entry, 52, 0, flags] {
        out.extend(word.to_le_bytes()); // e_version, e_entry, e_phoff, e_shoff, e_flags
    }
    for half in [52u16, 32, 1, 40, 0, 0] {
        out.extend(half.to_le_bytes()); // e_ehsize, e_phentsize, e_phnum, e_shentsize, ...
    }
    for word in [1, 0, BASE, BASE, filesz, filesz, 7, 0x1000] {
        out.extend(word.to_le_bytes()); // PT_LOAD covering the whole file, RWX
    }
    out.extend(body);
    out
}

/// An in-memory stream the container can write to while the test keeps a handle.
#[derive(Clone, Default)]
pub struct SharedBuf(Arc<Mutex<Vec<u8>>>);

impl SharedBuf {
    pub fn contents(&self) -> Vec<u8> {
        self.0.lock().unwrap().clone()
    }
}

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Load `elf` into a fresh reference vCPU with the given stdin, capturing stdout.
pub fn start(
    elf: &[u8],
    argv: &[&str],
    stdin: &[u8],
    mem: u32,
) -> Result<(Container<ReferenceVcpu>, SharedBuf), StartError> {
    let stdout = SharedBuf::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::Cursor::new(stdin.to_vec())),
        stdout: Box::new(stdout.clone()),
        stderr: Box::new(std::io::sink()),
    };
    let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    let container = Container::from_elf(ReferenceVcpu::new(mem), elf, &argv, &[], stdio)?;
    Ok((container, stdout))
}
