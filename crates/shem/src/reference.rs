//! Reference RV32IM vCPU: a plain interpreter that defines correct behaviour.
//!
//! A neural vCPU must produce the same registers, memory and exits as this one,
//! step for step. It is also fast enough to run build-time `WARM` steps, whose
//! snapshots then resume on a neural vCPU.

use crate::vcpu::{Exit, ExitReason, MachineState, MemoryFault, SnapshotError, Vcpu};

#[derive(Clone)]
pub struct ReferenceVcpu {
    regs: [u32; 32],
    pc: u32,
    mem: Vec<u8>,
    /// The guest address where memory starts. Accesses outside memory exit
    /// with a memory fault, which in VM mode is how devices are reached.
    base: u32,
    instret: u64,
    /// Runs 16-bit instructions too (the C extension), as a program built with it needs
    compressed: bool,
}

impl ReferenceVcpu {
    pub fn new(mem_size: u32) -> Self {
        Self::with_base(mem_size, 0)
    }

    /// A vCPU whose memory occupies `base .. base + mem_size`.
    pub fn with_base(mem_size: u32, base: u32) -> Self {
        Self {
            regs: [0; 32],
            pc: 0,
            mem: vec![0; mem_size as usize],
            base,
            instret: 0,
            compressed: false,
        }
    }

    /// Total instructions retired since the machine started.
    pub fn instret(&self) -> u64 {
        self.instret
    }

    /// Execute one instruction. On `Err`, no architectural state has changed.
    fn step(&mut self) -> Result<(), ExitReason> {
        let pc = self.pc;
        if pc & if self.compressed { 1 } else { 3 } != 0 {
            return Err(ExitReason::MemoryFault(pc));
        }
        // The length of the instruction, and its 32-bit form: a compressed one is expanded
        let mut len = 4;
        let mut ins = 0;
        if self.compressed {
            let half = self.load(pc, 2)?;
            if half & 3 != 3 {
                len = 2;
                ins = expand_compressed(half as u16).ok_or(ExitReason::IllegalInstruction(half))?;
            }
        }
        if len == 4 {
            ins = self.load(pc, 4)?;
        }
        let rd = ((ins >> 7) & 31) as usize;
        let f3 = (ins >> 12) & 7;
        let f7 = ins >> 25;
        let a = self.regs[((ins >> 15) & 31) as usize];
        let b = self.regs[((ins >> 20) & 31) as usize];
        let illegal = ExitReason::IllegalInstruction(ins);
        let mut next = pc.wrapping_add(len);

        let value = match ins & 0x7f {
            // OP: register-register ALU, and the M extension
            0x33 => Some(match (f7, f3) {
                (0x01, _) => muldiv(f3, a, b),
                (0x00, _) => alu(f3, a, b, false),
                (0x20, 0 | 5) => alu(f3, a, b, true),
                _ => return Err(illegal),
            }),
            // OP-IMM
            0x13 => Some(match (f3, f7) {
                (5, 0x20) => alu(f3, a, i_imm(ins), true),
                (1 | 5, 0x00) => alu(f3, a, i_imm(ins), false),
                (1 | 5, _) => return Err(illegal),
                _ => alu(f3, a, i_imm(ins), false),
            }),
            // LOAD
            0x03 => {
                let addr = a.wrapping_add(i_imm(ins));
                Some(match f3 {
                    0 => self.load(addr, 1)? as u8 as i8 as u32,
                    1 => self.load(addr, 2)? as u16 as i16 as u32,
                    2 => self.load(addr, 4)?,
                    4 => self.load(addr, 1)?,
                    5 => self.load(addr, 2)?,
                    _ => return Err(illegal),
                })
            }
            // STORE
            0x23 => {
                let addr = a.wrapping_add(s_imm(ins));
                let size = match f3 {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    _ => return Err(illegal),
                };
                self.store(addr, &b.to_le_bytes()[..size])?;
                None
            }
            // BRANCH
            0x63 => {
                let taken = match f3 {
                    0 => a == b,
                    1 => a != b,
                    4 => (a as i32) < (b as i32),
                    5 => (a as i32) >= (b as i32),
                    6 => a < b,
                    7 => a >= b,
                    _ => return Err(illegal),
                };
                if taken {
                    next = pc.wrapping_add(b_imm(ins));
                }
                None
            }
            0x37 => Some(ins & 0xffff_f000),                  // LUI
            0x17 => Some(pc.wrapping_add(ins & 0xffff_f000)), // AUIPC
            0x6f => {
                // JAL
                next = pc.wrapping_add(j_imm(ins));
                Some(pc.wrapping_add(len))
            }
            0x67 if f3 == 0 => {
                // JALR
                next = a.wrapping_add(i_imm(ins)) & !1;
                Some(pc.wrapping_add(len))
            }
            // FENCE and FENCE.I are no-ops on a single in-order hart.
            0x0f => None,
            0x73 => match ins {
                0x0000_0073 => return Err(ExitReason::Ecall),
                0x0010_0073 => return Err(ExitReason::Ebreak),
                _ => return Err(illegal), // CSR instructions are not implemented
            },
            _ => return Err(illegal),
        };

        if let Some(value) = value
            && rd != 0
        {
            self.regs[rd] = value;
        }
        self.pc = next;
        Ok(())
    }

    fn load(&self, addr: u32, size: usize) -> Result<u32, ExitReason> {
        let start = addr.wrapping_sub(self.base) as usize;
        let bytes = self
            .mem
            .get(start..start + size)
            .ok_or(ExitReason::MemoryFault(addr))?;
        let mut word = [0; 4];
        word[..size].copy_from_slice(bytes);
        Ok(u32::from_le_bytes(word))
    }

    fn store(&mut self, addr: u32, bytes: &[u8]) -> Result<(), ExitReason> {
        let start = addr.wrapping_sub(self.base) as usize;
        let slot = self
            .mem
            .get_mut(start..start + bytes.len())
            .ok_or(ExitReason::MemoryFault(addr))?;
        slot.copy_from_slice(bytes);
        Ok(())
    }

    fn range(&self, addr: u32, len: usize) -> Result<std::ops::Range<usize>, MemoryFault> {
        let start = addr.wrapping_sub(self.base) as usize;
        match start.checked_add(len) {
            Some(end) if end <= self.mem.len() => Ok(start..end),
            _ => Err(MemoryFault { addr, len }),
        }
    }
}

impl Vcpu for ReferenceVcpu {
    fn pc(&self) -> u32 {
        self.pc
    }

    fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    fn reg(&self, index: usize) -> u32 {
        self.regs[index]
    }

    fn set_reg(&mut self, index: usize, value: u32) {
        if index != 0 {
            self.regs[index] = value;
        }
    }

    fn mem_size(&self) -> u32 {
        self.mem.len() as u32
    }

    fn read_mem(&self, addr: u32, buf: &mut [u8]) -> Result<(), MemoryFault> {
        buf.copy_from_slice(&self.mem[self.range(addr, buf.len())?]);
        Ok(())
    }

    fn write_mem(&mut self, addr: u32, data: &[u8]) -> Result<(), MemoryFault> {
        let range = self.range(addr, data.len())?;
        self.mem[range].copy_from_slice(data);
        Ok(())
    }

    fn reset(&mut self) {
        self.regs = [0; 32];
        self.pc = 0;
        self.mem.fill(0);
        self.compressed = false;
    }

    fn set_compressed(&mut self, on: bool) -> bool {
        self.compressed = on;
        true
    }

    fn run(&mut self, budget: u64) -> Exit {
        let mut steps = 0;
        let reason = loop {
            if steps == budget {
                break ExitReason::Budget;
            }
            if let Err(reason) = self.step() {
                break reason;
            }
            steps += 1;
        };
        // The architectural counter wraps, whatever a restored state set it to
        self.instret = self.instret.wrapping_add(steps);
        Exit {
            reason,
            pc: self.pc,
            steps,
        }
    }

    fn snapshot(&self) -> MachineState {
        MachineState {
            pc: self.pc,
            regs: self.regs,
            instret: self.instret,
            mem: self.mem.clone(),
            compressed: self.compressed,
        }
    }

    fn restore(&mut self, state: &MachineState) -> Result<(), SnapshotError> {
        if state.mem.len() != self.mem.len() {
            return Err(SnapshotError::MemorySize {
                expected: self.mem.len(),
                found: state.mem.len(),
            });
        }
        self.pc = state.pc;
        self.regs = state.regs;
        self.regs[0] = 0;
        self.instret = state.instret;
        self.compressed = state.compressed;
        self.mem.copy_from_slice(&state.mem);
        Ok(())
    }
}

fn alu(f3: u32, a: u32, b: u32, alt: bool) -> u32 {
    match f3 {
        0 if alt => a.wrapping_sub(b),
        0 => a.wrapping_add(b),
        1 => a << (b & 31),
        2 => ((a as i32) < (b as i32)) as u32,
        3 => (a < b) as u32,
        4 => a ^ b,
        5 if alt => ((a as i32) >> (b & 31)) as u32,
        5 => a >> (b & 31),
        6 => a | b,
        _ => a & b,
    }
}

fn muldiv(f3: u32, a: u32, b: u32) -> u32 {
    let (sa, sb) = (a as i32 as i64, b as i32 as i64);
    match f3 {
        0 => a.wrapping_mul(b),
        1 => ((sa * sb) >> 32) as u32,                 // MULH
        2 => ((sa * b as i64) >> 32) as u32,           // MULHSU
        3 => ((a as u64 * b as u64) >> 32) as u32,     // MULHU
        4 if b == 0 => u32::MAX,                       // DIV by zero
        4 => (a as i32).wrapping_div(b as i32) as u32, // MIN / -1 wraps to MIN
        5 if b == 0 => u32::MAX,                       // DIVU by zero
        5 => a / b,
        6 if b == 0 => a,                              // REM by zero
        6 => (a as i32).wrapping_rem(b as i32) as u32, // MIN % -1 wraps to 0
        _ if b == 0 => a,                              // REMU by zero
        _ => a % b,
    }
}

fn i_imm(ins: u32) -> u32 {
    ((ins as i32) >> 20) as u32
}

fn s_imm(ins: u32) -> u32 {
    (((ins as i32) >> 20) as u32 & !0x1f) | ((ins >> 7) & 0x1f)
}

fn b_imm(ins: u32) -> u32 {
    (((ins as i32) >> 19) as u32 & !0xfff)
        | ((ins << 4) & 0x800)
        | ((ins >> 20) & 0x7e0)
        | ((ins >> 7) & 0x1e)
}

fn j_imm(ins: u32) -> u32 {
    (((ins as i32) >> 11) as u32 & !0xf_ffff)
        | (ins & 0xf_f000)
        | ((ins >> 9) & 0x800)
        | ((ins >> 20) & 0x7fe)
}

/// The 32-bit instruction a compressed (16-bit) one stands for, or None if it
/// is illegal or one of the floating-point and RV64 forms RV32IM has none of.
pub fn expand_compressed(h: u16) -> Option<u32> {
    let h = u32::from(h);
    let rd_full = (h >> 7) & 31;
    let rs2_full = (h >> 2) & 31;
    let rd3 = 8 + ((h >> 2) & 7); // the registers x8..x15 of the compact forms
    let rs1_3 = 8 + ((h >> 7) & 7);
    // The sign-extended 6-bit immediate of C.ADDI, C.LI and C.ANDI
    let imm6 = (((h >> 7) & 0x20) | ((h >> 2) & 0x1f)) as i32;
    let imm6 = (imm6 << 26 >> 26) as u32;
    let i_type = |imm: u32, rs1: u32, f3: u32, rd: u32, op: u32| {
        (imm & 0xfff) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
    };
    let r_type = |f7: u32, rs2: u32, rs1: u32, f3: u32, rd: u32| {
        f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | 0x33
    };
    let s_type = |imm: u32, rs2: u32, rs1: u32| {
        ((imm >> 5) & 0x7f) << 25 | rs2 << 20 | rs1 << 15 | 2 << 12 | (imm & 0x1f) << 7 | 0x23
    };
    let b_type = |imm: u32, rs1: u32, f3: u32| {
        ((imm >> 12) & 1) << 31
            | ((imm >> 5) & 0x3f) << 25
            | rs1 << 15
            | f3 << 12
            | ((imm >> 1) & 0xf) << 8
            | ((imm >> 11) & 1) << 7
            | 0x63
    };
    let j_type = |imm: u32, rd: u32| {
        ((imm >> 20) & 1) << 31
            | ((imm >> 1) & 0x3ff) << 21
            | ((imm >> 11) & 1) << 20
            | ((imm >> 12) & 0xff) << 12
            | rd << 7
            | 0x6f
    };
    // C.J and C.JAL's offset, sign-extended from 12 bits
    let j_imm = {
        let imm = ((h >> 1) & 0x800)
            | ((h << 2) & 0x400)
            | ((h >> 1) & 0x300)
            | ((h << 1) & 0x80)
            | ((h >> 1) & 0x40)
            | ((h << 3) & 0x20)
            | ((h >> 7) & 0x10)
            | ((h >> 2) & 0xe);
        (imm << 20) as i32 >> 20
    } as u32;
    // C.BEQZ and C.BNEZ's offset, sign-extended from 9 bits
    let b_imm = {
        let imm = ((h >> 4) & 0x100)
            | ((h << 1) & 0xc0)
            | ((h << 3) & 0x20)
            | ((h >> 7) & 0x18)
            | ((h >> 2) & 0x6);
        (imm << 23) as i32 >> 23
    } as u32;
    // The offsets of the word loads and stores
    let lw_imm = ((h >> 7) & 0x38) | ((h >> 4) & 0x4) | ((h << 1) & 0x40);
    let shamt = (h >> 2) & 0x1f;
    let shamt_ok = h & 0x1000 == 0; // a shift past 31 is RV64's

    match (h & 3, h >> 13) {
        // C.ADDI4SPN: addi rd', x2, nzuimm
        (0, 0) => {
            let imm = ((h >> 7) & 0x30) | ((h >> 1) & 0x3c0) | ((h >> 4) & 0x4) | ((h >> 2) & 0x8);
            (imm != 0).then(|| i_type(imm, 2, 0, rd3, 0x13))
        }
        (0, 2) => Some(i_type(lw_imm, rs1_3, 2, rd3, 0x03)), // C.LW
        (0, 6) => Some(s_type(lw_imm, rd3, rs1_3)),          // C.SW
        (1, 0) => Some(i_type(imm6, rd_full, 0, rd_full, 0x13)), // C.ADDI, C.NOP
        (1, 1) => Some(j_type(j_imm, 1)),                    // C.JAL
        (1, 2) => Some(i_type(imm6, 0, 0, rd_full, 0x13)),   // C.LI
        (1, 3) if rd_full == 2 => {
            // C.ADDI16SP
            let imm = ((h >> 3) & 0x200)
                | ((h >> 2) & 0x10)
                | ((h << 1) & 0x40)
                | ((h << 4) & 0x180)
                | ((h << 3) & 0x20);
            let imm = ((imm << 22) as i32 >> 22) as u32;
            (imm != 0).then(|| i_type(imm, 2, 0, 2, 0x13))
        }
        (1, 3) => {
            // C.LUI
            (imm6 != 0).then_some((imm6 << 12 & 0xffff_f000) | rd_full << 7 | 0x37)
        }
        (1, 4) => {
            let rd = rs1_3;
            match (h >> 10) & 3 {
                0 => shamt_ok.then(|| i_type(shamt, rd, 5, rd, 0x13)), // C.SRLI
                1 => shamt_ok.then(|| i_type(0x400 | shamt, rd, 5, rd, 0x13)), // C.SRAI
                2 => Some(i_type(imm6, rd, 7, rd, 0x13)),              // C.ANDI
                _ if h & 0x1000 != 0 => None,
                _ => {
                    let (f7, f3) = [(0x20, 0), (0, 4), (0, 6), (0, 7)][((h >> 5) & 3) as usize];
                    Some(r_type(f7, rd3, rd, f3, rd)) // C.SUB, C.XOR, C.OR, C.AND
                }
            }
        }
        (1, 5) => Some(j_type(j_imm, 0)),        // C.J
        (1, 6) => Some(b_type(b_imm, rs1_3, 0)), // C.BEQZ
        (1, 7) => Some(b_type(b_imm, rs1_3, 1)), // C.BNEZ
        (2, 0) => shamt_ok.then(|| i_type(shamt, rd_full, 1, rd_full, 0x13)), // C.SLLI
        (2, 2) => {
            // C.LWSP
            let imm = ((h >> 2) & 0x1c) | ((h >> 7) & 0x20) | ((h << 4) & 0xc0);
            (rd_full != 0).then(|| i_type(imm, 2, 2, rd_full, 0x03))
        }
        (2, 4) => Some(match (h & 0x1000 != 0, rs2_full) {
            (false, 0) if rd_full != 0 => i_type(0, rd_full, 0, 0, 0x67), // C.JR
            (false, 0) => return None,
            (false, rs2) => r_type(0, rs2, 0, 0, rd_full), // C.MV
            (true, 0) if rd_full == 0 => 0x0010_0073,      // C.EBREAK
            (true, 0) => i_type(0, rd_full, 0, 1, 0x67),   // C.JALR
            (true, rs2) => r_type(0, rs2, rd_full, 0, rd_full), // C.ADD
        }),
        (2, 6) => {
            // C.SWSP
            let imm = ((h >> 7) & 0x3c) | ((h >> 1) & 0xc0);
            Some(s_type(imm, rs2_full, 2))
        }
        _ => None,
    }
}
