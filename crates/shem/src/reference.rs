//! Reference RV32IM vCPU: a plain interpreter that defines correct behaviour.
//!
//! A neural vCPU must produce the same registers, memory and exits as this one,
//! step for step. It is also fast enough to run build-time `WARM` steps, whose
//! snapshots then resume on a neural vCPU.

use crate::vcpu::{Exit, ExitReason, MachineState, MemoryFault, SnapshotError, Vcpu};

pub struct ReferenceVcpu {
    regs: [u32; 32],
    pc: u32,
    mem: Vec<u8>,
    instret: u64,
}

impl ReferenceVcpu {
    pub fn new(mem_size: u32) -> Self {
        Self {
            regs: [0; 32],
            pc: 0,
            mem: vec![0; mem_size as usize],
            instret: 0,
        }
    }

    /// Total instructions retired since the machine started.
    pub fn instret(&self) -> u64 {
        self.instret
    }

    /// Execute one instruction. On `Err`, no architectural state has changed.
    fn step(&mut self) -> Result<(), ExitReason> {
        let pc = self.pc;
        if pc & 3 != 0 {
            return Err(ExitReason::MemoryFault(pc));
        }
        let ins = self.load(pc, 4)?;
        let rd = ((ins >> 7) & 31) as usize;
        let f3 = (ins >> 12) & 7;
        let f7 = ins >> 25;
        let a = self.regs[((ins >> 15) & 31) as usize];
        let b = self.regs[((ins >> 20) & 31) as usize];
        let illegal = ExitReason::IllegalInstruction(ins);
        let mut next = pc.wrapping_add(4);

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
                Some(pc.wrapping_add(4))
            }
            0x67 if f3 == 0 => {
                // JALR
                next = a.wrapping_add(i_imm(ins)) & !1;
                Some(pc.wrapping_add(4))
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
        let start = addr as usize;
        let bytes = self
            .mem
            .get(start..start + size)
            .ok_or(ExitReason::MemoryFault(addr))?;
        let mut word = [0; 4];
        word[..size].copy_from_slice(bytes);
        Ok(u32::from_le_bytes(word))
    }

    fn store(&mut self, addr: u32, bytes: &[u8]) -> Result<(), ExitReason> {
        let start = addr as usize;
        let slot = self
            .mem
            .get_mut(start..start + bytes.len())
            .ok_or(ExitReason::MemoryFault(addr))?;
        slot.copy_from_slice(bytes);
        Ok(())
    }

    fn range(&self, addr: u32, len: usize) -> Result<std::ops::Range<usize>, MemoryFault> {
        let start = addr as usize;
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
        self.instret += steps;
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
