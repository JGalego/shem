//! Host emulation of instructions the vCPU does not implement: trap and emulate.
//!
//! The vCPU exits with `IllegalInstruction`, the host carries the instruction out
//! through the register and memory interface, and the guest resumes after it.
//! Exits are cheap relative to neural instructions, so this keeps vCPUs small.
//!
//! The A extension comes first, because musl's riscv32 port needs LR/SC. A single
//! hart has no concurrency, so every atomic is a plain read-modify-write.

use crate::vcpu::Vcpu;

const OP_AMO: u32 = 0x2f;
const WIDTH_W: u32 = 2;

const LR: u32 = 0x02;
const SC: u32 = 0x03;
const AMOSWAP: u32 = 0x01;
const AMOADD: u32 = 0x00;
const AMOXOR: u32 = 0x04;
const AMOAND: u32 = 0x0c;
const AMOOR: u32 = 0x08;
const AMOMIN: u32 = 0x10;
const AMOMAX: u32 = 0x14;
const AMOMINU: u32 = 0x18;
const AMOMAXU: u32 = 0x1c;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emulation {
    /// The instruction was carried out and the pc moved past it.
    Done,
    /// The host does not emulate this instruction either.
    Unsupported,
    /// The instruction touched this address, which is misaligned or outside guest memory.
    Fault(u32),
}

/// Host-side state for emulated instructions.
#[derive(Debug, Default)]
pub struct Emulator {
    /// Address reserved by the last LR.W, until an SC.W consumes it.
    reservation: Option<u32>,
}

impl Emulator {
    /// Emulate `ins`, which the vCPU just refused at its current pc.
    pub fn emulate(&mut self, vcpu: &mut dyn Vcpu, ins: u32) -> Emulation {
        if ins & 0x7f != OP_AMO || (ins >> 12) & 7 != WIDTH_W {
            return Emulation::Unsupported;
        }
        let rd = ((ins >> 7) & 31) as usize;
        let addr = vcpu.reg(((ins >> 15) & 31) as usize);
        let rs2 = ((ins >> 20) & 31) as usize;
        let src = vcpu.reg(rs2);
        let funct5 = ins >> 27;
        if addr & 3 != 0 {
            return Emulation::Fault(addr);
        }
        let mut word = [0; 4];
        if vcpu.read_mem(addr, &mut word).is_err() {
            return Emulation::Fault(addr);
        }
        let old = u32::from_le_bytes(word);

        let (result, new) = match funct5 {
            LR if rs2 == 0 => {
                self.reservation = Some(addr);
                (old, None)
            }
            SC => match self.reservation.take() {
                Some(reserved) if reserved == addr => (0, Some(src)),
                _ => (1, None),
            },
            AMOSWAP => (old, Some(src)),
            AMOADD => (old, Some(old.wrapping_add(src))),
            AMOXOR => (old, Some(old ^ src)),
            AMOAND => (old, Some(old & src)),
            AMOOR => (old, Some(old | src)),
            AMOMIN => (old, Some((old as i32).min(src as i32) as u32)),
            AMOMAX => (old, Some((old as i32).max(src as i32) as u32)),
            AMOMINU => (old, Some(old.min(src))),
            AMOMAXU => (old, Some(old.max(src))),
            _ => return Emulation::Unsupported,
        };
        if let Some(new) = new
            && vcpu.write_mem(addr, &new.to_le_bytes()).is_err()
        {
            return Emulation::Fault(addr);
        }
        vcpu.set_reg(rd, result);
        vcpu.set_pc(vcpu.pc().wrapping_add(4));
        Emulation::Done
    }
}
