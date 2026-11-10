//! The hypervisor loop for container mode: run the vCPU, service its exits, repeat.

use thiserror::Error;

use crate::elf::{self, ElfError};
use crate::emulate::{Emulation, Emulator};
use crate::linux::{Action, Linux, Stdio};
use crate::vcpu::{A0, Exit, ExitReason, MemoryFault, SP, Vcpu};

/// Memory kept free for the stack at the top of guest memory, above the heap and mappings.
pub const STACK_SIZE: u32 = 1 << 20;
/// Instructions per `run` call. Each slice boundary is a preemption point.
pub const SLICE: u64 = 1 << 16;

#[derive(Debug, Error)]
pub enum StartError {
    #[error(transparent)]
    Elf(#[from] ElfError),
    #[error("arguments and environment do not fit on the stack: {0}")]
    Stack(#[from] MemoryFault),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RunError {
    /// The guest stopped on something the host cannot service.
    #[error("guest stopped: {:?} at pc {:#010x}", .0.reason, .0.pc)]
    Fault(Exit),
    #[error("guest did not exit within {0} instructions")]
    StepLimit(u64),
}

/// One guest process plus the host-side kernel state that belongs to it.
pub struct Container<V> {
    pub vcpu: V,
    pub linux: Linux,
    pub emulator: Emulator,
    /// Instructions retired so far by the vCPU. Emulated instructions are not counted.
    pub steps: u64,
}

impl<V: Vcpu> Container<V> {
    /// Load a static executable into `vcpu` and set it up to start at its entry point.
    pub fn from_elf(
        mut vcpu: V,
        elf: &[u8],
        argv: &[String],
        envp: &[String],
        stdio: Stdio,
    ) -> Result<Self, StartError> {
        let image = elf::load(&mut vcpu, elf)?;
        let stack_top = vcpu.mem_size();
        let sp = elf::setup_stack(&mut vcpu, &image, stack_top, argv, envp)?;
        vcpu.set_reg(SP, sp);
        vcpu.set_pc(image.entry);
        let linux = Linux::new(stdio, image.brk, stack_top.saturating_sub(STACK_SIZE));
        Ok(Self {
            vcpu,
            linux,
            emulator: Emulator::default(),
            steps: 0,
        })
    }

    /// Run until the guest exits, and return its exit status.
    pub fn run(&mut self, max_steps: Option<u64>) -> Result<u8, RunError> {
        loop {
            let budget = match max_steps {
                None => SLICE,
                Some(max) if self.steps >= max => return Err(RunError::StepLimit(max)),
                Some(max) => SLICE.min(max - self.steps),
            };
            let exit = self.vcpu.run(budget);
            self.steps += exit.steps;
            match exit.reason {
                // A scheduler running many guests would switch to another one here.
                ExitReason::Budget => {}
                ExitReason::Ecall => match self.linux.handle(&mut self.vcpu) {
                    Action::Exit(status) => return Ok(status),
                    Action::Return(value) => {
                        self.vcpu.set_reg(A0, value);
                        self.vcpu.set_pc(exit.pc.wrapping_add(4));
                    }
                },
                ExitReason::IllegalInstruction(ins) => {
                    match self.emulator.emulate(&mut self.vcpu, ins) {
                        Emulation::Done => {}
                        Emulation::Unsupported => return Err(RunError::Fault(exit)),
                        Emulation::Fault(addr) => {
                            let reason = ExitReason::MemoryFault(addr);
                            return Err(RunError::Fault(Exit { reason, ..exit }));
                        }
                    }
                }
                _ => return Err(RunError::Fault(exit)),
            }
        }
    }
}
