//! Signals: dispositions, masks, pending sets, and delivery to guest handlers.
//!
//! Only the host raises signals: `kill`, SIGCHLD when a child exits, and SIGPIPE
//! on a write nobody will read. So it always knows when one becomes pending,
//! and the container delivers it at the start of the process's next turn.
//!
//! Delivery works as on Linux. The host pushes a frame onto the guest stack
//! holding `siginfo_t`, a `ucontext_t` with the interrupted registers and mask,
//! and a two-instruction trampoline that calls `rt_sigreturn`. The handler
//! starts with `ra` pointing at the trampoline. RISC-V Linux keeps that
//! trampoline in the vDSO; shem has no vDSO, so it lives in the frame, which
//! works because guest memory has no execute protection.

use serde::{Deserialize, Serialize};

use super::Termination;
use super::abi::{Errno, SIG_DFL, SIG_IGN, SIGKILL, SIGSTOP};
use crate::vcpu::{A0, MemoryFault, SP, Vcpu};

const SA_RESTART: u32 = 0x1000_0000;
const SA_NODEFER: u32 = 0x4000_0000;
const SA_RESETHAND: u32 = 0x8000_0000;
const RT_SIGRETURN: u32 = 139;
const SIGSEGV: u32 = 11;
const EINTR: u32 = 4;
const RA: usize = 1;
const A1: usize = 11;
const A2: usize = 12;
const A7: u32 = 17;

/// Signals whose default action is to do nothing, or to stop the process, which
/// shem treats the same way.
const HARMLESS: [u32; 8] = [17, 18, 19, 20, 21, 22, 23, 28];
/// SIGKILL and SIGSTOP can be neither caught nor blocked.
const UNBLOCKABLE: u64 = (1 << (SIGKILL - 1)) | (1 << (SIGSTOP - 1));

// The frame: siginfo_t, then ucontext_t (mask at 20, registers at 160, room for
// floating-point state after them), then the trampoline.
const SIGINFO_SIZE: u32 = 128;
const UC_SIGMASK: u32 = 20;
const UC_REGS: u32 = 160;
const UCONTEXT_SIZE: u32 = UC_REGS + 32 * 4 + 528;
const FRAME_SIZE: u32 = (SIGINFO_SIZE + UCONTEXT_SIZE + 8).next_multiple_of(16);

/// What `sigaction` sets for one signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigAction {
    pub handler: u32,
    pub flags: u32,
    pub mask: u64,
}

const DEFAULT: SigAction = SigAction {
    handler: SIG_DFL,
    flags: 0,
    mask: 0,
};

#[derive(Debug, Clone)]
pub struct Signals {
    actions: [SigAction; 64],
    pub blocked: u64,
    pub pending: u64,
    /// The mask a `sigsuspend` in progress replaced, to restore when a signal
    /// interrupts it.
    suspended: Option<u64>,
}

/// What looking at a process's pending signals did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    Nothing,
    /// A handler is about to run.
    Handler,
    /// The signal's default action ended the process.
    Killed(Termination),
}

fn bit(signal: u32) -> u64 {
    1 << (signal - 1)
}

impl Default for Signals {
    fn default() -> Self {
        Self {
            actions: [DEFAULT; 64],
            blocked: 0,
            pending: 0,
            suspended: None,
        }
    }
}

impl Signals {
    /// A forked child keeps dispositions and mask, but starts with nothing pending.
    pub fn fork(&self) -> Self {
        Self {
            pending: 0,
            suspended: None,
            ..self.clone()
        }
    }

    /// A new program cannot keep handlers, which point into the old one; ignored
    /// signals stay ignored.
    pub fn exec(&mut self) {
        for action in &mut self.actions {
            if action.handler != SIG_IGN {
                *action = DEFAULT;
            }
        }
    }

    /// Raise `signal`. Returns how the process ends if the signal kills it now;
    /// otherwise the signal is ignored, or left pending for delivery.
    pub fn raise(&mut self, signal: u32) -> Option<Termination> {
        if signal == 0 {
            return None;
        }
        if signal == SIGKILL {
            return Some(Termination::Killed(signal));
        }
        let handler = self.actions[signal as usize - 1].handler;
        if handler == SIG_IGN || (handler == SIG_DFL && HARMLESS.contains(&signal)) {
            return None;
        }
        if handler == SIG_DFL && self.blocked & bit(signal) == 0 {
            return Some(Termination::Killed(signal));
        }
        self.pending |= bit(signal);
        None
    }

    /// Set the action for `signal` if `new` is given, and return the old one.
    pub fn sigaction(&mut self, signal: u32, new: Option<SigAction>) -> Result<SigAction, Errno> {
        if !(1..=64).contains(&signal) || (new.is_some() && UNBLOCKABLE & bit(signal) != 0) {
            return Err(Errno::EINVAL);
        }
        let slot = &mut self.actions[signal as usize - 1];
        let old = *slot;
        if let Some(new) = new {
            *slot = new;
            if new.handler == SIG_IGN {
                self.pending &= !bit(signal);
            }
        }
        Ok(old)
    }

    pub fn set_mask(&mut self, mask: u64) {
        self.blocked = mask & !UNBLOCKABLE;
    }

    /// Start a `sigsuspend`: wait with `mask` in place of the current mask.
    pub fn suspend(&mut self, mask: u64) {
        if self.suspended.is_none() {
            self.suspended = Some(self.blocked);
        }
        self.set_mask(mask);
    }

    /// Deliver the lowest pending signal that is not blocked, if there is one.
    /// `parked` says whether the process is parked on a system call: the signal
    /// interrupts it, so it returns EINTR, or with `SA_RESTART` runs again after
    /// the handler. An interrupted `sigsuspend` always returns EINTR.
    pub fn deliver(&mut self, vcpu: &mut dyn Vcpu, parked: bool) -> Delivery {
        let ready = self.pending & !self.blocked;
        if ready == 0 {
            return Delivery::Nothing;
        }
        let signal = ready.trailing_zeros() + 1;
        self.pending &= !bit(signal);
        let action = self.actions[signal as usize - 1];
        match action.handler {
            SIG_IGN => return Delivery::Nothing,
            SIG_DFL if HARMLESS.contains(&signal) => return Delivery::Nothing,
            SIG_DFL => return Delivery::Killed(Termination::Killed(signal)),
            _ => {}
        }
        let mut saved_mask = self.blocked;
        if parked {
            if let Some(mask) = self.suspended.take() {
                saved_mask = mask;
                return_eintr(vcpu);
            } else if action.flags & SA_RESTART == 0 {
                return_eintr(vcpu);
            }
        }
        if push_frame(vcpu, signal, action.handler, saved_mask).is_err() {
            return Delivery::Killed(Termination::Killed(SIGSEGV));
        }
        self.blocked |= action.mask & !UNBLOCKABLE;
        if action.flags & SA_NODEFER == 0 {
            self.blocked |= bit(signal);
        }
        if action.flags & SA_RESETHAND != 0 {
            self.actions[signal as usize - 1] = DEFAULT;
        }
        Delivery::Handler
    }

    /// The dispositions, mask and pending set, for a warm image.
    pub fn freeze(&self) -> FrozenSignals {
        FrozenSignals {
            actions: self.actions.to_vec(),
            blocked: self.blocked,
            pending: self.pending,
        }
    }

    pub fn thaw(frozen: &FrozenSignals) -> Self {
        let mut signals = Self::default();
        for (slot, action) in signals.actions.iter_mut().zip(&frozen.actions) {
            *slot = *action;
        }
        signals.blocked = frozen.blocked;
        signals.pending = frozen.pending;
        signals
    }

    /// `rt_sigreturn`: restore the registers and mask saved in the frame at sp.
    pub fn sigreturn(&mut self, vcpu: &mut dyn Vcpu) -> Result<(), MemoryFault> {
        let uc = vcpu.reg(SP).wrapping_add(SIGINFO_SIZE);
        let mut mask = [0u8; 8];
        vcpu.read_mem(uc.wrapping_add(UC_SIGMASK), &mut mask)?;
        let mut regs = [0u8; 128];
        vcpu.read_mem(uc.wrapping_add(UC_REGS), &mut regs)?;
        let word = |i: usize| u32::from_le_bytes(regs[4 * i..4 * i + 4].try_into().unwrap());
        vcpu.set_pc(word(0));
        for reg in 1..32 {
            vcpu.set_reg(reg, word(reg));
        }
        self.set_mask(u64::from_le_bytes(mask));
        Ok(())
    }
}

fn return_eintr(vcpu: &mut dyn Vcpu) {
    vcpu.set_reg(A0, EINTR.wrapping_neg());
    vcpu.set_pc(vcpu.pc().wrapping_add(4));
}

fn push_frame(
    vcpu: &mut dyn Vcpu,
    signal: u32,
    handler: u32,
    mask: u64,
) -> Result<(), MemoryFault> {
    let sp = vcpu.reg(SP);
    let fault = MemoryFault {
        addr: sp,
        len: FRAME_SIZE as usize,
    };
    let base = sp.checked_sub(FRAME_SIZE).ok_or(fault)? & !15;
    let mut frame = vec![0u8; FRAME_SIZE as usize];
    let mut put = |offset: u32, bytes: &[u8]| {
        let offset = offset as usize;
        frame[offset..offset + bytes.len()].copy_from_slice(bytes);
    };
    put(0, &signal.to_le_bytes()); // si_signo; si_code 0 is SI_USER
    put(SIGINFO_SIZE + UC_SIGMASK, &mask.to_le_bytes());
    let regs = SIGINFO_SIZE + UC_REGS;
    put(regs, &vcpu.pc().to_le_bytes());
    for reg in 1..32 {
        put(regs + 4 * reg as u32, &vcpu.reg(reg).to_le_bytes());
    }
    let trampoline = SIGINFO_SIZE + UCONTEXT_SIZE;
    put(
        trampoline,
        &(RT_SIGRETURN << 20 | A7 << 7 | 0x13).to_le_bytes(),
    ); // li a7, 139
    put(trampoline + 4, &0x73u32.to_le_bytes()); // ecall
    vcpu.write_mem(base, &frame)?;

    vcpu.set_reg(SP, base);
    vcpu.set_reg(A0, signal);
    vcpu.set_reg(A1, base);
    vcpu.set_reg(A2, base + SIGINFO_SIZE);
    vcpu.set_reg(RA, base + trampoline);
    vcpu.set_pc(handler);
    Ok(())
}

/// Signal state as a warm image stores it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenSignals {
    pub actions: Vec<SigAction>,
    pub blocked: u64,
    pub pending: u64,
}
