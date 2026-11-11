//! VM mode: a whole guest operating system on a vCPU, with the host as the
//! machine around it.
//!
//! The vCPU still implements only RV32IM, as in container mode. Everything
//! else is the host's, done when the vCPU exits (design rule 1):
//!
//! - CSR instructions, `mret` and `wfi` exit as illegal instructions, and the
//!   host carries them out against machine-mode CSRs it keeps;
//! - `ecall`, `ebreak`, illegal instructions and faults become traps, which the
//!   host delivers to the guest's trap vector, as hardware would;
//! - the timer is virtual, and the host raises its interrupt between slices;
//! - loads and stores outside RAM are device accesses: the host decodes the
//!   instruction and emulates the device, as KVM leaves MMIO to userspace;
//! - atomics are emulated, as in container mode.
//!
//! The machine is the one mini-rv32ima defines, so its no-MMU Linux images
//! boot unchanged: RAM at 0x80000000, a 16550 UART at 0x10000000, a CLINT timer
//! at 0x11000000, and a syscon at 0x11100000 for poweroff and reboot.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use thiserror::Error;

use crate::emulate::{Emulation, Emulator};
use crate::fdt::Fdt;
use crate::vcpu::{ExitReason, MemoryFault, Vcpu};

pub const RAM_BASE: u32 = 0x8000_0000;
const UART: u32 = 0x1000_0000;
const CLINT: u32 = 0x1100_0000;
const SYSCON: u32 = 0x1110_0000;
const POWEROFF: u32 = 0x5555;
const REBOOT: u32 = 0x7777;

/// The timer ticks at 1 MHz (the device tree's timebase), one tick per 100
/// instructions, as if the CPU ran at 100 MHz. Time is virtual, so runs are
/// repeatable, and `wfi` skips straight to the next timer interrupt.
const TIMEBASE: u32 = 1_000_000;
const INSTRUCTIONS_PER_TICK: u64 = 100;
const SLICE: u64 = 1 << 16;
pub const DEFAULT_BOOTARGS: &str = "earlycon=uart8250,mmio,0x10000000,1000000 console=ttyS0";

const MACHINE: u32 = 3;
const USER: u32 = 0;
const MSTATUS_MIE: u32 = 1 << 3;
const MSTATUS_MPIE: u32 = 1 << 7;
const MSTATUS_MPP: u32 = 3 << 11;
const MTIP: u32 = 1 << 7;
const MRET: u32 = 0x3020_0073;
const WFI: u32 = 0x1050_0073;

// Exception causes
const INSTRUCTION_FAULT: u32 = 1;
const ILLEGAL: u32 = 2;
const BREAKPOINT: u32 = 3;
const LOAD_FAULT: u32 = 5;
const STORE_FAULT: u32 = 7;
const ECALL_FROM_U: u32 = 8;
const ECALL_FROM_M: u32 = 11;
const TIMER_INTERRUPT: u32 = 0x8000_0007;

/// How the guest stopped the machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Halt {
    Poweroff,
    Reboot,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum VmError {
    #[error("the machine did not stop within {0} instructions")]
    StepLimit(u64),
    #[error("the kernel and device tree do not fit in {0} bytes of memory")]
    TooBig(u32),
}

/// The UART's other end: bytes the guest can read, and where its output goes.
pub struct Console {
    pub input: Receiver<u8>,
    pub output: Box<dyn Write + Send>,
    pending: VecDeque<u8>,
}

impl Console {
    pub fn new(input: Receiver<u8>, output: Box<dyn Write + Send>) -> Self {
        Self {
            input,
            output,
            pending: VecDeque::new(),
        }
    }

    /// Wait up to `timeout` for input, when the guest is idle. Virtual time
    /// does not depend on it: this only keeps an idle guest from spinning the
    /// host while a person types.
    fn wait(&mut self, timeout: Duration) {
        if self.pending.is_empty() {
            match self.input.recv_timeout(timeout) {
                Ok(byte) => self.pending.push_back(byte),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {}
            }
        }
    }

    fn has_input(&mut self) -> bool {
        if self.pending.is_empty()
            && let Ok(byte) = self.input.try_recv()
        {
            self.pending.push_back(byte);
        }
        !self.pending.is_empty()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Csrs {
    mstatus: u32,
    mie: u32,
    mip: u32,
    mtvec: u32,
    mscratch: u32,
    mepc: u32,
    mcause: u32,
    mtval: u32,
}

pub struct Machine<V> {
    pub vcpu: V,
    csr: Csrs,
    privilege: u32,
    mtimecmp: u64,
    /// Timer ticks skipped by `wfi`.
    idle_ticks: u64,
    emulator: Emulator,
    console: Console,
    /// Instructions retired.
    pub steps: u64,
}

/// The device tree for this machine with `ram` bytes of memory below the
/// tree itself.
pub fn device_tree(ram: u32, bootargs: &str) -> Vec<u8> {
    let mut fdt = Fdt::new();
    fdt.begin("")
        .cells("#address-cells", &[2])
        .cells("#size-cells", &[2])
        .strings("compatible", &["riscv-minimal-nommu"])
        .strings("model", &["shem"]);
    fdt.begin("chosen").strings("bootargs", &[bootargs]).end();
    fdt.begin("memory@80000000")
        .strings("device_type", &["memory"])
        .cells("reg", &[0, RAM_BASE, 0, ram])
        .end();
    fdt.begin("cpus")
        .cells("#address-cells", &[1])
        .cells("#size-cells", &[0])
        .cells("timebase-frequency", &[TIMEBASE]);
    fdt.begin("cpu@0")
        .cells("phandle", &[1])
        .strings("device_type", &["cpu"])
        .cells("reg", &[0])
        .strings("status", &["okay"])
        .strings("compatible", &["riscv"])
        .strings("riscv,isa", &["rv32ima"])
        .strings("mmu-type", &["riscv,none"]);
    fdt.begin("interrupt-controller")
        .cells("#interrupt-cells", &[1])
        .empty("interrupt-controller")
        .strings("compatible", &["riscv,cpu-intc"])
        .cells("phandle", &[2])
        .end();
    fdt.end().end();
    fdt.begin("soc")
        .cells("#address-cells", &[2])
        .cells("#size-cells", &[2])
        .strings("compatible", &["simple-bus"])
        .empty("ranges");
    fdt.begin("uart@10000000")
        .cells("clock-frequency", &[0x0100_0000])
        .cells("reg", &[0, UART, 0, 0x100])
        .strings("compatible", &["ns16850"])
        .end();
    for (name, value) in [("poweroff", POWEROFF), ("reboot", REBOOT)] {
        fdt.begin(name)
            .cells("value", &[value])
            .cells("offset", &[0])
            .cells("regmap", &[4])
            .strings("compatible", &[&format!("syscon-{name}")])
            .end();
    }
    fdt.begin("syscon@11100000")
        .cells("phandle", &[4])
        .cells("reg", &[0, SYSCON, 0, 0x1000])
        .strings("compatible", &["syscon"])
        .end();
    fdt.begin("clint@11000000")
        .cells("interrupts-extended", &[2, 3, 2, 7])
        .cells("reg", &[0, CLINT, 0, 0x10000])
        .strings("compatible", &["sifive,clint0", "riscv,clint0"])
        .end();
    fdt.end().end();
    fdt.finish()
}

impl<V: Vcpu> Machine<V> {
    /// A machine about to boot `kernel`, loaded at the start of RAM, with the
    /// device tree at the top of RAM, as the RISC-V boot protocol has it: the
    /// hart id in a0, the device tree's address in a1, in machine mode.
    pub fn boot(
        mut vcpu: V,
        kernel: &[u8],
        bootargs: &str,
        console: Console,
    ) -> Result<Self, VmError> {
        let mem = vcpu.mem_size();
        let too_big = VmError::TooBig(mem);
        let tree_size = device_tree(0, bootargs).len() as u32;
        let tree_at = mem.checked_sub(tree_size).ok_or(too_big.clone())? & !7;
        if (kernel.len() as u32) > tree_at {
            return Err(too_big);
        }
        let tree = device_tree(tree_at, bootargs);
        let load = |vcpu: &mut V, at: u32, data: &[u8]| vcpu.write_mem(RAM_BASE + at, data);
        load(&mut vcpu, 0, kernel).map_err(|_| too_big.clone())?;
        load(&mut vcpu, tree_at, &tree).map_err(|_| too_big)?;
        vcpu.set_pc(RAM_BASE);
        vcpu.set_reg(10, 0);
        vcpu.set_reg(11, RAM_BASE + tree_at);
        Ok(Self {
            vcpu,
            csr: Csrs::default(),
            privilege: MACHINE,
            mtimecmp: u64::MAX,
            idle_ticks: 0,
            emulator: Emulator::default(),
            console,
            steps: 0,
        })
    }

    /// Run until the guest powers off or reboots.
    pub fn run(&mut self, max_steps: Option<u64>) -> Result<Halt, VmError> {
        loop {
            if let Some(max) = max_steps
                && self.steps >= max
            {
                return Err(VmError::StepLimit(max));
            }
            if self.timer_interrupt_due() {
                let pc = self.vcpu.pc();
                self.trap(TIMER_INTERRUPT, 0, pc);
                continue;
            }
            let mut budget = SLICE;
            if self.timer_enabled() {
                let ticks = self.mtimecmp.saturating_sub(self.mtime()).max(1);
                budget = budget.min(ticks.saturating_mul(INSTRUCTIONS_PER_TICK));
            }
            if let Some(max) = max_steps {
                budget = budget.min(max - self.steps);
            }
            let exit = self.vcpu.run(budget);
            self.steps += exit.steps;
            let pc = exit.pc;
            match exit.reason {
                ExitReason::Budget => {}
                ExitReason::Ecall => {
                    let cause = if self.privilege == MACHINE {
                        ECALL_FROM_M
                    } else {
                        ECALL_FROM_U
                    };
                    self.trap(cause, 0, pc);
                }
                ExitReason::Ebreak => self.trap(BREAKPOINT, pc, pc),
                ExitReason::IllegalInstruction(ins) => {
                    match self.emulator.emulate(&mut self.vcpu, ins) {
                        Emulation::Done => {}
                        Emulation::Fault(addr) => self.trap(STORE_FAULT, addr, pc),
                        Emulation::Unsupported => {
                            if !self.system(ins, pc) {
                                self.trap(ILLEGAL, ins, pc);
                            }
                        }
                    }
                }
                ExitReason::MemoryFault(addr) if addr == pc => {
                    self.trap(INSTRUCTION_FAULT, addr, pc)
                }
                ExitReason::MemoryFault(addr) => match self.mmio(pc, addr) {
                    Mmio::Done => self.vcpu.set_pc(pc.wrapping_add(4)),
                    Mmio::Halt(halt) => return Ok(halt),
                    Mmio::Load => self.trap(LOAD_FAULT, addr, pc),
                    Mmio::Store => self.trap(STORE_FAULT, addr, pc),
                },
            }
        }
    }

    fn mtime(&self) -> u64 {
        self.steps / INSTRUCTIONS_PER_TICK + self.idle_ticks
    }

    fn timer_enabled(&self) -> bool {
        let globally = self.privilege < MACHINE || self.csr.mstatus & MSTATUS_MIE != 0;
        globally && self.csr.mie & MTIP != 0
    }

    fn timer_interrupt_due(&mut self) -> bool {
        if self.mtime() >= self.mtimecmp {
            self.csr.mip |= MTIP;
        } else {
            self.csr.mip &= !MTIP;
        }
        self.csr.mip & MTIP != 0 && self.timer_enabled()
    }

    /// Enter the guest's trap handler, as hardware does.
    fn trap(&mut self, cause: u32, tval: u32, pc: u32) {
        let csr = &mut self.csr;
        csr.mepc = pc;
        csr.mcause = cause;
        csr.mtval = tval;
        let mie = csr.mstatus & MSTATUS_MIE;
        csr.mstatus = (csr.mstatus & !(MSTATUS_MIE | MSTATUS_MPIE | MSTATUS_MPP))
            | (mie << 4)
            | (self.privilege << 11);
        self.privilege = MACHINE;
        let vectored = csr.mtvec & 3 == 1 && cause & 0x8000_0000 != 0;
        let base = csr.mtvec & !3;
        let target = if vectored {
            base + 4 * (cause & 0x7fff_ffff)
        } else {
            base
        };
        self.vcpu.set_pc(target);
    }

    /// Carry out a privileged instruction. Returns false if it is not one the
    /// machine has, which makes it illegal.
    fn system(&mut self, ins: u32, pc: u32) -> bool {
        if ins & 0x7f != 0x73 {
            return false;
        }
        let next = pc.wrapping_add(4);
        match ins {
            MRET => {
                let csr = &mut self.csr;
                self.privilege = (csr.mstatus & MSTATUS_MPP) >> 11;
                let mpie = csr.mstatus & MSTATUS_MPIE;
                csr.mstatus = (csr.mstatus & !(MSTATUS_MIE | MSTATUS_MPP))
                    | (mpie >> 4)
                    | MSTATUS_MPIE
                    | (USER << 11);
                self.vcpu.set_pc(csr.mepc);
                return true;
            }
            WFI => {
                // Nothing runs until an interrupt, so skip to the timer's.
                if self.csr.mie & MTIP != 0
                    && self.mtimecmp > self.mtime()
                    && self.mtimecmp != u64::MAX
                {
                    let skipped = self.mtimecmp - self.mtime();
                    self.idle_ticks += skipped;
                    self.console
                        .wait(Duration::from_micros(skipped.min(10_000)));
                }
                self.vcpu.set_pc(next);
                return true;
            }
            _ if ins >> 25 == 0x09 && (ins >> 7) & 0x1f == 0 && (ins >> 12) & 7 == 0 => {
                self.vcpu.set_pc(next); // sfence.vma: no MMU, nothing to flush
                return true;
            }
            _ => {}
        }
        let funct3 = (ins >> 12) & 7;
        if funct3 == 0 || funct3 == 4 {
            return false;
        }
        let csr = ins >> 20;
        let rd = ((ins >> 7) & 31) as usize;
        let rs1 = (ins >> 15) & 31;
        let source = if funct3 >= 5 {
            rs1
        } else {
            self.vcpu.reg(rs1 as usize)
        };
        let old = self.read_csr(csr);
        let new = match funct3 & 3 {
            1 => Some(source),
            2 if rs1 != 0 => Some(old | source),
            3 if rs1 != 0 => Some(old & !source),
            _ => None,
        };
        if let Some(new) = new {
            self.write_csr(csr, new);
        }
        self.vcpu.set_reg(rd, old);
        self.vcpu.set_pc(next);
        true
    }

    fn read_csr(&mut self, csr: u32) -> u32 {
        let mtime = self.mtime();
        match csr {
            0x300 => self.csr.mstatus,
            0x301 => 0x4040_1101, // misa: RV32, I, M, A, and non-standard extensions
            0x304 => self.csr.mie,
            0x305 => self.csr.mtvec,
            0x340 => self.csr.mscratch,
            0x341 => self.csr.mepc,
            0x342 => self.csr.mcause,
            0x343 => self.csr.mtval,
            0x344 => {
                self.timer_interrupt_due();
                self.csr.mip
            }
            0xf11 => 0xff0f_f0ff, // mvendorid, as mini-rv32ima reports it
            0xc00 | 0xb00 | 0xc02 | 0xb02 => self.steps as u32,
            0xc80 | 0xb80 | 0xc82 | 0xb82 => (self.steps >> 32) as u32,
            0xc01 => mtime as u32,
            0xc81 => (mtime >> 32) as u32,
            _ => 0, // hart ids, PMP and anything else read as 0
        }
    }

    fn write_csr(&mut self, csr: u32, value: u32) {
        match csr {
            0x300 => self.csr.mstatus = value & (MSTATUS_MIE | MSTATUS_MPIE | MSTATUS_MPP),
            0x304 => self.csr.mie = value,
            0x305 => self.csr.mtvec = value,
            0x340 => self.csr.mscratch = value,
            0x341 => self.csr.mepc = value,
            0x342 => self.csr.mcause = value,
            0x343 => self.csr.mtval = value,
            0x344 => self.csr.mip = (self.csr.mip & MTIP) | (value & !MTIP),
            _ => {} // writes to anything else are ignored
        }
    }

    /// Emulate the device access by the load or store at `pc`.
    fn mmio(&mut self, pc: u32, addr: u32) -> Mmio {
        let mut word = [0u8; 4];
        if self.vcpu.read_mem(pc, &mut word).is_err() {
            return Mmio::Load;
        }
        let ins = u32::from_le_bytes(word);
        let f3 = (ins >> 12) & 7;
        match ins & 0x7f {
            0x03 => {
                let raw = self.device_read(addr);
                let Some(raw) = raw else { return Mmio::Load };
                let value = match f3 {
                    0 => raw as u8 as i8 as u32,
                    1 => raw as u16 as i16 as u32,
                    4 => raw & 0xff,
                    5 => raw & 0xffff,
                    _ => raw,
                };
                self.vcpu.set_reg(((ins >> 7) & 31) as usize, value);
                Mmio::Done
            }
            0x23 => {
                let value = self.vcpu.reg(((ins >> 20) & 31) as usize);
                let value = match f3 {
                    0 => value & 0xff,
                    1 => value & 0xffff,
                    _ => value,
                };
                self.device_write(addr, value)
            }
            _ => Mmio::Load,
        }
    }

    fn device_read(&mut self, addr: u32) -> Option<u32> {
        let mtime = self.mtime();
        match addr {
            UART => Some(self.console.pending.pop_front().map_or(0, u32::from)),
            a if a == UART + 5 => Some(0x60 | self.console.has_input() as u32),
            a if (UART..UART + 0x100).contains(&a) => Some(0),
            a if a == CLINT + 0xbff8 => Some(mtime as u32),
            a if a == CLINT + 0xbffc => Some((mtime >> 32) as u32),
            a if a == CLINT + 0x4000 => Some(self.mtimecmp as u32),
            a if a == CLINT + 0x4004 => Some((self.mtimecmp >> 32) as u32),
            a if (CLINT..CLINT + 0x10000).contains(&a) => Some(0),
            a if (SYSCON..SYSCON + 0x1000).contains(&a) => Some(0),
            _ => None,
        }
    }

    fn device_write(&mut self, addr: u32, value: u32) -> Mmio {
        match addr {
            UART => {
                let _ = self.console.output.write_all(&[value as u8]);
                let _ = self.console.output.flush();
            }
            a if (UART..UART + 0x100).contains(&a) => {}
            a if a == CLINT + 0x4000 => {
                self.mtimecmp = (self.mtimecmp & !0xffff_ffff) | value as u64
            }
            a if a == CLINT + 0x4004 => {
                self.mtimecmp = (self.mtimecmp & 0xffff_ffff) | (value as u64) << 32;
            }
            a if (CLINT..CLINT + 0x10000).contains(&a) => {}
            SYSCON if value == POWEROFF => return Mmio::Halt(Halt::Poweroff),
            SYSCON if value == REBOOT => return Mmio::Halt(Halt::Reboot),
            a if (SYSCON..SYSCON + 0x1000).contains(&a) => {}
            _ => return Mmio::Store,
        }
        Mmio::Done
    }

    /// Read guest memory at a physical address, for tests and tools.
    pub fn read_phys(&self, addr: u32, buf: &mut [u8]) -> Result<(), MemoryFault> {
        self.vcpu.read_mem(addr, buf)
    }
}

enum Mmio {
    Done,
    Halt(Halt),
    /// Not a device: a load or store access fault.
    Load,
    Store,
}
