//! The interface every vCPU implements, whether it is an interpreter or a model.
//!
//! This is shem's equivalent of the KVM vCPU API: the host calls [`Vcpu::run`],
//! the guest executes until something needs the host, and the host services
//! the [`Exit`] and resumes it.

use thiserror::Error;

/// Stack pointer register (x2).
pub const SP: usize = 2;
/// First argument / return value register (x10).
pub const A0: usize = 10;
/// System call number register (x17).
pub const A7: usize = 17;

/// Why [`Vcpu::run`] returned.
///
/// For every reason except `Budget`, `pc` points at the instruction that
/// caused the exit and that instruction has not retired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    /// The step budget ran out. This is a preemption point, not an error.
    Budget,
    /// The guest made a system call.
    Ecall,
    /// The guest hit a breakpoint.
    Ebreak,
    /// The guest tried to execute an instruction this vCPU does not implement.
    IllegalInstruction(u32),
    /// The guest touched an address outside its memory, or fetched from a misaligned pc.
    MemoryFault(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exit {
    pub reason: ExitReason,
    pub pc: u32,
    /// Instructions retired during this call to [`Vcpu::run`].
    pub steps: u64,
}

/// The host tried to read or write guest memory outside the guest's address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("guest memory access out of range: {len} bytes at {addr:#010x}")]
pub struct MemoryFault {
    pub addr: u32,
    pub len: usize,
}

pub trait Vcpu {
    fn pc(&self) -> u32;
    fn set_pc(&mut self, pc: u32);
    fn reg(&self, index: usize) -> u32;
    /// Writes to x0 are ignored.
    fn set_reg(&mut self, index: usize, value: u32);
    fn mem_size(&self) -> u32;
    fn read_mem(&self, addr: u32, buf: &mut [u8]) -> Result<(), MemoryFault>;
    fn write_mem(&mut self, addr: u32, data: &[u8]) -> Result<(), MemoryFault>;
    /// Zero the registers and memory, ready for a new program (`execve`), which
    /// runs without compressed instructions until it says otherwise.
    fn reset(&mut self);
    /// Run compressed (16-bit) instructions, as a program built with the C
    /// extension has, or not. False if this vCPU cannot run them: only the
    /// reference vCPU does, and the neural vCPU refuses such a program.
    fn set_compressed(&mut self, on: bool) -> bool {
        !on
    }
    /// Execute at most `budget` instructions.
    fn run(&mut self, budget: u64) -> Exit;
    /// Capture the machine in the canonical, vCPU-independent format.
    fn snapshot(&self) -> MachineState;
    /// Load a canonical machine state. A neural vCPU must be able to rebuild its
    /// whole internal state from this alone, so a snapshot taken on one vCPU can
    /// resume on another.
    fn restore(&mut self, state: &MachineState) -> Result<(), SnapshotError>;
}

/// Everything that defines a machine at an instruction boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineState {
    pub pc: u32,
    pub regs: [u32; 32],
    pub instret: u64,
    pub mem: Vec<u8>,
    /// The machine runs compressed instructions
    pub compressed: bool,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SnapshotError {
    #[error("not a shem snapshot")]
    BadMagic,
    #[error("unsupported snapshot version {0}")]
    UnsupportedVersion(u32),
    #[error("snapshot is truncated")]
    Truncated,
    #[error("snapshot has {found} bytes of memory, this vCPU has {expected}")]
    MemorySize { expected: usize, found: usize },
    #[error("this vCPU does not run compressed instructions, which the snapshot's program uses")]
    Compressed,
}

const SNAPSHOT_MAGIC: &[u8; 4] = b"SHEM";
/// Version 1 is what a machine without compressed instructions is written as;
/// version 2 adds a word of flags (bit 0: compressed) after the header.
const SNAPSHOT_VERSION: u32 = 1;
const SNAPSHOT_VERSION_FLAGS: u32 = 2;
// magic, version, pc, regs, instret, memory length
const SNAPSHOT_HEADER: usize = 4 + 4 + 4 + 32 * 4 + 8 + 4;

impl MachineState {
    /// Serialize as little-endian: magic, version, pc, x0..x31, instret, memory length, memory.
    /// With compressed instructions, a word of flags comes before the memory.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SNAPSHOT_HEADER + 4 + self.mem.len());
        out.extend_from_slice(SNAPSHOT_MAGIC);
        let version = if self.compressed {
            SNAPSHOT_VERSION_FLAGS
        } else {
            SNAPSHOT_VERSION
        };
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&self.pc.to_le_bytes());
        for reg in self.regs {
            out.extend_from_slice(&reg.to_le_bytes());
        }
        out.extend_from_slice(&self.instret.to_le_bytes());
        out.extend_from_slice(&(self.mem.len() as u32).to_le_bytes());
        if self.compressed {
            out.extend_from_slice(&1u32.to_le_bytes());
        }
        out.extend_from_slice(&self.mem);
        out
    }

    pub fn decode(data: &[u8]) -> Result<Self, SnapshotError> {
        let header = data
            .get(..SNAPSHOT_HEADER)
            .ok_or(SnapshotError::Truncated)?;
        if &header[..4] != SNAPSHOT_MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        let word =
            |offset: usize| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
        let version = word(4);
        if version != SNAPSHOT_VERSION && version != SNAPSHOT_VERSION_FLAGS {
            return Err(SnapshotError::UnsupportedVersion(version));
        }
        let (flags, body) = if version == SNAPSHOT_VERSION_FLAGS {
            let flags = data
                .get(SNAPSHOT_HEADER..SNAPSHOT_HEADER + 4)
                .ok_or(SnapshotError::Truncated)?;
            (
                u32::from_le_bytes(flags.try_into().unwrap()),
                SNAPSHOT_HEADER + 4,
            )
        } else {
            (0, SNAPSHOT_HEADER)
        };
        let regs = std::array::from_fn(|i| word(12 + 4 * i));
        let instret = u64::from_le_bytes(header[140..148].try_into().unwrap());
        let mem_len = word(148) as usize;
        let mem = data.get(body..).ok_or(SnapshotError::Truncated)?;
        if mem.len() != mem_len {
            return Err(SnapshotError::Truncated);
        }
        Ok(Self {
            pc: word(8),
            regs,
            instret,
            mem: mem.to_vec(),
            compressed: flags & 1 != 0,
        })
    }
}
