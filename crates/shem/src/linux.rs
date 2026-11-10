//! Linux system calls for RV32 guests, implemented by the host.
//!
//! In container mode this is the guest's kernel: the vCPU exits on every
//! `ecall`, the host does the work, and the result goes back in a0. It is also
//! the security boundary, so every guest pointer and length is treated as hostile.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};

use crate::mm::{AddressSpace, MapError};
use crate::vcpu::{A0, A7, MemoryFault, Vcpu};

// asm-generic system call numbers, which riscv32 uses
const SYS_IOCTL: u32 = 29;
const SYS_CLOSE: u32 = 57;
const SYS_READ: u32 = 63;
const SYS_WRITE: u32 = 64;
const SYS_WRITEV: u32 = 66;
const SYS_EXIT: u32 = 93;
const SYS_EXIT_GROUP: u32 = 94;
const SYS_SET_TID_ADDRESS: u32 = 96;
const SYS_SCHED_GETAFFINITY: u32 = 123;
const SYS_BRK: u32 = 214;
const SYS_MUNMAP: u32 = 215;
const SYS_MREMAP: u32 = 216;
const SYS_MMAP: u32 = 222;
const SYS_MPROTECT: u32 = 226;
const SYS_MADVISE: u32 = 233;

const MAP_FIXED: u32 = 0x10;
const MAP_ANONYMOUS: u32 = 0x20;

/// Largest single transfer. Short reads and writes are legal, and the cap stops
/// a guest from making the host allocate gigabytes.
const MAX_IO: u32 = 1 << 20;
const IOV_MAX: u32 = 1024;

/// A Linux errno. Guests see these numbers, so they must not come from the host OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Errno(pub u32);

impl Errno {
    pub const EIO: Self = Self(5);
    pub const EBADF: Self = Self(9);
    pub const EFAULT: Self = Self(14);
    pub const ENOMEM: Self = Self(12);
    pub const ENODEV: Self = Self(19);
    pub const EINVAL: Self = Self(22);
    pub const ENOTTY: Self = Self(25);
    pub const ENOSYS: Self = Self(38);
}

impl From<MemoryFault> for Errno {
    fn from(_: MemoryFault) -> Self {
        Self::EFAULT
    }
}

impl From<MapError> for Errno {
    fn from(err: MapError) -> Self {
        match err {
            MapError::Invalid => Self::EINVAL,
            MapError::NoMemory => Self::ENOMEM,
        }
    }
}

impl From<std::io::Error> for Errno {
    fn from(_: std::io::Error) -> Self {
        Self::EIO
    }
}

/// The host streams behind the guest's file descriptors 0, 1 and 2.
pub struct Stdio {
    pub stdin: Box<dyn Read + Send>,
    pub stdout: Box<dyn Write + Send>,
    pub stderr: Box<dyn Write + Send>,
}

/// What the host should do after servicing a system call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Put this value in a0 and resume the guest.
    Return(u32),
    /// The guest exited with this status.
    Exit(u8),
}

pub struct Linux {
    stdio: Stdio,
    open: BTreeSet<u32>,
    pub mm: AddressSpace,
    /// System calls the guest made that are not implemented, and how many times.
    pub unsupported: BTreeMap<u32, u64>,
}

impl Linux {
    /// `brk` is where the heap starts; the heap and mappings share memory up to `top`.
    pub fn new(stdio: Stdio, brk: u32, top: u32) -> Self {
        Self {
            stdio,
            open: BTreeSet::from([0, 1, 2]),
            mm: AddressSpace::new(brk, top),
            unsupported: BTreeMap::new(),
        }
    }

    pub fn handle(&mut self, vcpu: &mut dyn Vcpu) -> Action {
        let nr = vcpu.reg(A7);
        let [a0, a1, a2, a3] = std::array::from_fn(|i| vcpu.reg(A0 + i));
        let result = match nr {
            SYS_EXIT | SYS_EXIT_GROUP => return Action::Exit(a0 as u8),
            SYS_READ => self.read(vcpu, a0, a1, a2),
            SYS_WRITE => self.write(vcpu, a0, a1, a2),
            SYS_WRITEV => self.writev(vcpu, a0, a1, a2),
            SYS_CLOSE => self.close(a0),
            SYS_BRK => Ok(self.mm.set_brk(vcpu, a0)),
            SYS_MMAP => self.mmap(vcpu, a0, a1, a3),
            SYS_MUNMAP => self.mm.unmap(a0, a1).map(|()| 0).map_err(Errno::from),
            SYS_MREMAP => Err(Errno::ENOMEM), // callers fall back to map, copy, unmap
            SYS_MPROTECT | SYS_MADVISE => Ok(0), // guest memory has no protection bits
            SYS_IOCTL => Err(Errno::ENOTTY),
            SYS_SET_TID_ADDRESS => Ok(1), // the only thread is tid 1
            SYS_SCHED_GETAFFINITY => self.sched_getaffinity(vcpu, a1, a2),
            _ => {
                *self.unsupported.entry(nr).or_default() += 1;
                Err(Errno::ENOSYS)
            }
        };
        Action::Return(match result {
            Ok(value) => value,
            Err(Errno(errno)) => errno.wrapping_neg(),
        })
    }

    fn read(&mut self, vcpu: &mut dyn Vcpu, fd: u32, buf: u32, count: u32) -> Result<u32, Errno> {
        if fd != 0 || !self.open.contains(&fd) {
            return Err(Errno::EBADF);
        }
        let mut data = vec![0; count.min(MAX_IO) as usize];
        let n = self.stdio.stdin.read(&mut data)?;
        vcpu.write_mem(buf, &data[..n])?;
        Ok(n as u32)
    }

    fn write(&mut self, vcpu: &dyn Vcpu, fd: u32, buf: u32, count: u32) -> Result<u32, Errno> {
        let mut data = vec![0; count.min(MAX_IO) as usize];
        vcpu.read_mem(buf, &mut data)?;
        let out = self.output(fd)?;
        out.write_all(&data)?;
        out.flush()?;
        Ok(data.len() as u32)
    }

    fn writev(&mut self, vcpu: &dyn Vcpu, fd: u32, iov: u32, iovcnt: u32) -> Result<u32, Errno> {
        if iovcnt > IOV_MAX {
            return Err(Errno::EINVAL);
        }
        let mut total = 0u32;
        for i in 0..iovcnt {
            let mut entry = [0; 8];
            vcpu.read_mem(iov.checked_add(8 * i).ok_or(Errno::EFAULT)?, &mut entry)?;
            let base = u32::from_le_bytes(entry[..4].try_into().unwrap());
            let len = u32::from_le_bytes(entry[4..].try_into().unwrap());
            let written = self.write(vcpu, fd, base, len)?;
            total = total.saturating_add(written);
            if written < len {
                break;
            }
        }
        Ok(total)
    }

    fn close(&mut self, fd: u32) -> Result<u32, Errno> {
        if self.open.remove(&fd) {
            Ok(0)
        } else {
            Err(Errno::EBADF)
        }
    }

    /// The guest has one hart, so its CPU mask has one bit set. Like Linux, write
    /// only the kernel's mask size; libc zeroes the rest of the caller's buffer.
    fn sched_getaffinity(&self, vcpu: &mut dyn Vcpu, size: u32, mask: u32) -> Result<u32, Errno> {
        if size < 4 {
            return Err(Errno::EINVAL);
        }
        vcpu.write_mem(mask, &1u32.to_le_bytes())?;
        Ok(4)
    }

    /// Anonymous mappings only, until there is a filesystem to map files from.
    /// The offset argument is ignored, which is correct for anonymous memory.
    fn mmap(&mut self, vcpu: &mut dyn Vcpu, addr: u32, len: u32, flags: u32) -> Result<u32, Errno> {
        if flags & MAP_ANONYMOUS == 0 {
            return Err(Errno::ENODEV);
        }
        Ok(self.mm.map(vcpu, addr, len, flags & MAP_FIXED != 0)?)
    }

    fn output(&mut self, fd: u32) -> Result<&mut dyn Write, Errno> {
        if !self.open.contains(&fd) {
            return Err(Errno::EBADF);
        }
        match fd {
            1 => Ok(&mut *self.stdio.stdout),
            2 => Ok(&mut *self.stdio.stderr),
            _ => Err(Errno::EBADF),
        }
    }
}
