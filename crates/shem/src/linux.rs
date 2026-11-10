//! Linux system calls for RV32 guests, implemented by the host.
//!
//! In container mode this is the guest's kernel: the vCPU exits on every
//! `ecall`, the host does the work, and the result goes back in a0. It is also
//! the security boundary, so every guest pointer and length is treated as hostile.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};

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
const SYS_BRK: u32 = 214;

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
    pub const EINVAL: Self = Self(22);
    pub const ENOTTY: Self = Self(25);
    pub const ENOSYS: Self = Self(38);
}

impl From<MemoryFault> for Errno {
    fn from(_: MemoryFault) -> Self {
        Self::EFAULT
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
    brk_start: u32,
    brk: u32,
    brk_limit: u32,
    /// System calls the guest made that are not implemented, and how many times.
    pub unsupported: BTreeMap<u32, u64>,
}

impl Linux {
    pub fn new(stdio: Stdio, brk: u32, brk_limit: u32) -> Self {
        Self {
            stdio,
            open: BTreeSet::from([0, 1, 2]),
            brk_start: brk,
            brk,
            brk_limit,
            unsupported: BTreeMap::new(),
        }
    }

    /// The current program break.
    pub fn brk(&self) -> u32 {
        self.brk
    }

    pub fn handle(&mut self, vcpu: &mut dyn Vcpu) -> Action {
        let nr = vcpu.reg(A7);
        let [a0, a1, a2] = std::array::from_fn(|i| vcpu.reg(A0 + i));
        let result = match nr {
            SYS_EXIT | SYS_EXIT_GROUP => return Action::Exit(a0 as u8),
            SYS_READ => self.read(vcpu, a0, a1, a2),
            SYS_WRITE => self.write(vcpu, a0, a1, a2),
            SYS_WRITEV => self.writev(vcpu, a0, a1, a2),
            SYS_CLOSE => self.close(a0),
            SYS_BRK => Ok(self.set_brk(vcpu, a0)),
            SYS_IOCTL => Err(Errno::ENOTTY),
            SYS_SET_TID_ADDRESS => Ok(1), // the only thread is tid 1
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

    /// Move the program break if the request is in range, and return where the
    /// break now is, as Linux does.
    fn set_brk(&mut self, vcpu: &mut dyn Vcpu, addr: u32) -> u32 {
        if (self.brk_start..=self.brk_limit).contains(&addr) {
            // Memory past the break may hold data from before it last shrank.
            let grown = addr.saturating_sub(self.brk) as usize;
            if grown == 0 || vcpu.write_mem(self.brk, &vec![0; grown]).is_ok() {
                self.brk = addr;
            }
        }
        self.brk
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
