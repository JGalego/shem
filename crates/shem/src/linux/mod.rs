//! Linux system calls for RV32 guests, implemented by the host.
//!
//! In container mode this is the guest's kernel: the vCPU exits on every
//! `ecall`, the host does the work, and the result goes back in a0. It is also
//! the security boundary, so every guest pointer and length is treated as hostile.
//!
//! Time and randomness are virtual, so a run depends only on its inputs. The
//! clock advances one nanosecond per retired instruction, as if the guest ran at
//! 1 GHz, plus whatever time the guest sleeps, which costs nothing. `getrandom`
//! is a seeded PRNG.

mod abi;
mod files;
mod rootfs;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

pub use abi::Errno;
use abi::*;
use files::{FdTable, MAX_FDS};
pub use rootfs::{Resolved, RootFs};

use crate::container::STACK_SIZE;
use crate::mm::AddressSpace;
use crate::vcpu::{A0, A7, Vcpu};

/// Largest single transfer. Short reads and writes are legal, and the cap stops
/// a guest from making the host allocate gigabytes.
const MAX_IO: u32 = 1 << 20;
const IOV_MAX: u32 = 1024;

/// The guest's wall clock starts at 2026-01-01T00:00:00Z.
const EPOCH_NS: u64 = 1_767_225_600 * NS_PER_SEC;
pub(crate) const NS_PER_SEC: u64 = 1_000_000_000;
const RANDOM_SEED: u64 = 0x5348_454d;

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
    files: FdTable,
    root: Option<RootFs>,
    cwd: Vec<u8>,
    umask: u32,
    /// Virtual nanoseconds the guest has spent asleep.
    slept_ns: u64,
    random: u64,
    pub mm: AddressSpace,
    /// System calls the guest made that are not implemented, and how many times.
    pub unsupported: BTreeMap<u32, u64>,
}

impl Linux {
    /// `brk` is where the heap starts; the heap and mappings share memory up to `top`.
    pub fn new(stdio: Stdio, brk: u32, top: u32) -> Self {
        Self {
            files: FdTable::new(stdio),
            root: None,
            cwd: b"/".to_vec(),
            umask: 0o022,
            slept_ns: 0,
            random: RANDOM_SEED,
            mm: AddressSpace::new(brk, top),
            unsupported: BTreeMap::new(),
        }
    }

    /// Give the guest a filesystem: the host directory `root` becomes its `/`.
    /// Without one, every path lookup fails with ENOENT.
    pub fn set_root(&mut self, root: &Path) -> std::io::Result<()> {
        self.root = Some(RootFs::new(root)?);
        Ok(())
    }

    /// Service the system call the guest just made. `retired` counts the
    /// instructions the guest has retired so far, which drives the clock.
    pub fn handle(&mut self, vcpu: &mut dyn Vcpu, retired: u64) -> Action {
        let nr = vcpu.reg(A7);
        let a: [u32; 6] = std::array::from_fn(|i| vcpu.reg(A0 + i));
        let result = match nr {
            nr::EXIT | nr::EXIT_GROUP => return Action::Exit(a[0] as u8),
            nr::KILL | nr::TGKILL => {
                let signal = if nr == nr::KILL { a[1] } else { a[2] };
                match signal_status(signal) {
                    Ok(Some(status)) => return Action::Exit(status),
                    Ok(None) => Ok(0),
                    Err(errno) => Err(errno),
                }
            }
            nr::READ => self.read(vcpu, a[0], a[1], a[2]),
            nr::WRITE => self.write(vcpu, a[0], a[1], a[2]),
            nr::READV => self.readv(vcpu, a[0], a[1], a[2]),
            nr::WRITEV => self.writev(vcpu, a[0], a[1], a[2]),
            nr::OPENAT => self.openat(vcpu, a[0], a[1], a[2], a[3]),
            nr::CLOSE => self.close(a[0]),
            nr::LLSEEK => self.llseek(vcpu, a[0], (a[1], a[2]), a[3], a[4]),
            nr::GETDENTS64 => self.getdents64(vcpu, a[0], a[1], a[2]),
            nr::STATX => self.statx(vcpu, a[0], a[1], a[2], a[4]),
            nr::FACCESSAT | nr::FACCESSAT2 => self.faccessat(vcpu, a[0], a[1], a[2]),
            nr::GETCWD => self.getcwd(vcpu, a[0], a[1]),
            nr::CHDIR => self.chdir(vcpu, a[0]),
            nr::MKDIRAT => self.mkdirat(vcpu, a[0], a[1], a[2]),
            nr::UNLINKAT => self.unlinkat(vcpu, a[0], a[1], a[2]),
            nr::SYMLINKAT => self.symlinkat(vcpu, a[0], a[1], a[2]),
            nr::READLINKAT => self.readlinkat(vcpu, a[0], a[1], a[2], a[3]),
            nr::RENAMEAT2 => self.renameat2(vcpu, (a[0], a[1]), (a[2], a[3]), a[4]),
            nr::SENDFILE64 => self.sendfile(vcpu, a[0], a[1], a[2], a[3]),
            nr::UTIMENSAT_TIME64 => self.utimensat(vcpu, a[0], a[1], a[2], a[3], retired),
            nr::DUP => self.dup(a[0], 0, false),
            nr::DUP3 => self.dup3(a[0], a[1], a[2]),
            nr::FCNTL64 => self.fcntl(a[0], a[1], a[2]),
            nr::FSYNC => self.fsync(a[0]),
            nr::IOCTL => Err(Errno::ENOTTY), // no terminals
            nr::UMASK => Ok(std::mem::replace(&mut self.umask, a[0] & 0o777)),
            nr::BRK => Ok(self.mm.set_brk(vcpu, a[0])),
            nr::MMAP => self.mmap(vcpu, a[0], a[1], a[3]),
            nr::MUNMAP => self.mm.unmap(a[0], a[1]).map(|()| 0).map_err(Errno::from),
            nr::MREMAP => Err(Errno::ENOMEM), // callers fall back to map, copy, unmap
            nr::MPROTECT | nr::MADVISE => Ok(0), // guest memory has no protection bits
            nr::CLOCK_GETTIME64 => self.clock_gettime(vcpu, a[0], a[1], retired),
            nr::CLOCK_NANOSLEEP_TIME64 => self.clock_nanosleep(vcpu, a[0], a[1], a[2], retired),
            nr::GETRANDOM => self.getrandom(vcpu, a[0], a[1]),
            nr::UNAME => uname(vcpu, a[0]),
            // No signal is ever delivered: handlers are accepted and never run,
            // and the previous action and mask always read as the defaults.
            nr::RT_SIGACTION => zero_fill(vcpu, a[2], 8 + a[3].min(128)),
            nr::RT_SIGPROCMASK => zero_fill(vcpu, a[2], a[3].min(128)),
            nr::PRLIMIT64 => prlimit(vcpu, a[1], a[3]),
            nr::SCHED_GETAFFINITY => sched_getaffinity(vcpu, a[1], a[2]),
            // The only process and thread, running as root with no parent.
            nr::SET_TID_ADDRESS | nr::GETPID | nr::GETTID => Ok(1),
            nr::GETGROUPS => Ok(0), // no supplementary groups
            nr::GETPPID | nr::GETUID | nr::GETEUID | nr::GETGID | nr::GETEGID => Ok(0),
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

    /// Anonymous mappings only. The offset argument is ignored, which is correct
    /// for anonymous memory.
    fn mmap(&mut self, vcpu: &mut dyn Vcpu, addr: u32, len: u32, flags: u32) -> Result<u32, Errno> {
        if flags & MAP_ANONYMOUS == 0 {
            return Err(Errno::ENODEV);
        }
        Ok(self.mm.map(vcpu, addr, len, flags & MAP_FIXED != 0)?)
    }

    fn now_ns(&self, clock: u32, retired: u64) -> Result<u64, Errno> {
        let elapsed = retired + self.slept_ns;
        match clock {
            CLOCK_REALTIME | CLOCK_REALTIME_COARSE | CLOCK_TAI => Ok(EPOCH_NS + elapsed),
            0..=11 => Ok(elapsed), // monotonic, boot-time and CPU-time clocks
            _ => Err(Errno::EINVAL),
        }
    }

    fn clock_gettime(
        &self,
        vcpu: &mut dyn Vcpu,
        clock: u32,
        ts: u32,
        retired: u64,
    ) -> Result<u32, Errno> {
        let ns = self.now_ns(clock, retired)?;
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&((ns / NS_PER_SEC) as i64).to_le_bytes());
        out[8..].copy_from_slice(&((ns % NS_PER_SEC) as i64).to_le_bytes());
        vcpu.write_mem(ts, &out)?;
        Ok(0)
    }

    /// Sleeping advances the virtual clock and returns at once.
    fn clock_nanosleep(
        &mut self,
        vcpu: &mut dyn Vcpu,
        clock: u32,
        flags: u32,
        request: u32,
        retired: u64,
    ) -> Result<u32, Errno> {
        let mut ts = [0u8; 16];
        vcpu.read_mem(request, &mut ts)?;
        let sec = i64::from_le_bytes(ts[..8].try_into().unwrap());
        let nsec = i64::from_le_bytes(ts[8..].try_into().unwrap());
        if sec < 0 || !(0..NS_PER_SEC as i64).contains(&nsec) {
            return Err(Errno::EINVAL);
        }
        let requested = (sec as u64)
            .saturating_mul(NS_PER_SEC)
            .saturating_add(nsec as u64);
        let now = self.now_ns(clock, retired)?;
        let duration = if flags & TIMER_ABSTIME != 0 {
            requested.saturating_sub(now)
        } else {
            requested
        };
        self.slept_ns = self.slept_ns.saturating_add(duration);
        Ok(0)
    }

    fn getrandom(&mut self, vcpu: &mut dyn Vcpu, buf: u32, len: u32) -> Result<u32, Errno> {
        let mut out = Vec::with_capacity(len.min(MAX_IO) as usize);
        while out.len() < out.capacity() {
            // SplitMix64
            self.random = self.random.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.random;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            out.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
        }
        out.truncate(len.min(MAX_IO) as usize);
        vcpu.write_mem(buf, &out)?;
        Ok(out.len() as u32)
    }
}

/// What sending `signal` to itself does to a process that has no handlers
/// running: nothing for signals that are ignored by default, otherwise exit
/// with 128 plus the signal number, the status a shell reports.
fn signal_status(signal: u32) -> Result<Option<u8>, Errno> {
    const IGNORED: [u32; 5] = [0, 17, 18, 23, 28]; // none, SIGCHLD, SIGCONT, SIGURG, SIGWINCH
    match signal {
        _ if IGNORED.contains(&signal) => Ok(None),
        1..=64 => Ok(Some((128 + signal) as u8)),
        _ => Err(Errno::EINVAL),
    }
}

fn uname(vcpu: &mut dyn Vcpu, buf: u32) -> Result<u32, Errno> {
    const FIELD: usize = 65;
    let fields: [&[u8]; 6] = [
        b"Linux",
        b"shem",
        b"6.6.0-shem",
        b"#1",
        b"riscv32",
        b"(none)",
    ];
    let mut out = [0u8; 6 * FIELD];
    for (i, field) in fields.iter().enumerate() {
        out[i * FIELD..i * FIELD + field.len()].copy_from_slice(field);
    }
    vcpu.write_mem(buf, &out)?;
    Ok(0)
}

fn prlimit(vcpu: &mut dyn Vcpu, resource: u32, old: u32) -> Result<u32, Errno> {
    if old != 0 {
        let limit = match resource {
            RLIMIT_NOFILE => MAX_FDS as u64,
            RLIMIT_STACK => STACK_SIZE as u64,
            _ => RLIM_INFINITY,
        };
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&limit.to_le_bytes());
        out[8..].copy_from_slice(&limit.to_le_bytes());
        vcpu.write_mem(old, &out)?;
    }
    Ok(0) // new limits are accepted and ignored
}

/// The guest has one hart, so its CPU mask has one bit set. Like Linux, write
/// only the kernel's mask size; libc zeroes the rest of the caller's buffer.
fn sched_getaffinity(vcpu: &mut dyn Vcpu, size: u32, mask: u32) -> Result<u32, Errno> {
    if size < 4 {
        return Err(Errno::EINVAL);
    }
    vcpu.write_mem(mask, &1u32.to_le_bytes())?;
    Ok(4)
}

/// Zero `len` bytes at `addr`, unless `addr` is null.
fn zero_fill(vcpu: &mut dyn Vcpu, addr: u32, len: u32) -> Result<u32, Errno> {
    if addr != 0 {
        vcpu.write_mem(addr, &vec![0; len as usize])?;
    }
    Ok(0)
}
