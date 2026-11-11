//! Linux system calls for RV32 guests, implemented by the host.
//!
//! In container mode this is the guest's kernel: the vCPU exits on every
//! `ecall`, the host does the work, and the result goes back in a0. It is also
//! the security boundary, so every guest pointer and length is treated as hostile.
//!
//! State is split the way Linux splits it. [`Linux`] holds what the whole
//! container shares, and [`Process`] what each process owns. Calls that create,
//! replace, wait for or signal processes come back as an [`Action`] for the
//! container, which owns the vCPUs.
//!
//! Time and randomness are virtual, so a run depends only on its inputs. The
//! clock advances one nanosecond per retired instruction, as if the guest ran at
//! 1 GHz, plus whatever time the guest sleeps, which costs nothing. `getrandom`
//! is a seeded PRNG.

mod abi;
mod files;
mod rootfs;
mod signal;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use abi::Errno;
use abi::*;
use files::{FdTable, MAX_FDS};
pub use rootfs::{Resolved, RootFs};
pub use signal::{Delivery, FrozenSignals, SigAction, Signals};

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

/// How a process ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    Exited(u8),
    Killed(u32),
}

impl Termination {
    /// The status a shell reports: the exit code, or 128 plus the signal.
    pub fn status(self) -> u8 {
        match self {
            Self::Exited(code) => code,
            Self::Killed(signal) => (128 + signal) as u8,
        }
    }

    /// The status word `wait4` stores.
    pub fn wait_status(self) -> u32 {
        match self {
            Self::Exited(code) => (code as u32) << 8,
            Self::Killed(signal) => signal & 0x7f,
        }
    }
}

/// A program to replace the calling process's image, read by `execve`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exec {
    pub elf: Vec<u8>,
    pub argv: Vec<Vec<u8>>,
    pub envp: Vec<Vec<u8>>,
}

/// A request to reap a child: `wait4` or `waitid`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wait {
    /// The child to wait for, or `None` for any child.
    pub pid: Option<u32>,
    pub nohang: bool,
    pub report: Report,
}

/// Where a wait call reports the child it reaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Report {
    /// `wait4`: a status word at this address (0 for none); returns the pid.
    Status(u32),
    /// `waitid`: a `siginfo_t` at this address; returns 0.
    Siginfo(u32),
}

/// What the container should do after a system call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Put this value in a0 and resume after the `ecall`.
    Return(u32),
    /// The call cannot finish yet: park the process and retry the call later.
    Block,
    /// The process ends.
    Exit(Termination),
    /// Fork the process. The child gets `stack` as its stack pointer if it is
    /// not 0, and its pid is stored at `parent_tid` if that is not 0.
    Fork { stack: u32, parent_tid: u32 },
    /// Replace the process image.
    Exec(Exec),
    /// Reap a child.
    Wait(Wait),
    /// Send `signal` to `pid`, which follows `kill`'s conventions for 0 and
    /// negative values.
    Kill { pid: i32, signal: u32 },
    /// The registers are already set, as after `rt_sigreturn`: resume there.
    Resume,
    /// Move `pid`, a child of the caller, into process group `pgid`.
    SetPgid { pid: u32, pgid: u32 },
    /// Report the process group of `pid`.
    GetPgid { pid: u32 },
}

/// The shared state a warm image stores: the virtual clock and the PRNG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenLinux {
    pub slept_ns: u64,
    pub random: u64,
}

/// What a warm image stores of a process. The root filesystem is not part of
/// it: it comes from the image's layers when the image runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenProcess {
    pub cwd: String,
    pub umask: u32,
    pub mm: AddressSpace,
    pub signals: FrozenSignals,
}

/// Why a process cannot be frozen.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FreezeError {
    #[error("only a single process can be frozen, and this container has {0}")]
    Processes(usize),
    #[error("files other than stdin, stdout and stderr are open, and they cannot be frozen yet")]
    OpenFiles,
    #[error("the working directory is not valid UTF-8")]
    Cwd,
}

/// What the whole container shares.
pub struct Linux {
    /// Virtual nanoseconds the guest has spent asleep.
    slept_ns: u64,
    random: u64,
    /// System calls the guest made that are not implemented, and how many times.
    pub unsupported: BTreeMap<u32, u64>,
}

/// What each process owns.
pub struct Process {
    pub pid: u32,
    pub ppid: u32,
    pub pgid: u32,
    pub sid: u32,
    files: FdTable,
    root: Option<Arc<RootFs>>,
    cwd: Vec<u8>,
    umask: u32,
    pub signals: Signals,
    pub mm: AddressSpace,
}

impl Default for Linux {
    fn default() -> Self {
        Self::new()
    }
}

impl Linux {
    pub fn new() -> Self {
        Self {
            slept_ns: 0,
            random: RANDOM_SEED,
            unsupported: BTreeMap::new(),
        }
    }

    pub fn freeze(&self) -> FrozenLinux {
        FrozenLinux {
            slept_ns: self.slept_ns,
            random: self.random,
        }
    }

    pub fn thaw(frozen: &FrozenLinux) -> Self {
        Self {
            slept_ns: frozen.slept_ns,
            random: frozen.random,
            unsupported: BTreeMap::new(),
        }
    }

    /// Service the system call `process` just made. `retired` counts the
    /// instructions the container has retired so far, which drives the clock.
    pub fn handle(&mut self, process: &mut Process, vcpu: &mut dyn Vcpu, retired: u64) -> Action {
        let nr = vcpu.reg(A7);
        let a: [u32; 6] = std::array::from_fn(|i| vcpu.reg(A0 + i));
        let p = process;
        let result = match nr {
            nr::EXIT | nr::EXIT_GROUP => return Action::Exit(Termination::Exited(a[0] as u8)),
            nr::CLONE => match clone(a[0]) {
                Ok(()) => {
                    return Action::Fork {
                        stack: a[1],
                        parent_tid: if a[0] & CLONE_PARENT_SETTID != 0 {
                            a[2]
                        } else {
                            0
                        },
                    };
                }
                Err(errno) => Err(errno),
            },
            nr::EXECVE => match p.execve(vcpu, a[0], a[1], a[2]) {
                Ok(exec) => return Action::Exec(exec),
                Err(errno) => Err(errno),
            },
            nr::WAIT4 => match zero_fill(vcpu, a[3], RUSAGE_SIZE) {
                Ok(_) => return Action::Wait(wait4(a[0], a[1], a[2])),
                Err(errno) => Err(errno),
            },
            nr::WAITID => match waitid(a[0], a[1], a[2], a[3]) {
                Ok(wait) => return Action::Wait(wait),
                Err(errno) => Err(errno),
            },
            nr::KILL | nr::TKILL | nr::TGKILL => {
                let (pid, signal) = match nr {
                    nr::TGKILL => (a[1], a[2]),
                    _ => (a[0], a[1]),
                };
                if signal > 64 {
                    Err(Errno::EINVAL)
                } else {
                    return Action::Kill {
                        pid: pid as i32,
                        signal,
                    };
                }
            }
            nr::READ => p.read(vcpu, a[0], a[1], a[2]),
            nr::WRITE => p.write(vcpu, a[0], a[1], a[2]),
            nr::READV => p.readv(vcpu, a[0], a[1], a[2]),
            nr::WRITEV => p.writev(vcpu, a[0], a[1], a[2]),
            nr::OPENAT => p.openat(vcpu, a[0], a[1], a[2], a[3]),
            nr::CLOSE => p.close(a[0]),
            nr::PIPE2 => p.pipe2(vcpu, a[0], a[1]),
            nr::LLSEEK => p.llseek(vcpu, a[0], (a[1], a[2]), a[3], a[4]),
            nr::GETDENTS64 => p.getdents64(vcpu, a[0], a[1], a[2]),
            nr::STATX => p.statx(vcpu, a[0], a[1], a[2], a[4]),
            nr::FACCESSAT | nr::FACCESSAT2 => p.faccessat(vcpu, a[0], a[1], a[2]),
            nr::GETCWD => p.getcwd(vcpu, a[0], a[1]),
            nr::CHDIR => p.chdir(vcpu, a[0]),
            nr::MKDIRAT => p.mkdirat(vcpu, a[0], a[1], a[2]),
            nr::UNLINKAT => p.unlinkat(vcpu, a[0], a[1], a[2]),
            nr::SYMLINKAT => p.symlinkat(vcpu, a[0], a[1], a[2]),
            nr::READLINKAT => p.readlinkat(vcpu, a[0], a[1], a[2], a[3]),
            nr::RENAMEAT2 => p.renameat2(vcpu, (a[0], a[1]), (a[2], a[3]), a[4]),
            nr::SENDFILE64 => p.sendfile(vcpu, a[0], a[1], a[2], a[3]),
            nr::UTIMENSAT_TIME64 => {
                let now = EPOCH_NS + retired + self.slept_ns;
                p.utimensat(vcpu, a[0], a[1], a[2], a[3], now)
            }
            nr::DUP => p.dup(a[0], 0, false),
            nr::DUP3 => p.dup3(a[0], a[1], a[2]),
            nr::FCNTL64 => p.fcntl(a[0], a[1], a[2]),
            nr::FSYNC => p.fsync(a[0]),
            nr::IOCTL => Err(Errno::ENOTTY), // no terminals
            nr::UMASK => Ok(std::mem::replace(&mut p.umask, a[0] & 0o777)),
            nr::BRK => Ok(p.mm.set_brk(vcpu, a[0])),
            nr::MMAP => p.mmap(vcpu, a[0], a[1], a[3]),
            nr::MUNMAP => p.mm.unmap(a[0], a[1]).map(|()| 0).map_err(Errno::from),
            nr::MREMAP => Err(Errno::ENOMEM), // callers fall back to map, copy, unmap
            nr::MPROTECT | nr::MADVISE => Ok(0), // guest memory has no protection bits
            nr::CLOCK_GETTIME64 => self.clock_gettime(vcpu, a[0], a[1], retired),
            nr::CLOCK_NANOSLEEP_TIME64 => self.clock_nanosleep(vcpu, a[0], a[1], a[2], retired),
            nr::GETRANDOM => self.getrandom(vcpu, a[0], a[1]),
            nr::UNAME => uname(vcpu, a[0]),
            nr::RT_SIGACTION => p.sigaction(vcpu, a[0], a[1], a[2]),
            // Wait with a temporary mask until a signal arrives. The container
            // delivers it to the parked process, which ends the call with EINTR.
            nr::RT_SIGSUSPEND => match read_u64(vcpu, a[0]) {
                Ok(mask) => {
                    p.signals.suspend(mask);
                    return Action::Block;
                }
                Err(errno) => Err(errno),
            },
            nr::RT_SIGRETURN => {
                return match p.signals.sigreturn(vcpu) {
                    Ok(()) => Action::Resume,
                    Err(_) => Action::Exit(Termination::Killed(SIGSEGV)),
                };
            }
            nr::RT_SIGPROCMASK => p.sigprocmask(vcpu, a[0], a[1], a[2]),
            nr::PRLIMIT64 => prlimit(vcpu, a[1], a[3]),
            nr::SCHED_GETAFFINITY => sched_getaffinity(vcpu, a[1], a[2]),
            nr::SET_TID_ADDRESS | nr::GETPID | nr::GETTID => Ok(p.pid),
            nr::GETPPID => Ok(p.ppid),
            nr::GETPGID if a[0] == 0 || a[0] == p.pid => Ok(p.pgid),
            nr::GETPGID => return Action::GetPgid { pid: a[0] },
            nr::SETPGID => {
                let pid = if a[0] == 0 { p.pid } else { a[0] };
                let pgid = if a[1] == 0 { pid } else { a[1] };
                if pid != p.pid {
                    return Action::SetPgid { pid, pgid };
                }
                p.pgid = pgid;
                Ok(0)
            }
            nr::GETSID => Ok(p.sid),
            nr::SETSID if p.pgid == p.pid => Err(Errno::EPERM),
            nr::SETSID => {
                (p.sid, p.pgid) = (p.pid, p.pid);
                Ok(p.pid)
            }
            nr::GETUID | nr::GETEUID | nr::GETGID | nr::GETEGID => Ok(0),
            nr::GETGROUPS => Ok(0),
            _ => {
                *self.unsupported.entry(nr).or_default() += 1;
                Err(Errno::ENOSYS)
            }
        };
        match result {
            Ok(value) => Action::Return(value),
            Err(Errno::WOULD_BLOCK) => Action::Block,
            // Writing to a pipe nobody reads raises SIGPIPE, which kills by default.
            Err(Errno::EPIPE) => match p.signals.raise(SIGPIPE) {
                Some(termination) => Action::Exit(termination),
                None => Action::Return(Errno::EPIPE.0.wrapping_neg()),
            },
            Err(Errno(errno)) => Action::Return(errno.wrapping_neg()),
        }
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

impl Process {
    /// A process with no parent, its standard streams on `stdio`, and a heap that
    /// starts at `brk` and shares memory with mappings up to `top`.
    pub fn new(pid: u32, stdio: Stdio, brk: u32, top: u32) -> Self {
        Self {
            pid,
            ppid: 0,
            pgid: pid,
            sid: pid,
            files: FdTable::new(stdio),
            root: None,
            cwd: b"/".to_vec(),
            umask: 0o022,
            signals: Signals::default(),
            mm: AddressSpace::new(brk, top),
        }
    }

    /// Capture what a warm image needs to resume this process.
    pub fn freeze(&self) -> Result<FrozenProcess, FreezeError> {
        if !self.files.only_stdio() {
            return Err(FreezeError::OpenFiles);
        }
        Ok(FrozenProcess {
            cwd: String::from_utf8(self.cwd.clone()).map_err(|_| FreezeError::Cwd)?,
            umask: self.umask,
            mm: self.mm.clone(),
            signals: self.signals.freeze(),
        })
    }

    /// Rebuild a frozen process as init, with its standard streams on `stdio`.
    pub fn thaw(frozen: &FrozenProcess, stdio: Stdio) -> Self {
        let mut process = Self::new(1, stdio, 0, 0);
        process.cwd = frozen.cwd.clone().into_bytes();
        process.umask = frozen.umask;
        process.mm = frozen.mm.clone();
        process.signals = Signals::thaw(&frozen.signals);
        process
    }

    /// Change the working directory, which must be a directory in the root.
    pub fn set_cwd(&mut self, path: &str) -> Result<(), Errno> {
        let root = self.root.as_ref().ok_or(Errno::ENOENT)?;
        let resolved = root.resolve(&self.cwd, path.as_bytes(), true)?;
        if !std::fs::metadata(&resolved.host)?.is_dir() {
            return Err(Errno::ENOTDIR);
        }
        self.cwd = resolved.guest;
        Ok(())
    }

    /// The process's root filesystem, if it has one.
    pub fn root(&self) -> Option<&RootFs> {
        self.root.as_deref()
    }

    /// Give the process a filesystem: the host directory `root` becomes its `/`,
    /// and children inherit it. Without one, every path lookup fails with ENOENT.
    pub fn set_root(&mut self, root: &Path) -> std::io::Result<()> {
        self.root = Some(Arc::new(RootFs::new(root)?));
        Ok(())
    }

    /// The child of a fork: a copy of this process that shares its open files.
    pub fn fork(&self, pid: u32) -> Self {
        Self {
            pid,
            ppid: self.pid,
            pgid: self.pgid,
            sid: self.sid,
            files: self.files.fork(),
            root: self.root.clone(),
            cwd: self.cwd.clone(),
            umask: self.umask,
            signals: self.signals.fork(),
            mm: self.mm.clone(),
        }
    }

    /// Start a new program: a fresh address space, close-on-exec files closed,
    /// and handlers reset to the default. Ignored signals stay ignored.
    pub fn exec(&mut self, brk: u32, top: u32) {
        self.mm = AddressSpace::new(brk, top);
        self.files.close_on_exec();
        self.signals.exec();
    }

    /// Close everything the process holds, as when it exits.
    pub fn release(&mut self) {
        self.files.close_all();
    }

    fn sigaction(
        &mut self,
        vcpu: &mut dyn Vcpu,
        signal: u32,
        act: u32,
        old: u32,
    ) -> Result<u32, Errno> {
        // struct sigaction on riscv32: handler, flags, then a 64-bit mask
        let new = if act == 0 {
            None
        } else {
            let mut raw = [0u8; 16];
            vcpu.read_mem(act, &mut raw)?;
            Some(SigAction {
                handler: u32::from_le_bytes(raw[..4].try_into().unwrap()),
                flags: u32::from_le_bytes(raw[4..8].try_into().unwrap()),
                mask: u64::from_le_bytes(raw[8..].try_into().unwrap()),
            })
        };
        let previous = self.signals.sigaction(signal, new)?;
        if old != 0 {
            let mut out = [0u8; 16];
            out[..4].copy_from_slice(&previous.handler.to_le_bytes());
            out[4..8].copy_from_slice(&previous.flags.to_le_bytes());
            out[8..].copy_from_slice(&previous.mask.to_le_bytes());
            vcpu.write_mem(old, &out)?;
        }
        Ok(0)
    }

    fn sigprocmask(
        &mut self,
        vcpu: &mut dyn Vcpu,
        how: u32,
        set: u32,
        old: u32,
    ) -> Result<u32, Errno> {
        let current = self.signals.blocked;
        if set != 0 {
            let set = read_u64(vcpu, set)?;
            self.signals.set_mask(match how {
                SIG_BLOCK => current | set,
                SIG_UNBLOCK => current & !set,
                SIG_SETMASK => set,
                _ => return Err(Errno::EINVAL),
            });
        }
        if old != 0 {
            vcpu.write_mem(old, &current.to_le_bytes())?;
        }
        Ok(0)
    }

    /// Anonymous mappings only. The offset argument is ignored, which is correct
    /// for anonymous memory.
    fn mmap(&mut self, vcpu: &mut dyn Vcpu, addr: u32, len: u32, flags: u32) -> Result<u32, Errno> {
        if flags & MAP_ANONYMOUS == 0 {
            return Err(Errno::ENODEV);
        }
        Ok(self.mm.map(vcpu, addr, len, flags & MAP_FIXED != 0)?)
    }
}

/// `struct rusage` with 32-bit `timeval`s and 14 longs.
const RUSAGE_SIZE: u32 = 72;

/// Forks are supported, threads are not. `vfork` children get a copy of memory
/// instead of sharing it, which is safe for the usual pattern of calling
/// `execve` or `_exit` straight away.
fn clone(flags: u32) -> Result<(), Errno> {
    if flags & CLONE_THREAD != 0 || (flags & CLONE_VM != 0 && flags & CLONE_VFORK == 0) {
        return Err(Errno::EINVAL);
    }
    Ok(())
}

/// Process groups are not tracked, so waiting for a group waits for any child.
fn wait4(pid: u32, status: u32, options: u32) -> Wait {
    let pid = pid as i32;
    Wait {
        pid: (pid > 0).then_some(pid as u32),
        nohang: options & WNOHANG != 0,
        report: Report::Status(status),
    }
}

fn waitid(idtype: u32, id: u32, info: u32, options: u32) -> Result<Wait, Errno> {
    if options & WEXITED == 0 {
        return Err(Errno::EINVAL);
    }
    Ok(Wait {
        // process groups are not tracked: P_PGID waits for any child, like P_ALL
        pid: (idtype == P_PID).then_some(id),
        nohang: options & WNOHANG != 0,
        report: Report::Siginfo(info),
    })
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

fn read_u64(vcpu: &dyn Vcpu, addr: u32) -> Result<u64, Errno> {
    let mut raw = [0u8; 8];
    vcpu.read_mem(addr, &mut raw)?;
    Ok(u64::from_le_bytes(raw))
}
