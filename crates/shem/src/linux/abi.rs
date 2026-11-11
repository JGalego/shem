//! The Linux ABI as an RV32 guest sees it: asm-generic system call numbers with
//! 64-bit time, errno values, flags, and the layouts of structs the host writes.

use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;

use crate::mm::MapError;
use crate::vcpu::MemoryFault;

/// System call numbers.
pub mod nr {
    pub const GETCWD: u32 = 17;
    pub const DUP: u32 = 23;
    pub const DUP3: u32 = 24;
    pub const FCNTL64: u32 = 25;
    pub const IOCTL: u32 = 29;
    pub const MKDIRAT: u32 = 34;
    pub const UNLINKAT: u32 = 35;
    pub const SYMLINKAT: u32 = 36;
    pub const FACCESSAT: u32 = 48;
    pub const CHDIR: u32 = 49;
    pub const OPENAT: u32 = 56;
    pub const CLOSE: u32 = 57;
    pub const PIPE2: u32 = 59;
    pub const GETDENTS64: u32 = 61;
    pub const LLSEEK: u32 = 62;
    pub const READ: u32 = 63;
    pub const WRITE: u32 = 64;
    pub const READV: u32 = 65;
    pub const WRITEV: u32 = 66;
    pub const READLINKAT: u32 = 78;
    pub const FSYNC: u32 = 82;
    pub const EXIT: u32 = 93;
    pub const EXIT_GROUP: u32 = 94;
    pub const WAITID: u32 = 95;
    pub const SET_TID_ADDRESS: u32 = 96;
    pub const SCHED_GETAFFINITY: u32 = 123;
    pub const KILL: u32 = 129;
    pub const TKILL: u32 = 130;
    pub const TGKILL: u32 = 131;
    pub const RT_SIGSUSPEND: u32 = 133;
    pub const RT_SIGACTION: u32 = 134;
    pub const RT_SIGPROCMASK: u32 = 135;
    pub const RT_SIGRETURN: u32 = 139;
    pub const UNAME: u32 = 160;
    pub const UMASK: u32 = 166;
    pub const GETPID: u32 = 172;
    pub const GETPPID: u32 = 173;
    pub const GETUID: u32 = 174;
    pub const GETEUID: u32 = 175;
    pub const GETGID: u32 = 176;
    pub const GETEGID: u32 = 177;
    pub const GETTID: u32 = 178;
    pub const BRK: u32 = 214;
    pub const MUNMAP: u32 = 215;
    pub const MREMAP: u32 = 216;
    pub const CLONE: u32 = 220;
    pub const EXECVE: u32 = 221;
    pub const MMAP: u32 = 222;
    pub const MPROTECT: u32 = 226;
    pub const MADVISE: u32 = 233;
    pub const WAIT4: u32 = 260;
    pub const PRLIMIT64: u32 = 261;
    pub const SENDFILE64: u32 = 71;
    pub const SETPGID: u32 = 154;
    pub const GETPGID: u32 = 155;
    pub const GETSID: u32 = 156;
    pub const SETSID: u32 = 157;
    pub const GETGROUPS: u32 = 158;
    pub const RENAMEAT2: u32 = 276;
    pub const GETRANDOM: u32 = 278;
    pub const STATX: u32 = 291;
    pub const CLOCK_GETTIME64: u32 = 403;
    pub const CLOCK_NANOSLEEP_TIME64: u32 = 407;
    pub const UTIMENSAT_TIME64: u32 = 412;
    pub const FACCESSAT2: u32 = 439;
}

/// A Linux errno. Guests see these numbers, so they must not come from the host OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Errno(pub u32);

impl Errno {
    pub const EPERM: Self = Self(1);
    pub const ENOENT: Self = Self(2);
    pub const ESRCH: Self = Self(3);
    pub const EIO: Self = Self(5);
    pub const ENXIO: Self = Self(6);
    pub const E2BIG: Self = Self(7);
    pub const ENOEXEC: Self = Self(8);
    pub const EBADF: Self = Self(9);
    pub const ECHILD: Self = Self(10);
    pub const EAGAIN: Self = Self(11);
    pub const ENOMEM: Self = Self(12);
    pub const EACCES: Self = Self(13);
    pub const EFAULT: Self = Self(14);
    pub const EBUSY: Self = Self(16);
    pub const EEXIST: Self = Self(17);
    pub const EXDEV: Self = Self(18);
    pub const ENODEV: Self = Self(19);
    pub const ENOTDIR: Self = Self(20);
    pub const EISDIR: Self = Self(21);
    pub const EINVAL: Self = Self(22);
    pub const EMFILE: Self = Self(24);
    pub const ENOTTY: Self = Self(25);
    pub const EFBIG: Self = Self(27);
    pub const ENOSPC: Self = Self(28);
    pub const ESPIPE: Self = Self(29);
    pub const EROFS: Self = Self(30);
    pub const EMLINK: Self = Self(31);
    pub const EPIPE: Self = Self(32);
    pub const ERANGE: Self = Self(34);
    pub const ENAMETOOLONG: Self = Self(36);
    pub const ENOSYS: Self = Self(38);
    pub const ENOTEMPTY: Self = Self(39);
    pub const ELOOP: Self = Self(40);
    /// Not a real errno: the call cannot finish yet, so the process blocks and
    /// the system call is retried later. Never returned to the guest.
    pub const WOULD_BLOCK: Self = Self(u32::MAX);
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
    /// Map by kind rather than by raw OS error, whose numbers differ between hosts.
    fn from(err: std::io::Error) -> Self {
        use std::io::ErrorKind::*;
        match err.kind() {
            NotFound => Self::ENOENT,
            PermissionDenied => Self::EACCES,
            AlreadyExists => Self::EEXIST,
            NotADirectory => Self::ENOTDIR,
            IsADirectory => Self::EISDIR,
            DirectoryNotEmpty => Self::ENOTEMPTY,
            ReadOnlyFilesystem => Self::EROFS,
            StorageFull => Self::ENOSPC,
            FileTooLarge => Self::EFBIG,
            ResourceBusy => Self::EBUSY,
            CrossesDevices => Self::EXDEV,
            TooManyLinks => Self::EMLINK,
            NotSeekable => Self::ESPIPE,
            BrokenPipe => Self::EPIPE,
            InvalidInput => Self::EINVAL,
            OutOfMemory => Self::ENOMEM,
            _ => Self::EIO,
        }
    }
}

pub const AT_FDCWD: u32 = -100i32 as u32;
pub const AT_SYMLINK_NOFOLLOW: u32 = 0x100;
pub const AT_REMOVEDIR: u32 = 0x200;
pub const AT_EMPTY_PATH: u32 = 0x1000;

pub const O_ACCMODE: u32 = 0o3;
pub const O_RDONLY: u32 = 0o0;
pub const O_WRONLY: u32 = 0o1;
pub const O_CREAT: u32 = 0o100;
pub const O_EXCL: u32 = 0o200;
pub const O_TRUNC: u32 = 0o1000;
pub const O_APPEND: u32 = 0o2000;
pub const O_NONBLOCK: u32 = 0o4000;
pub const O_DIRECTORY: u32 = 0o200000;
pub const O_NOFOLLOW: u32 = 0o400000;
pub const O_CLOEXEC: u32 = 0o2000000;

pub const F_DUPFD: u32 = 0;
pub const F_GETFD: u32 = 1;
pub const F_SETFD: u32 = 2;
pub const F_GETFL: u32 = 3;
pub const F_SETFL: u32 = 4;
pub const F_DUPFD_CLOEXEC: u32 = 1030;
pub const FD_CLOEXEC: u32 = 1;

pub const SEEK_SET: u32 = 0;
pub const SEEK_CUR: u32 = 1;
pub const SEEK_END: u32 = 2;

pub const X_OK: u32 = 1;
pub const RENAME_NOREPLACE: u32 = 1;
pub const TIMER_ABSTIME: u32 = 1;
pub const UTIME_NOW: i64 = (1 << 30) - 1;
pub const UTIME_OMIT: i64 = (1 << 30) - 2;

pub const CLOCK_REALTIME: u32 = 0;
pub const CLOCK_REALTIME_COARSE: u32 = 5;
pub const CLOCK_TAI: u32 = 11;

pub const MAP_FIXED: u32 = 0x10;
pub const MAP_ANONYMOUS: u32 = 0x20;

pub const WNOHANG: u32 = 1;
pub const WEXITED: u32 = 4;
pub const P_PID: u32 = 1;

pub const CLONE_VM: u32 = 0x100;
pub const CLONE_VFORK: u32 = 0x4000;
pub const CLONE_THREAD: u32 = 0x10000;
pub const CLONE_PARENT_SETTID: u32 = 0x100000;

pub const SIGKILL: u32 = 9;
pub const SIGPIPE: u32 = 13;
pub const SIGSTOP: u32 = 19;
pub const SIGSEGV: u32 = 11;
pub const SIG_BLOCK: u32 = 0;
pub const SIG_UNBLOCK: u32 = 1;
pub const SIG_SETMASK: u32 = 2;
pub const SIG_DFL: u32 = 0;
pub const SIG_IGN: u32 = 1;

pub const RLIMIT_STACK: u32 = 3;
pub const RLIMIT_NOFILE: u32 = 7;
pub const RLIM_INFINITY: u64 = u64::MAX;

const S_IFCHR: u32 = 0o020000;
const S_IFIFO: u32 = 0o010000;

pub const DT_UNKNOWN: u8 = 0;
pub const DT_FIFO: u8 = 1;
pub const DT_CHR: u8 = 2;
pub const DT_DIR: u8 = 4;
pub const DT_BLK: u8 = 6;
pub const DT_REG: u8 = 8;
pub const DT_LNK: u8 = 10;
pub const DT_SOCK: u8 = 12;

/// File status, in the form `statx` reports it. Owners are always root: inside the
/// container the guest is root, whoever owns the files on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stat {
    pub mode: u32,
    pub nlink: u32,
    pub ino: u64,
    pub size: u64,
    pub blocks: u64,
    pub blksize: u32,
    pub atime: (i64, u32),
    pub mtime: (i64, u32),
    pub ctime: (i64, u32),
    pub dev: u64,
    pub rdev: u64,
}

const STATX_BASIC_STATS: u32 = 0x7ff;
pub const STATX_SIZE: usize = 256;

impl Stat {
    pub fn from_metadata(meta: &Metadata) -> Self {
        Self {
            mode: meta.mode(),
            nlink: meta.nlink() as u32,
            ino: meta.ino(),
            size: meta.size(),
            blocks: meta.blocks(),
            blksize: meta.blksize() as u32,
            atime: (meta.atime(), meta.atime_nsec() as u32),
            mtime: (meta.mtime(), meta.mtime_nsec() as u32),
            ctime: (meta.ctime(), meta.ctime_nsec() as u32),
            dev: meta.dev(),
            rdev: meta.rdev(),
        }
    }

    /// A device file such as /dev/null, with its major and minor numbers.
    pub fn device(major: u32, minor: u32) -> Self {
        let (major, minor) = (major as u64, minor as u64);
        Self {
            mode: S_IFCHR | 0o666,
            rdev: (minor & 0xff) | ((major & 0xfff) << 8) | ((minor & !0xff) << 12),
            ..Self::char_device()
        }
    }

    /// The terminal-like character device behind stdin, stdout and stderr.
    pub fn char_device() -> Self {
        Self {
            mode: S_IFCHR | 0o620,
            nlink: 1,
            ino: 0,
            size: 0,
            blocks: 0,
            blksize: 1024,
            atime: (0, 0),
            mtime: (0, 0),
            ctime: (0, 0),
            dev: 0,
            rdev: 0,
        }
    }

    /// An anonymous pipe.
    pub fn fifo() -> Self {
        Self {
            mode: S_IFIFO | 0o600,
            blksize: 4096,
            ..Self::char_device()
        }
    }

    /// Encode as `struct statx`.
    pub fn statx(&self) -> [u8; STATX_SIZE] {
        let mut out = [0u8; STATX_SIZE];
        let mut put = |offset: usize, bytes: &[u8]| {
            out[offset..offset + bytes.len()].copy_from_slice(bytes);
        };
        put(0, &STATX_BASIC_STATS.to_le_bytes());
        put(4, &self.blksize.to_le_bytes());
        put(16, &self.nlink.to_le_bytes());
        // uid at 20 and gid at 24 stay 0
        put(28, &(self.mode as u16).to_le_bytes());
        put(32, &self.ino.to_le_bytes());
        put(40, &self.size.to_le_bytes());
        put(48, &self.blocks.to_le_bytes());
        for (offset, (sec, nsec)) in [(64, self.atime), (96, self.ctime), (112, self.mtime)] {
            put(offset, &sec.to_le_bytes());
            put(offset + 8, &nsec.to_le_bytes());
        }
        put(128, &major(self.rdev).to_le_bytes());
        put(132, &minor(self.rdev).to_le_bytes());
        put(136, &major(self.dev).to_le_bytes());
        put(140, &minor(self.dev).to_le_bytes());
        out
    }
}

fn major(dev: u64) -> u32 {
    (((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff)) as u32
}

fn minor(dev: u64) -> u32 {
    ((dev & 0xff) | ((dev >> 12) & !0xff)) as u32
}

/// Encode one `struct linux_dirent64` record, padded to 8 bytes.
pub fn dirent64(ino: u64, next: u64, kind: u8, name: &[u8]) -> Vec<u8> {
    let len = (19 + name.len() + 1).next_multiple_of(8);
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(&ino.to_le_bytes());
    out.extend_from_slice(&next.to_le_bytes());
    out.extend_from_slice(&(len as u16).to_le_bytes());
    out.push(kind);
    out.extend_from_slice(name);
    out.resize(len, 0);
    out
}
