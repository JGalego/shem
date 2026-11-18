//! The guest's root filesystem: a host directory, and path resolution that keeps
//! every guest path inside it.
//!
//! Resolution has chroot semantics. `..` stops at the root, and symlinks,
//! absolute ones included, resolve relative to the root, so a rootfs whose
//! `/bin/ls` links to `/bin/busybox` works as it would in a container. Each
//! component is inspected on the host before the next one is followed, and
//! symlinks are expanded here rather than by the host, so `..` is applied to
//! real directories, as Linux does.
//!
//! Resolving a path and then acting on it are separate host calls, so a
//! concurrent writer to the host directory could swap a checked component for
//! a symlink in between. A single guest cannot do this, because the host
//! services its system calls one at a time. On Linux the acting calls close
//! the gap: opens, stats, link reads, time changes, listings and the changes
//! to a directory (`mkdir`, `unlink`, `symlink`, `rename`) go through
//! `openat2` with `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`, which fails
//! rather than follow a link. Elsewhere, and before Linux 5.6, they act on
//! the host path as checked.

use crate::sys::ffi::OsStrExt;
use std::collections::{HashMap, VecDeque};
use std::ffi::OsStr;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::abi::{Errno, Stat};

/// Linux's limit on symlinks followed in one resolution.
const MAX_SYMLINKS: usize = 40;
pub const PATH_MAX: usize = 4096;
const NAME_MAX: usize = 255;

/// The names in a directory, each with its host inode and file type.
pub type Listing = Vec<(Vec<u8>, u64, std::fs::FileType)>;

pub struct RootFs {
    root: PathBuf,
    /// The root, opened, for `open_beneath`
    #[cfg(target_os = "linux")]
    dir: std::fs::File,
    /// Guest inode numbers by host device and inode, handed out in the order
    /// files are first seen, so they do not depend on the host filesystem.
    inodes: Mutex<HashMap<(u64, u64), u64>>,
}

/// A resolved path: where it lives on the host, and its canonical guest path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub host: PathBuf,
    pub guest: Vec<u8>,
}

impl RootFs {
    /// Use the host directory `root` as the guest's `/`.
    pub fn new(root: &Path) -> std::io::Result<Self> {
        let root = std::fs::canonicalize(root)?;
        if !root.is_dir() {
            return Err(ErrorKind::NotADirectory.into());
        }
        #[cfg(target_os = "linux")]
        let dir = std::fs::File::open(&root)?;
        Ok(Self {
            #[cfg(target_os = "linux")]
            dir,
            root,
            inodes: Mutex::new(HashMap::new()),
        })
    }

    /// Open the file at the canonical guest path `guest` (as `resolve` made
    /// it, with no links in it), in one step the kernel keeps under the root:
    /// no component may be a symlink, and none may lead out, even if another
    /// process on the host swaps one for a link between `resolve` and this.
    /// `flags` and `mode` are those of `open(2)`. None where the kernel has
    /// no `openat2` (Linux before 5.6, or another system), when the caller
    /// opens the host path as it always did.
    pub fn open_beneath(
        &self,
        guest: &[u8],
        flags: i32,
        mode: u32,
    ) -> Option<std::io::Result<std::fs::File>> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::{AsRawFd, FromRawFd};
            use std::sync::atomic::{AtomicBool, Ordering};
            static MISSING: AtomicBool = AtomicBool::new(false);
            if MISSING.load(Ordering::Relaxed) {
                return None;
            }
            #[repr(C)]
            struct OpenHow {
                flags: u64,
                mode: u64,
                resolve: u64,
            }
            const RESOLVE_NO_MAGICLINKS: u64 = 0x02;
            const RESOLVE_NO_SYMLINKS: u64 = 0x04;
            const RESOLVE_BENEATH: u64 = 0x08;
            let relative = guest.strip_prefix(b"/").unwrap_or(guest);
            let relative = if relative.is_empty() { b"." } else { relative };
            let path = std::ffi::CString::new(relative).ok()?;
            let how = OpenHow {
                flags: (flags | libc::O_CLOEXEC) as u64,
                mode: u64::from(mode),
                resolve: RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS | RESOLVE_BENEATH,
            };
            // SAFETY: the descriptor, the path and the struct are valid for the call
            let fd = unsafe {
                libc::syscall(
                    libc::SYS_openat2,
                    self.dir.as_raw_fd(),
                    path.as_ptr(),
                    &how,
                    std::mem::size_of::<OpenHow>(),
                )
            };
            if fd >= 0 {
                // SAFETY: a new descriptor that nothing else owns
                return Some(Ok(unsafe { std::fs::File::from_raw_fd(fd as i32) }));
            }
            let err = std::io::Error::last_os_error();
            match err.raw_os_error() {
                Some(libc::ENOSYS) => {
                    MISSING.store(true, Ordering::Relaxed);
                    None
                }
                _ => Some(Err(err)),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (guest, flags, mode);
            None
        }
    }

    /// The guest inode number of the host file `ino` on the host device `dev`.
    pub fn inode(&self, dev: u64, ino: u64) -> u64 {
        let mut inodes = self.inodes.lock().unwrap();
        let next = inodes.len() as u64 + 1;
        *inodes.entry((dev, ino)).or_insert(next)
    }

    /// Resolve `path` against `base`, an absolute guest directory. The last
    /// component may be missing, so it can be created; every earlier one must be
    /// a directory. With `follow` false, a symlink in the last component is
    /// returned as the link itself.
    pub fn resolve(&self, base: &[u8], path: &[u8], follow: bool) -> Result<Resolved, Errno> {
        if path.is_empty() {
            return Err(Errno::ENOENT);
        }
        if path.len() >= PATH_MAX {
            return Err(Errno::ENAMETOOLONG);
        }
        // A trailing slash means the path must name a directory, through any symlink.
        let must_be_dir = path.ends_with(b"/");
        let follow = follow || must_be_dir;

        // A relative path walks its base from the root like the rest: the
        // guest may have swapped its working directory, or a directory it
        // holds open, for a symlink since it got there, and the host must
        // never follow one.
        let mut resolved: Vec<Vec<u8>> = Vec::new();
        let mut pending: VecDeque<Vec<u8>> = VecDeque::new();
        if !path.starts_with(b"/") {
            pending.extend(components(base).map(<[u8]>::to_vec));
        }
        pending.extend(components(path).map(<[u8]>::to_vec));
        let mut links = 0;

        while let Some(name) = pending.pop_front() {
            match name.as_slice() {
                b"." => continue,
                b".." => {
                    resolved.pop();
                    continue;
                }
                _ if name.len() > NAME_MAX => return Err(Errno::ENAMETOOLONG),
                // Windows takes these as more than a name: a separator, a drive, a stream
                #[cfg(windows)]
                _ if name.iter().any(|b| {
                    matches!(b, b'\\' | b':' | b'*' | b'?' | b'"' | b'<' | b'>' | b'|')
                }) =>
                {
                    return Err(Errno::EINVAL);
                }
                _ => {}
            }
            let last = pending.is_empty();
            let host = self.host_path(&resolved).join(OsStr::from_bytes(&name));
            match std::fs::symlink_metadata(&host) {
                Ok(meta) if meta.is_symlink() && (follow || !last) => {
                    links += 1;
                    if links > MAX_SYMLINKS {
                        return Err(Errno::ELOOP);
                    }
                    let target = std::fs::read_link(&host)?;
                    let target = target.as_os_str().as_bytes();
                    if target.starts_with(b"/") {
                        resolved.clear();
                    }
                    for component in components(target).rev() {
                        pending.push_front(component.to_vec());
                    }
                }
                Ok(meta) if !last && !meta.is_dir() => return Err(Errno::ENOTDIR),
                Ok(meta) if last && must_be_dir && !meta.is_dir() => {
                    return Err(Errno::ENOTDIR);
                }
                Ok(_) => resolved.push(name),
                Err(err) if err.kind() == ErrorKind::NotFound && last => resolved.push(name),
                Err(err) => return Err(err.into()),
            }
        }

        let mut guest = Vec::new();
        for name in &resolved {
            guest.push(b'/');
            guest.extend_from_slice(name);
        }
        if guest.is_empty() {
            guest.push(b'/');
        }
        Ok(Resolved {
            host: self.host_path(&resolved),
            guest,
        })
    }

    /// Run `change` on the directory that holds the canonical guest path
    /// `guest` and the name it has there, with the directory opened beneath the
    /// root as `open_beneath` opens a file: a `mkdirat` or `unlinkat` on it
    /// cannot be sent through a link a host process swapped in since the
    /// path was resolved. None where `open_beneath` has no `openat2`.
    #[cfg(unix)]
    fn at_parent<T>(
        &self,
        guest: &[u8],
        change: impl FnOnce(std::os::fd::RawFd, &std::ffi::CStr) -> std::io::Result<T>,
    ) -> Option<std::io::Result<T>> {
        use std::os::fd::AsRawFd;
        let cut = guest.iter().rposition(|&b| b == b'/')?;
        let (parent, name) = (&guest[..cut.max(1)], &guest[cut + 1..]);
        if name.is_empty() {
            return None; // the root itself, which has no parent here: by path, as it was
        }
        let name = std::ffi::CString::new(name).ok()?;
        let dir = match self.open_beneath(
            parent,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW,
            0,
        )? {
            Ok(dir) => dir,
            Err(err) => return Some(Err(err)),
        };
        Some(change(dir.as_raw_fd(), &name))
    }

    /// The metadata of the canonical guest path `guest`, itself and not what a
    /// link there leads to, read through its parent opened beneath the root.
    /// None as `at_parent` is.
    #[allow(clippy::unnecessary_cast)] // the platform's field types differ
    pub fn stat_beneath(&self, guest: &[u8]) -> Option<std::io::Result<Stat>> {
        #[cfg(unix)]
        {
            self.at_parent(guest, |dir, name| {
                // SAFETY: a zeroed stat that fstatat fills in, and a NUL-terminated name
                let mut stat: libc::stat = unsafe { std::mem::zeroed() };
                let status = unsafe {
                    libc::fstatat(dir, name.as_ptr(), &mut stat, libc::AT_SYMLINK_NOFOLLOW)
                };
                if status == 0 {
                    Ok(Stat::from_stat(
                        &stat,
                        self.inode(stat.st_dev as u64, stat.st_ino as u64),
                    ))
                } else {
                    Err(std::io::Error::last_os_error())
                }
            })
        }
        #[cfg(not(unix))]
        {
            let _ = guest;
            None
        }
    }

    /// Where the symlink at the canonical guest path `guest` points, read
    /// through its parent opened beneath the root. None as `at_parent` is.
    pub fn readlink_beneath(&self, guest: &[u8]) -> Option<std::io::Result<Vec<u8>>> {
        #[cfg(unix)]
        {
            self.at_parent(guest, |dir, name| {
                let mut buf = vec![0u8; PATH_MAX];
                // SAFETY: a buffer of the length given, and a NUL-terminated name
                let n = unsafe {
                    libc::readlinkat(dir, name.as_ptr(), buf.as_mut_ptr().cast(), buf.len())
                };
                if n < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                buf.truncate(n as usize);
                Ok(buf)
            })
        }
        #[cfg(not(unix))]
        {
            let _ = guest;
            None
        }
    }

    /// Set the times of the canonical guest path `guest` (a link's own, not
    /// what it leads to) through its parent opened beneath the root, in
    /// nanoseconds since the epoch, leaving out those that are `None`. None
    /// as `at_parent` is.
    pub fn set_times_beneath(
        &self,
        guest: &[u8],
        access: Option<u64>,
        modify: Option<u64>,
    ) -> Option<std::io::Result<()>> {
        #[cfg(unix)]
        {
            let times = timespecs(access, modify);
            self.at_parent(guest, |dir, name| {
                // SAFETY: a NUL-terminated name and two timespecs, as utimensat expects
                let status = unsafe {
                    libc::utimensat(
                        dir,
                        name.as_ptr(),
                        times.as_ptr(),
                        libc::AT_SYMLINK_NOFOLLOW,
                    )
                };
                if status == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            })
        }
        #[cfg(not(unix))]
        {
            let _ = (guest, access, modify);
            None
        }
    }

    /// Make the directory at the canonical guest path `guest`, beneath the
    /// root. None as `at_parent` is.
    pub fn mkdir_beneath(&self, guest: &[u8], mode: u32) -> Option<std::io::Result<()>> {
        #[cfg(unix)]
        {
            self.at_parent(guest, |dir, name| {
                // SAFETY: `dir` is open and `name` is a C string, for the call
                match unsafe { libc::mkdirat(dir, name.as_ptr(), mode as libc::mode_t) } {
                    0 => Ok(()),
                    _ => Err(std::io::Error::last_os_error()),
                }
            })
        }
        #[cfg(not(unix))]
        {
            let _ = (guest, mode);
            None
        }
    }

    /// Delete the file, or with `directory` the directory, at the canonical
    /// guest path `guest`, beneath the root: the kernel answers EISDIR and
    /// ENOTDIR as Linux does for the wrong kind. None as `at_parent` is.
    pub fn unlink_beneath(&self, guest: &[u8], directory: bool) -> Option<std::io::Result<()>> {
        #[cfg(unix)]
        {
            let how = if directory { libc::AT_REMOVEDIR } else { 0 };
            self.at_parent(guest, |dir, name| {
                // SAFETY: `dir` is open and `name` is a C string, for the call
                match unsafe { libc::unlinkat(dir, name.as_ptr(), how) } {
                    0 => Ok(()),
                    _ => Err(std::io::Error::last_os_error()),
                }
            })
        }
        #[cfg(not(unix))]
        {
            let _ = (guest, directory);
            None
        }
    }

    /// Make a symlink to `target` at the canonical guest path `guest`,
    /// beneath the root. None as `at_parent` is.
    pub fn symlink_beneath(&self, guest: &[u8], target: &[u8]) -> Option<std::io::Result<()>> {
        #[cfg(unix)]
        {
            let link = std::ffi::CString::new(target).ok()?;
            self.at_parent(guest, |dir, name| {
                // SAFETY: `dir` is open, and `link` and `name` are C strings, for the call
                match unsafe { libc::symlinkat(link.as_ptr(), dir, name.as_ptr()) } {
                    0 => Ok(()),
                    _ => Err(std::io::Error::last_os_error()),
                }
            })
        }
        #[cfg(not(unix))]
        {
            let _ = (guest, target);
            None
        }
    }

    /// Rename `from` to `to`, canonical guest paths, with `flags` of
    /// `renameat2`, beneath the root. None as `at_parent` is, and where the
    /// kernel has no `renameat2`, which the caller does by paths.
    pub fn rename_beneath(
        &self,
        from: &[u8],
        to: &[u8],
        flags: u32,
    ) -> Option<std::io::Result<()>> {
        #[cfg(target_os = "linux")]
        {
            self.at_parent(from, |from_dir, from_name| {
                self.at_parent(to, |to_dir, to_name| {
                    // SAFETY: both descriptors are open and both names are C strings, for the call
                    let done = unsafe {
                        libc::syscall(
                            libc::SYS_renameat2,
                            from_dir,
                            from_name.as_ptr(),
                            to_dir,
                            to_name.as_ptr(),
                            flags,
                        )
                    };
                    if done == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                })
                .unwrap_or_else(|| Err(std::io::Error::from_raw_os_error(libc::ENOSYS)))
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (from, to, flags);
            None
        }
    }

    /// The names in the directory at the canonical guest path `guest`, read
    /// from the directory opened beneath the root, with each one's inode and
    /// file type. None where `open_beneath` is, or the host has no way to
    /// list a descriptor.
    #[cfg(target_os = "linux")]
    pub fn list_beneath(&self, guest: &[u8]) -> Option<std::io::Result<Listing>> {
        use std::os::fd::AsRawFd;
        let dir = match self.open_beneath(guest, libc::O_RDONLY | libc::O_DIRECTORY, 0)? {
            Ok(dir) => dir,
            Err(err) => return Some(Err(err)),
        };
        let through = format!("/proc/self/fd/{}", dir.as_raw_fd());
        let entries = match std::fs::read_dir(&through) {
            Ok(entries) => entries,
            Err(err) if err.kind() == ErrorKind::NotFound => return None,
            Err(err) => return Some(Err(err)),
        };
        let mut listed = Vec::new();
        for entry in entries {
            let result = entry.and_then(|entry| {
                use crate::sys::fs::DirEntryExt;
                Ok((
                    entry.file_name().as_bytes().to_vec(),
                    entry.ino(),
                    entry.file_type()?,
                ))
            });
            match result {
                Ok(item) => listed.push(item),
                Err(err) => return Some(Err(err)),
            }
        }
        Some(Ok(listed))
    }

    fn host_path(&self, components: &[Vec<u8>]) -> PathBuf {
        let mut host = self.root.clone();
        for name in components {
            host.push(OsStr::from_bytes(name));
        }
        host
    }
}

fn components(path: &[u8]) -> impl DoubleEndedIterator<Item = &[u8]> {
    path.split(|&b| b == b'/').filter(|c| !c.is_empty())
}

/// An access and a modification time, in nanoseconds since the epoch, as
/// `utimensat` takes them: `None` leaves the time as it is.
#[cfg(unix)]
pub(super) fn timespecs(access: Option<u64>, modify: Option<u64>) -> [libc::timespec; 2] {
    const NS_PER_SEC: u64 = 1_000_000_000;
    let timespec = |ns: Option<u64>| match ns {
        Some(ns) => libc::timespec {
            tv_sec: (ns / NS_PER_SEC) as libc::time_t,
            tv_nsec: (ns % NS_PER_SEC) as libc::c_long,
        },
        None => libc::timespec {
            tv_sec: 0,
            tv_nsec: libc::UTIME_OMIT,
        },
    };
    [timespec(access), timespec(modify)]
}
