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
//! Known gap: resolving a path and then opening it are separate host calls. A
//! concurrent writer to the host directory could swap a checked component for a
//! symlink in between. A single guest cannot do this, because the host services
//! its system calls one at a time. On Linux, `openat2` with `RESOLVE_IN_ROOT`
//! would close the gap.

use std::collections::{HashMap, VecDeque};
use std::ffi::OsStr;
use std::io::ErrorKind;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::abi::Errno;

/// Linux's limit on symlinks followed in one resolution.
const MAX_SYMLINKS: usize = 40;
pub const PATH_MAX: usize = 4096;
const NAME_MAX: usize = 255;

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
    pub fn at_parent<T>(
        &self,
        guest: &[u8],
        change: impl FnOnce(std::os::fd::RawFd, &std::ffi::CStr) -> std::io::Result<T>,
    ) -> Option<std::io::Result<T>> {
        use std::os::fd::AsRawFd;
        let cut = guest.iter().rposition(|&b| b == b'/')?;
        let (parent, name) = (&guest[..cut.max(1)], &guest[cut + 1..]);
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
