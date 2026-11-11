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
        Ok(Self {
            root,
            inodes: Mutex::new(HashMap::new()),
        })
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

        let mut resolved: Vec<Vec<u8>> = if path.starts_with(b"/") {
            Vec::new()
        } else {
            components(base).map(<[u8]>::to_vec).collect()
        };
        let mut pending: VecDeque<Vec<u8>> = components(path).map(<[u8]>::to_vec).collect();
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
