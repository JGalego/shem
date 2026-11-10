//! File descriptors, and the system calls that work on files and paths.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, DirEntryExt, FileExt, MetadataExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use super::abi::*;
use super::rootfs::{PATH_MAX, Resolved};
use super::{IOV_MAX, Linux, MAX_IO, NS_PER_SEC, Stdio};
use crate::vcpu::Vcpu;

/// Lowest fd number that `open` cannot hand out, as `RLIMIT_NOFILE` reports.
pub const MAX_FDS: u32 = 1024;

/// An open file description. `dup` shares one between several descriptors,
/// along with its offset.
pub struct Description {
    kind: Kind,
    /// Access mode and status flags, as `F_GETFL` reports them.
    flags: u32,
}

enum Kind {
    Input(Box<dyn Read + Send>),
    Output(Box<dyn Write + Send>),
    File(File),
    Dir(Dir),
}

struct Dir {
    guest: Vec<u8>,
    host: PathBuf,
    /// Listed when the directory is opened, sorted by name so that runs are
    /// deterministic whatever order the host filesystem uses.
    entries: Vec<Dirent>,
    next: usize,
}

struct Dirent {
    ino: u64,
    kind: u8,
    name: Vec<u8>,
}

struct Slot {
    description: Arc<Mutex<Description>>,
    cloexec: bool,
}

pub struct FdTable {
    slots: BTreeMap<u32, Slot>,
}

impl FdTable {
    pub fn new(stdio: Stdio) -> Self {
        let mut table = Self {
            slots: BTreeMap::new(),
        };
        for (kind, flags) in [
            (Kind::Input(stdio.stdin), O_RDONLY),
            (Kind::Output(stdio.stdout), O_WRONLY),
            (Kind::Output(stdio.stderr), O_WRONLY),
        ] {
            table
                .install(Description { kind, flags }, 0, false)
                .unwrap();
        }
        table
    }

    fn get(&self, fd: u32) -> Result<Arc<Mutex<Description>>, Errno> {
        self.slots
            .get(&fd)
            .map(|slot| slot.description.clone())
            .ok_or(Errno::EBADF)
    }

    fn install(&mut self, description: Description, min: u32, cloexec: bool) -> Result<u32, Errno> {
        self.share(Arc::new(Mutex::new(description)), min, cloexec)
    }

    /// Put `description` at the lowest free fd that is at least `min`.
    fn share(
        &mut self,
        description: Arc<Mutex<Description>>,
        min: u32,
        cloexec: bool,
    ) -> Result<u32, Errno> {
        let fd = (min..MAX_FDS)
            .find(|fd| !self.slots.contains_key(fd))
            .ok_or(Errno::EMFILE)?;
        self.slots.insert(
            fd,
            Slot {
                description,
                cloexec,
            },
        );
        Ok(fd)
    }
}

impl Description {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Errno> {
        if self.flags & O_ACCMODE == O_WRONLY {
            return Err(Errno::EBADF);
        }
        match &mut self.kind {
            Kind::Input(input) => Ok(input.read(buf)?),
            Kind::File(file) => Ok(file.read(buf)?),
            Kind::Dir(_) => Err(Errno::EISDIR),
            Kind::Output(_) => Err(Errno::EBADF),
        }
    }

    fn write(&mut self, data: &[u8]) -> Result<usize, Errno> {
        if self.flags & O_ACCMODE == O_RDONLY {
            return Err(Errno::EBADF);
        }
        match &mut self.kind {
            Kind::Output(output) => {
                output.write_all(data)?;
                output.flush()?;
            }
            Kind::File(file) => file.write_all(data)?,
            Kind::Dir(_) => return Err(Errno::EISDIR),
            Kind::Input(_) => return Err(Errno::EBADF),
        }
        Ok(data.len())
    }

    fn stat(&self) -> Result<Stat, Errno> {
        match &self.kind {
            Kind::File(file) => Ok(Stat::from_metadata(&file.metadata()?)),
            Kind::Dir(dir) => Ok(Stat::from_metadata(&std::fs::metadata(&dir.host)?)),
            Kind::Input(_) | Kind::Output(_) => Ok(Stat::char_device()),
        }
    }
}

impl Linux {
    fn resolve_at(
        &self,
        vcpu: &dyn Vcpu,
        dirfd: u32,
        path: u32,
        follow: bool,
    ) -> Result<Resolved, Errno> {
        let path = read_cstr(vcpu, path)?;
        let base = self.base_dir(dirfd)?;
        self.root
            .as_ref()
            .ok_or(Errno::ENOENT)?
            .resolve(&base, &path, follow)
    }

    /// The guest directory that relative paths start from: the working directory
    /// for `AT_FDCWD`, otherwise the directory open on `dirfd`.
    fn base_dir(&self, dirfd: u32) -> Result<Vec<u8>, Errno> {
        if dirfd == AT_FDCWD {
            return Ok(self.cwd.clone());
        }
        match &self.files.get(dirfd)?.lock().unwrap().kind {
            Kind::Dir(dir) => Ok(dir.guest.clone()),
            _ => Err(Errno::ENOTDIR),
        }
    }

    pub(super) fn openat(
        &mut self,
        vcpu: &dyn Vcpu,
        dirfd: u32,
        path: u32,
        flags: u32,
        mode: u32,
    ) -> Result<u32, Errno> {
        let resolved = self.resolve_at(vcpu, dirfd, path, flags & O_NOFOLLOW == 0)?;
        let access = flags & O_ACCMODE;
        let creating = flags & O_CREAT != 0;
        let kind = match std::fs::symlink_metadata(&resolved.host) {
            Ok(meta) if meta.is_symlink() => return Err(Errno::ELOOP),
            Ok(_) if creating && flags & O_EXCL != 0 => return Err(Errno::EEXIST),
            Ok(meta) if meta.is_dir() => {
                if access != O_RDONLY || creating {
                    return Err(Errno::EISDIR);
                }
                Kind::Dir(list_dir(resolved)?)
            }
            Ok(_) if flags & O_DIRECTORY != 0 => return Err(Errno::ENOTDIR),
            Err(err) if !creating || err.kind() != std::io::ErrorKind::NotFound => {
                return Err(err.into());
            }
            _ => {
                let mut options = OpenOptions::new();
                options
                    .read(access != O_WRONLY)
                    .write(access != O_RDONLY)
                    .append(flags & O_APPEND != 0)
                    .truncate(flags & O_TRUNC != 0 && access != O_RDONLY)
                    .mode((mode & !self.umask) & 0o7777)
                    .custom_flags(libc::O_NOFOLLOW);
                if creating {
                    // The host only creates files opened for writing; Linux also
                    // creates them for read-only opens.
                    options.create(true).write(true);
                }
                let file = options.open(&resolved.host)?;
                Kind::File(file)
            }
        };
        let description = Description {
            kind,
            flags: flags & (O_ACCMODE | O_APPEND),
        };
        self.files.install(description, 0, flags & O_CLOEXEC != 0)
    }

    pub(super) fn close(&mut self, fd: u32) -> Result<u32, Errno> {
        self.files.slots.remove(&fd).map(|_| 0).ok_or(Errno::EBADF)
    }

    pub(super) fn read(
        &mut self,
        vcpu: &mut dyn Vcpu,
        fd: u32,
        buf: u32,
        count: u32,
    ) -> Result<u32, Errno> {
        let mut data = vec![0; count.min(MAX_IO) as usize];
        let n = self.files.get(fd)?.lock().unwrap().read(&mut data)?;
        vcpu.write_mem(buf, &data[..n])?;
        Ok(n as u32)
    }

    pub(super) fn write(
        &mut self,
        vcpu: &dyn Vcpu,
        fd: u32,
        buf: u32,
        count: u32,
    ) -> Result<u32, Errno> {
        let mut data = vec![0; count.min(MAX_IO) as usize];
        vcpu.read_mem(buf, &mut data)?;
        let n = self.files.get(fd)?.lock().unwrap().write(&data)?;
        Ok(n as u32)
    }

    pub(super) fn readv(
        &mut self,
        vcpu: &mut dyn Vcpu,
        fd: u32,
        iov: u32,
        count: u32,
    ) -> Result<u32, Errno> {
        let mut total = 0u32;
        for (base, len) in iovecs(vcpu, iov, count)? {
            let n = self.read(vcpu, fd, base, len)?;
            total = total.saturating_add(n);
            if n < len {
                break;
            }
        }
        Ok(total)
    }

    pub(super) fn writev(
        &mut self,
        vcpu: &dyn Vcpu,
        fd: u32,
        iov: u32,
        count: u32,
    ) -> Result<u32, Errno> {
        let mut total = 0u32;
        for (base, len) in iovecs(vcpu, iov, count)? {
            let n = self.write(vcpu, fd, base, len)?;
            total = total.saturating_add(n);
            if n < len {
                break;
            }
        }
        Ok(total)
    }

    /// `_llseek`: the 64-bit offset arrives in two halves, and the result goes to memory.
    pub(super) fn llseek(
        &mut self,
        vcpu: &mut dyn Vcpu,
        fd: u32,
        (high, low): (u32, u32),
        result: u32,
        whence: u32,
    ) -> Result<u32, Errno> {
        let offset = ((high as u64) << 32 | low as u64) as i64;
        let position = match &mut self.files.get(fd)?.lock().unwrap().kind {
            Kind::File(file) => file.seek(match whence {
                SEEK_SET => SeekFrom::Start(u64::try_from(offset).map_err(|_| Errno::EINVAL)?),
                SEEK_CUR => SeekFrom::Current(offset),
                SEEK_END => SeekFrom::End(offset),
                _ => return Err(Errno::EINVAL),
            })?,
            // Directory offsets are the `d_off` values getdents64 hands out.
            Kind::Dir(dir) if whence == SEEK_SET && offset >= 0 => {
                dir.next = (offset as usize).min(dir.entries.len());
                offset as u64
            }
            Kind::Dir(_) => return Err(Errno::EINVAL),
            Kind::Input(_) | Kind::Output(_) => return Err(Errno::ESPIPE),
        };
        vcpu.write_mem(result, &position.to_le_bytes())?;
        Ok(0)
    }

    pub(super) fn getdents64(
        &mut self,
        vcpu: &mut dyn Vcpu,
        fd: u32,
        buf: u32,
        count: u32,
    ) -> Result<u32, Errno> {
        let description = self.files.get(fd)?;
        let mut description = description.lock().unwrap();
        let Kind::Dir(dir) = &mut description.kind else {
            return Err(Errno::ENOTDIR);
        };
        let mut out = Vec::new();
        while let Some(entry) = dir.entries.get(dir.next) {
            let record = dirent64(entry.ino, dir.next as u64 + 1, entry.kind, &entry.name);
            if out.len() + record.len() > count.min(MAX_IO) as usize {
                if out.is_empty() {
                    return Err(Errno::EINVAL);
                }
                break;
            }
            out.extend(record);
            dir.next += 1;
        }
        vcpu.write_mem(buf, &out)?;
        Ok(out.len() as u32)
    }

    pub(super) fn statx(
        &mut self,
        vcpu: &mut dyn Vcpu,
        dirfd: u32,
        path: u32,
        flags: u32,
        buf: u32,
    ) -> Result<u32, Errno> {
        let name = read_cstr(vcpu, path)?;
        let stat = if name.is_empty() && flags & AT_EMPTY_PATH != 0 {
            if dirfd == AT_FDCWD {
                self.stat_path(&self.cwd.clone(), true)?
            } else {
                self.files.get(dirfd)?.lock().unwrap().stat()?
            }
        } else {
            let resolved = self.resolve_at(vcpu, dirfd, path, flags & AT_SYMLINK_NOFOLLOW == 0)?;
            stat_host(&resolved, flags & AT_SYMLINK_NOFOLLOW == 0)?
        };
        vcpu.write_mem(buf, &stat.statx())?;
        Ok(0)
    }

    fn stat_path(&self, guest: &[u8], follow: bool) -> Result<Stat, Errno> {
        let root = self.root.as_ref().ok_or(Errno::ENOENT)?;
        stat_host(&root.resolve(b"/", guest, follow)?, follow)
    }

    pub(super) fn faccessat(
        &mut self,
        vcpu: &dyn Vcpu,
        dirfd: u32,
        path: u32,
        mode: u32,
    ) -> Result<u32, Errno> {
        let stat = stat_host(&self.resolve_at(vcpu, dirfd, path, true)?, true)?;
        // The guest is root, so only execute permission can be missing.
        let is_dir = stat.mode & 0o170000 == 0o040000;
        if mode & X_OK != 0 && !is_dir && stat.mode & 0o111 == 0 {
            return Err(Errno::EACCES);
        }
        Ok(0)
    }

    pub(super) fn getcwd(&self, vcpu: &mut dyn Vcpu, buf: u32, size: u32) -> Result<u32, Errno> {
        let cwd = [self.cwd.as_slice(), &[0]].concat();
        if (size as usize) < cwd.len() {
            return Err(Errno::ERANGE);
        }
        vcpu.write_mem(buf, &cwd)?;
        Ok(cwd.len() as u32)
    }

    pub(super) fn chdir(&mut self, vcpu: &dyn Vcpu, path: u32) -> Result<u32, Errno> {
        let resolved = self.resolve_at(vcpu, AT_FDCWD, path, true)?;
        if !std::fs::metadata(&resolved.host)?.is_dir() {
            return Err(Errno::ENOTDIR);
        }
        self.cwd = resolved.guest;
        Ok(0)
    }

    pub(super) fn mkdirat(
        &mut self,
        vcpu: &dyn Vcpu,
        dirfd: u32,
        path: u32,
        mode: u32,
    ) -> Result<u32, Errno> {
        let resolved = self.resolve_at(vcpu, dirfd, path, false)?;
        std::fs::DirBuilder::new()
            .mode((mode & !self.umask) & 0o7777)
            .create(&resolved.host)?;
        Ok(0)
    }

    pub(super) fn unlinkat(
        &mut self,
        vcpu: &dyn Vcpu,
        dirfd: u32,
        path: u32,
        flags: u32,
    ) -> Result<u32, Errno> {
        let resolved = self.resolve_at(vcpu, dirfd, path, false)?;
        if resolved.guest == b"/" {
            return Err(Errno::EBUSY);
        }
        let meta = std::fs::symlink_metadata(&resolved.host)?;
        if flags & AT_REMOVEDIR != 0 {
            if !meta.is_dir() {
                return Err(Errno::ENOTDIR);
            }
            std::fs::remove_dir(&resolved.host)?;
        } else {
            if meta.is_dir() {
                return Err(Errno::EISDIR);
            }
            std::fs::remove_file(&resolved.host)?;
        }
        Ok(0)
    }

    pub(super) fn symlinkat(
        &mut self,
        vcpu: &dyn Vcpu,
        target: u32,
        dirfd: u32,
        path: u32,
    ) -> Result<u32, Errno> {
        let target = read_cstr(vcpu, target)?;
        if target.is_empty() {
            return Err(Errno::ENOENT);
        }
        let resolved = self.resolve_at(vcpu, dirfd, path, false)?;
        // The target is stored verbatim; the resolver interprets it inside the root.
        std::os::unix::fs::symlink(std::ffi::OsStr::from_bytes(&target), &resolved.host)?;
        Ok(0)
    }

    pub(super) fn readlinkat(
        &mut self,
        vcpu: &mut dyn Vcpu,
        dirfd: u32,
        path: u32,
        buf: u32,
        size: u32,
    ) -> Result<u32, Errno> {
        let resolved = self.resolve_at(vcpu, dirfd, path, false)?;
        if !std::fs::symlink_metadata(&resolved.host)?.is_symlink() {
            return Err(Errno::EINVAL);
        }
        let target = std::fs::read_link(&resolved.host)?;
        let target = target.as_os_str().as_bytes();
        let n = target.len().min(size as usize);
        vcpu.write_mem(buf, &target[..n])?;
        Ok(n as u32)
    }

    pub(super) fn renameat2(
        &mut self,
        vcpu: &dyn Vcpu,
        (old_dirfd, old_path): (u32, u32),
        (new_dirfd, new_path): (u32, u32),
        flags: u32,
    ) -> Result<u32, Errno> {
        if flags & !RENAME_NOREPLACE != 0 {
            return Err(Errno::EINVAL);
        }
        let from = self.resolve_at(vcpu, old_dirfd, old_path, false)?;
        let to = self.resolve_at(vcpu, new_dirfd, new_path, false)?;
        if flags & RENAME_NOREPLACE != 0 && std::fs::symlink_metadata(&to.host).is_ok() {
            return Err(Errno::EEXIST);
        }
        std::fs::rename(&from.host, &to.host)?;
        Ok(0)
    }

    /// Copy from one descriptor to another, as a `read` followed by a `write`.
    /// With an offset pointer, read at that offset and leave the file position alone.
    pub(super) fn sendfile(
        &mut self,
        vcpu: &mut dyn Vcpu,
        out_fd: u32,
        in_fd: u32,
        offset: u32,
        count: u32,
    ) -> Result<u32, Errno> {
        let mut data = vec![0; count.min(MAX_IO) as usize];
        let input = self.files.get(in_fd)?;
        let n = if offset == 0 {
            input.lock().unwrap().read(&mut data)?
        } else {
            let mut position = [0u8; 8];
            vcpu.read_mem(offset, &mut position)?;
            let position = u64::from_le_bytes(position);
            let input = input.lock().unwrap();
            let Kind::File(file) = &input.kind else {
                return Err(Errno::ESPIPE);
            };
            if input.flags & O_ACCMODE == O_WRONLY {
                return Err(Errno::EBADF);
            }
            let n = file.read_at(&mut data, position)?;
            vcpu.write_mem(offset, &(position + n as u64).to_le_bytes())?;
            n
        };
        let n = self.files.get(out_fd)?.lock().unwrap().write(&data[..n])?;
        Ok(n as u32)
    }

    /// Set access and modification times. `UTIME_NOW` means the virtual clock's
    /// current time. Symlinks are always followed.
    pub(super) fn utimensat(
        &mut self,
        vcpu: &mut dyn Vcpu,
        dirfd: u32,
        path: u32,
        times: u32,
        flags: u32,
        retired: u64,
    ) -> Result<u32, Errno> {
        let now = self.now_ns(CLOCK_REALTIME, retired)?;
        let mut stamps = [Some(now); 2];
        if times != 0 {
            let mut raw = [0u8; 32];
            vcpu.read_mem(times, &mut raw)?;
            for (i, stamp) in stamps.iter_mut().enumerate() {
                let field = |at: usize| i64::from_le_bytes(raw[at..at + 8].try_into().unwrap());
                let (sec, nsec) = (field(16 * i), field(16 * i + 8));
                *stamp = match nsec {
                    UTIME_NOW => Some(now),
                    UTIME_OMIT => None,
                    0..1_000_000_000 if sec >= 0 => Some(sec as u64 * NS_PER_SEC + nsec as u64),
                    _ => return Err(Errno::EINVAL),
                };
            }
        }
        let file = if path == 0 {
            match &self.files.get(dirfd)?.lock().unwrap().kind {
                Kind::File(file) => file.try_clone()?,
                Kind::Dir(dir) => File::open(&dir.host)?,
                Kind::Input(_) | Kind::Output(_) => return Ok(0),
            }
        } else {
            let follow = flags & AT_SYMLINK_NOFOLLOW == 0;
            File::open(self.resolve_at(vcpu, dirfd, path, follow)?.host)?
        };
        let at = |ns: u64| SystemTime::UNIX_EPOCH + Duration::from_nanos(ns);
        let mut file_times = std::fs::FileTimes::new();
        if let Some(ns) = stamps[0] {
            file_times = file_times.set_accessed(at(ns));
        }
        if let Some(ns) = stamps[1] {
            file_times = file_times.set_modified(at(ns));
        }
        file.set_times(file_times)?;
        Ok(0)
    }

    pub(super) fn dup(&mut self, fd: u32, min: u32, cloexec: bool) -> Result<u32, Errno> {
        let description = self.files.get(fd)?;
        self.files.share(description, min, cloexec)
    }

    pub(super) fn dup3(&mut self, old: u32, new: u32, flags: u32) -> Result<u32, Errno> {
        if old == new || flags & !O_CLOEXEC != 0 || new >= MAX_FDS {
            return Err(Errno::EINVAL);
        }
        let description = self.files.get(old)?;
        self.files.slots.remove(&new);
        self.files.share(description, new, flags & O_CLOEXEC != 0)
    }

    pub(super) fn fcntl(&mut self, fd: u32, cmd: u32, arg: u32) -> Result<u32, Errno> {
        let slot = self.files.slots.get_mut(&fd).ok_or(Errno::EBADF)?;
        match cmd {
            F_DUPFD => self.dup(fd, arg, false),
            F_DUPFD_CLOEXEC => self.dup(fd, arg, true),
            F_GETFD => Ok(slot.cloexec as u32),
            F_SETFD => {
                slot.cloexec = arg & FD_CLOEXEC != 0;
                Ok(0)
            }
            F_GETFL => Ok(slot.description.lock().unwrap().flags),
            F_SETFL => Ok(0),
            _ => Err(Errno::EINVAL),
        }
    }

    pub(super) fn fsync(&mut self, fd: u32) -> Result<u32, Errno> {
        if let Kind::File(file) = &self.files.get(fd)?.lock().unwrap().kind {
            file.sync_all()?;
        }
        Ok(0)
    }
}

fn list_dir(resolved: Resolved) -> Result<Dir, Errno> {
    let ino = |path: &std::path::Path| std::fs::metadata(path).map(|meta| meta.ino());
    let own = ino(&resolved.host)?;
    let parent = if resolved.guest == b"/" {
        own
    } else {
        ino(resolved.host.parent().unwrap())?
    };
    let mut entries = vec![
        Dirent {
            ino: own,
            kind: DT_DIR,
            name: b".".to_vec(),
        },
        Dirent {
            ino: parent,
            kind: DT_DIR,
            name: b"..".to_vec(),
        },
    ];
    let mut listed = Vec::new();
    for entry in std::fs::read_dir(&resolved.host)? {
        let entry = entry?;
        listed.push(Dirent {
            ino: entry.ino(),
            kind: dirent_type(entry.file_type()?),
            name: entry.file_name().as_bytes().to_vec(),
        });
    }
    listed.sort_by(|a, b| a.name.cmp(&b.name));
    entries.extend(listed);
    Ok(Dir {
        guest: resolved.guest,
        host: resolved.host,
        entries,
        next: 0,
    })
}

fn dirent_type(kind: std::fs::FileType) -> u8 {
    use std::os::unix::fs::FileTypeExt;
    if kind.is_dir() {
        DT_DIR
    } else if kind.is_file() {
        DT_REG
    } else if kind.is_symlink() {
        DT_LNK
    } else if kind.is_char_device() {
        DT_CHR
    } else if kind.is_block_device() {
        DT_BLK
    } else if kind.is_fifo() {
        DT_FIFO
    } else if kind.is_socket() {
        DT_SOCK
    } else {
        DT_UNKNOWN
    }
}

fn stat_host(resolved: &Resolved, follow: bool) -> Result<Stat, Errno> {
    let meta = if follow {
        std::fs::metadata(&resolved.host)?
    } else {
        std::fs::symlink_metadata(&resolved.host)?
    };
    Ok(Stat::from_metadata(&meta))
}

fn iovecs(vcpu: &dyn Vcpu, iov: u32, count: u32) -> Result<Vec<(u32, u32)>, Errno> {
    if count > IOV_MAX {
        return Err(Errno::EINVAL);
    }
    (0..count)
        .map(|i| {
            let mut entry = [0; 8];
            vcpu.read_mem(iov.checked_add(8 * i).ok_or(Errno::EFAULT)?, &mut entry)?;
            let base = u32::from_le_bytes(entry[..4].try_into().unwrap());
            let len = u32::from_le_bytes(entry[4..].try_into().unwrap());
            Ok((base, len))
        })
        .collect()
}

/// Read a NUL-terminated string from guest memory, such as a path.
pub fn read_cstr(vcpu: &dyn Vcpu, addr: u32) -> Result<Vec<u8>, Errno> {
    let mut out = Vec::new();
    let mut chunk = [0u8; 256];
    let mut at = addr;
    loop {
        let room = (vcpu.mem_size().saturating_sub(at) as usize).min(chunk.len());
        if room == 0 {
            return Err(Errno::EFAULT);
        }
        vcpu.read_mem(at, &mut chunk[..room])?;
        if let Some(nul) = chunk[..room].iter().position(|&b| b == 0) {
            out.extend_from_slice(&chunk[..nul]);
            return Ok(out);
        }
        out.extend_from_slice(&chunk[..room]);
        if out.len() >= PATH_MAX {
            return Err(Errno::ENAMETOOLONG);
        }
        at += room as u32;
    }
}
