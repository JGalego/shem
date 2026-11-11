//! File descriptors, and the system calls that work on files and paths.

use std::collections::{BTreeMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, DirEntryExt, FileExt, MetadataExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use super::abi::*;
use super::rootfs::{PATH_MAX, Resolved};
use super::{Exec, IOV_MAX, MAX_IO, NS_PER_SEC, Process, Stdio};
use crate::elf;
use crate::vcpu::Vcpu;

/// Lowest fd number that `open` cannot hand out, as `RLIMIT_NOFILE` reports.
pub const MAX_FDS: u32 = 1024;
/// Bytes a pipe holds before writers block, as on Linux.
const PIPE_CAPACITY: usize = 64 << 10;
/// Largest program `execve` loads.
const MAX_EXECUTABLE: u64 = 64 << 20;
/// Most argument and environment bytes `execve` accepts.
const ARG_MAX: usize = 256 << 10;
const MAX_ARG_STRLEN: usize = 128 << 10;

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
    Pipe(PipeEnd),
    Device(Device),
}

/// The device files every container has under /dev, whatever its rootfs holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    Null,
    Zero,
    Full,
    /// /dev/random and /dev/urandom: a seeded PRNG, so runs stay repeatable.
    Random(u64),
}

impl Device {
    /// The device a guest path names, decided on the path's text so that /dev
    /// need not exist in the rootfs. Standard streams name a descriptor to share.
    fn named(path: &[u8]) -> Option<Result<Self, u32>> {
        Some(match path {
            b"/dev/null" => Ok(Self::Null),
            b"/dev/zero" => Ok(Self::Zero),
            b"/dev/full" => Ok(Self::Full),
            b"/dev/random" | b"/dev/urandom" => Ok(Self::Random(0x5348_454d)),
            b"/dev/stdin" => Err(0),
            b"/dev/stdout" => Err(1),
            b"/dev/stderr" => Err(2),
            _ => return None,
        })
    }

    fn stat(self) -> Stat {
        match self {
            Self::Null => Stat::device(1, 3),
            Self::Zero => Stat::device(1, 5),
            Self::Full => Stat::device(1, 7),
            Self::Random(_) => Stat::device(1, 9),
        }
    }
}

struct Pipe {
    buffer: VecDeque<u8>,
    readers: usize,
    writers: usize,
}

/// One end of a pipe. Dropping the last description that holds an end closes
/// it, which is how readers see end-of-file and writers see EPIPE.
struct PipeEnd {
    pipe: Arc<Mutex<Pipe>>,
    writer: bool,
}

impl Drop for PipeEnd {
    fn drop(&mut self) {
        let mut pipe = self.pipe.lock().unwrap();
        if self.writer {
            pipe.writers -= 1;
        } else {
            pipe.readers -= 1;
        }
    }
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

    /// The table a forked child starts with: the same descriptions, shared.
    pub fn fork(&self) -> Self {
        let slots = self.slots.iter().map(|(&fd, slot)| {
            let description = slot.description.clone();
            (
                fd,
                Slot {
                    description,
                    cloexec: slot.cloexec,
                },
            )
        });
        Self {
            slots: slots.collect(),
        }
    }

    /// Close the descriptors marked close-on-exec.
    pub fn close_on_exec(&mut self) {
        self.slots.retain(|_, slot| !slot.cloexec);
    }

    /// Close every descriptor, as when a process exits.
    pub fn close_all(&mut self) {
        self.slots.clear();
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
        let nonblock = self.flags & O_NONBLOCK != 0;
        match &mut self.kind {
            Kind::Device(Device::Null) => Ok(0),
            Kind::Device(Device::Zero | Device::Full) => {
                buf.fill(0);
                Ok(buf.len())
            }
            Kind::Device(Device::Random(state)) => {
                for chunk in buf.chunks_mut(8) {
                    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut z = *state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    chunk.copy_from_slice(&(z ^ (z >> 31)).to_le_bytes()[..chunk.len()]);
                }
                Ok(buf.len())
            }
            Kind::Pipe(end) => {
                let mut pipe = end.pipe.lock().unwrap();
                if pipe.buffer.is_empty() {
                    return if pipe.writers == 0 {
                        Ok(0)
                    } else {
                        Err(blocked(nonblock))
                    };
                }
                let n = buf.len().min(pipe.buffer.len());
                for (slot, byte) in buf.iter_mut().zip(pipe.buffer.drain(..n)) {
                    *slot = byte;
                }
                Ok(n)
            }
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
        let nonblock = self.flags & O_NONBLOCK != 0;
        match &mut self.kind {
            Kind::Device(Device::Full) => return Err(Errno::ENOSPC),
            Kind::Device(_) => return Ok(data.len()),
            Kind::Pipe(end) => {
                let mut pipe = end.pipe.lock().unwrap();
                if pipe.readers == 0 {
                    return Err(Errno::EPIPE);
                }
                let room = PIPE_CAPACITY - pipe.buffer.len();
                if room == 0 && !data.is_empty() {
                    return Err(blocked(nonblock));
                }
                let n = data.len().min(room);
                pipe.buffer.extend(&data[..n]);
                return Ok(n);
            }
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
            Kind::Pipe(_) => Ok(Stat::fifo()),
            Kind::Device(device) => Ok(device.stat()),
        }
    }
}

impl Process {
    fn resolve_at(
        &self,
        vcpu: &dyn Vcpu,
        dirfd: u32,
        path: u32,
        follow: bool,
    ) -> Result<Resolved, Errno> {
        let path = read_cstr(vcpu, path, PATH_MAX)?;
        let base = self.base_dir(dirfd)?;
        self.root
            .as_ref()
            .ok_or(Errno::ENOENT)?
            .resolve(&base, &path, follow)
    }

    /// The device `path` names relative to `dirfd`, if it names one.
    fn device_at(
        &self,
        vcpu: &dyn Vcpu,
        dirfd: u32,
        path: u32,
    ) -> Result<Option<Result<Device, u32>>, Errno> {
        let path = read_cstr(vcpu, path, PATH_MAX)?;
        if path.is_empty() {
            return Ok(None);
        }
        Ok(Device::named(&normalize(&self.base_dir(dirfd)?, &path)))
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
        match self.device_at(vcpu, dirfd, path)? {
            Some(Ok(device)) => {
                let flags = flags & (O_ACCMODE | O_APPEND | O_NONBLOCK);
                let description = Description {
                    kind: Kind::Device(device),
                    flags,
                };
                return self.files.install(description, 0, flags & O_CLOEXEC != 0);
            }
            Some(Err(stream)) => return self.dup(stream, 0, flags & O_CLOEXEC != 0),
            None => {}
        }
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
            let n = match self.read(vcpu, fd, base, len) {
                Err(err) if total > 0 && is_blocked(err) => break,
                result => result?,
            };
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
            let n = match self.write(vcpu, fd, base, len) {
                Err(err) if total > 0 && is_blocked(err) => break,
                result => result?,
            };
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
            Kind::Device(_) => 0,
            Kind::Input(_) | Kind::Output(_) | Kind::Pipe(_) => return Err(Errno::ESPIPE),
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
        let name = read_cstr(vcpu, path, PATH_MAX)?;
        let stat = if name.is_empty() && flags & AT_EMPTY_PATH != 0 {
            if dirfd == AT_FDCWD {
                self.stat_path(&self.cwd.clone(), true)?
            } else {
                self.files.get(dirfd)?.lock().unwrap().stat()?
            }
        } else {
            match self.device_at(vcpu, dirfd, path)? {
                Some(Ok(device)) => device.stat(),
                Some(Err(stream)) => self.files.get(stream)?.lock().unwrap().stat()?,
                None => {
                    let follow = flags & AT_SYMLINK_NOFOLLOW == 0;
                    stat_host(&self.resolve_at(vcpu, dirfd, path, follow)?, follow)?
                }
            }
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
        if self.device_at(vcpu, dirfd, path)?.is_some() {
            return Ok(0);
        }
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
        let target = read_cstr(vcpu, target, PATH_MAX)?;
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
        now: u64,
    ) -> Result<u32, Errno> {
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
                Kind::Input(_) | Kind::Output(_) | Kind::Pipe(_) | Kind::Device(_) => return Ok(0),
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

    pub(super) fn pipe2(
        &mut self,
        vcpu: &mut dyn Vcpu,
        fds: u32,
        flags: u32,
    ) -> Result<u32, Errno> {
        if flags & !(O_CLOEXEC | O_NONBLOCK) != 0 {
            return Err(Errno::EINVAL);
        }
        let pipe = Arc::new(Mutex::new(Pipe {
            buffer: VecDeque::new(),
            readers: 1,
            writers: 1,
        }));
        let end = |writer: bool| Description {
            kind: Kind::Pipe(PipeEnd {
                pipe: pipe.clone(),
                writer,
            }),
            flags: if writer { O_WRONLY } else { O_RDONLY } | (flags & O_NONBLOCK),
        };
        let cloexec = flags & O_CLOEXEC != 0;
        let read = self.files.install(end(false), 0, cloexec)?;
        let installed = self.files.install(end(true), 0, cloexec).and_then(|write| {
            let pair = [read.to_le_bytes(), write.to_le_bytes()].concat();
            vcpu.write_mem(fds, &pair).map_err(|fault| {
                self.files.slots.remove(&write);
                Errno::from(fault)
            })
        });
        if let Err(errno) = installed {
            self.files.slots.remove(&read);
            return Err(errno);
        }
        Ok(0)
    }

    /// Read the program `execve` names, following one `#!` line, along with the
    /// new arguments and environment. The caller replaces the process image.
    pub(super) fn execve(
        &self,
        vcpu: &dyn Vcpu,
        path: u32,
        argv: u32,
        envp: u32,
    ) -> Result<Exec, Errno> {
        let path = read_cstr(vcpu, path, PATH_MAX)?;
        let mut argv = read_strings(vcpu, argv)?;
        let envp = read_strings(vcpu, envp)?;
        let mut elf = self.read_executable(&path)?;
        if let Some(rest) = elf.strip_prefix(b"#!") {
            let end = rest
                .iter()
                .position(|&b| b == b'\n')
                .unwrap_or(rest.len())
                .min(255);
            let line = rest[..end].trim_ascii();
            let split = line
                .iter()
                .position(u8::is_ascii_whitespace)
                .unwrap_or(line.len());
            let (interpreter, arg) = (line[..split].to_vec(), line[split..].trim_ascii().to_vec());
            if interpreter.is_empty() {
                return Err(Errno::ENOEXEC);
            }
            let mut script_argv = vec![interpreter.clone()];
            if !arg.is_empty() {
                script_argv.push(arg);
            }
            script_argv.push(path);
            script_argv.extend(argv.into_iter().skip(1));
            argv = script_argv;
            elf = self.read_executable(&interpreter)?;
        }
        elf::parse(&elf, vcpu.mem_size()).map_err(|_| Errno::ENOEXEC)?;
        Ok(Exec { elf, argv, envp })
    }

    fn read_executable(&self, path: &[u8]) -> Result<Vec<u8>, Errno> {
        let root = self.root.as_ref().ok_or(Errno::ENOENT)?;
        let resolved = root.resolve(&self.cwd, path, true)?;
        let meta = std::fs::metadata(&resolved.host)?;
        if !meta.is_file() || meta.mode() & 0o111 == 0 {
            return Err(Errno::EACCES);
        }
        if meta.len() > MAX_EXECUTABLE {
            return Err(Errno::ENOMEM);
        }
        Ok(std::fs::read(&resolved.host)?)
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
            F_SETFL => {
                let mut description = slot.description.lock().unwrap();
                let settable = O_APPEND | O_NONBLOCK;
                description.flags = (description.flags & !settable) | (arg & settable);
                Ok(0)
            }
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

/// Join `path` to the guest directory `base` and resolve `.` and `..` by text
/// alone, the way device paths are matched.
fn normalize(base: &[u8], path: &[u8]) -> Vec<u8> {
    let mut parts: Vec<&[u8]> = Vec::new();
    let start: &[u8] = if path.starts_with(b"/") { b"" } else { base };
    for part in start
        .split(|&b| b == b'/')
        .chain(path.split(|&b| b == b'/'))
    {
        match part {
            b"" | b"." => {}
            b".." => {
                parts.pop();
            }
            _ => parts.push(part),
        }
    }
    let mut out = Vec::new();
    for part in parts {
        out.push(b'/');
        out.extend_from_slice(part);
    }
    out
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

/// A guest call that cannot finish yet: blocked, or would block on a
/// non-blocking descriptor.
fn blocked(nonblock: bool) -> Errno {
    if nonblock {
        Errno::EAGAIN
    } else {
        Errno::WOULD_BLOCK
    }
}

fn is_blocked(err: Errno) -> bool {
    err == Errno::WOULD_BLOCK || err == Errno::EAGAIN
}

/// Read a NULL-terminated array of string pointers, as `execve` takes them.
fn read_strings(vcpu: &dyn Vcpu, list: u32) -> Result<Vec<Vec<u8>>, Errno> {
    let mut strings = Vec::new();
    let mut total = 0;
    if list == 0 {
        return Ok(strings);
    }
    for i in 0u32.. {
        let mut pointer = [0u8; 4];
        vcpu.read_mem(list.checked_add(4 * i).ok_or(Errno::EFAULT)?, &mut pointer)?;
        let pointer = u32::from_le_bytes(pointer);
        if pointer == 0 {
            break;
        }
        let string = read_cstr(vcpu, pointer, MAX_ARG_STRLEN).map_err(|err| match err {
            Errno::ENAMETOOLONG => Errno::E2BIG,
            err => err,
        })?;
        total += string.len() + 1 + 4;
        if total > ARG_MAX {
            return Err(Errno::E2BIG);
        }
        strings.push(string);
    }
    Ok(strings)
}

/// Read a NUL-terminated string of at most `max` bytes from guest memory.
pub fn read_cstr(vcpu: &dyn Vcpu, addr: u32, max: usize) -> Result<Vec<u8>, Errno> {
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
        if out.len() >= max {
            return Err(Errno::ENAMETOOLONG);
        }
        at += room as u32;
    }
}
