//! The host's operating system, behind the names `std::os::unix` has.
//!
//! On a Unix the module re-exports the traits the rest of shem reads files
//! with; on Windows it gives the same names to what Windows has the nearest
//! of, so the code that serves a guest's system calls is the same on both.
//! What Windows has no kind of (inodes, permission bits beyond read-only,
//! devices, sockets as files, signals) is made up as a plain file would have
//! it, or is not there: the guest sees an error, as on a kernel without it.

pub mod ffi {
    #[cfg(unix)]
    pub use std::os::unix::ffi::OsStrExt;

    #[cfg(windows)]
    pub trait OsStrExt {
        fn from_bytes(bytes: &[u8]) -> &Self;
        fn as_bytes(&self) -> &[u8];
    }

    #[cfg(windows)]
    impl OsStrExt for std::ffi::OsStr {
        /// A name Windows has: guest names that are not UTF-8 are not
        /// representable in its paths, and name nothing.
        fn from_bytes(bytes: &[u8]) -> &Self {
            std::ffi::OsStr::new(std::str::from_utf8(bytes).unwrap_or("\u{fffd}"))
        }

        fn as_bytes(&self) -> &[u8] {
            self.as_encoded_bytes()
        }
    }
}

pub mod fs {
    #[cfg(unix)]
    pub use std::os::unix::fs::{
        DirBuilderExt, DirEntryExt, FileExt, FileTypeExt, MetadataExt, OpenOptionsExt,
        PermissionsExt, symlink,
    };

    #[cfg(windows)]
    pub use windows::*;

    /// The permission bits of `path`, as `chmod` would set them: on Windows
    /// the write bits go with the read-only attribute and nothing else.
    pub fn set_mode(path: &std::path::Path, mode: u32) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            std::fs::set_permissions(path, PermissionsExt::from_mode(mode))
        }
        #[cfg(windows)]
        {
            let mut permissions = std::fs::metadata(path)?.permissions();
            permissions.set_readonly(mode & 0o222 == 0);
            std::fs::set_permissions(path, permissions)
        }
    }

    /// What names the file at `path` on its host, when its metadata does not:
    /// its volume and its index there, which only Windows keeps apart from
    /// the metadata std gives. None where `MetadataExt` has them.
    pub fn identity_of_path(path: &std::path::Path) -> Option<(u64, u64)> {
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
            // Opened for nothing but its information, which a directory allows too
            let file = std::fs::OpenOptions::new()
                .access_mode(0)
                .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
                .open(path)
                .ok()?;
            identity_of_file(&file)
        }
        #[cfg(not(windows))]
        {
            let _ = path;
            None
        }
    }

    /// As `identity_of_path`, for a file that is open.
    pub fn identity_of_file(file: &std::fs::File) -> Option<(u64, u64)> {
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Storage::FileSystem::{
                BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
            };
            // SAFETY: the handle is open for the call, and the struct is one it fills in
            let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
            let done = unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) };
            (done != 0).then_some((
                u64::from(info.dwVolumeSerialNumber),
                u64::from(info.nFileIndexHigh) << 32 | u64::from(info.nFileIndexLow),
            ))
        }
        #[cfg(not(windows))]
        {
            let _ = file;
            None
        }
    }

    /// The permission bits of `meta`.
    pub fn mode_of(meta: &std::fs::Metadata) -> u32 {
        MetadataExt::mode(meta) & 0o7777
    }

    #[cfg(windows)]
    mod windows {
        use std::fs::{File, Metadata};
        use std::io;
        use std::os::windows::fs::{FileExt as _, MetadataExt as _};
        use std::path::Path;

        const S_IFDIR: u32 = 0o040000;
        const S_IFREG: u32 = 0o100000;
        const S_IFLNK: u32 = 0o120000;
        const FILE_ATTRIBUTE_READONLY: u32 = 1;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        /// 100 ns ticks from 1601 to 1970
        const EPOCH: u64 = 116_444_736_000_000_000;

        fn unix_time(ticks: u64) -> (i64, i64) {
            let since = ticks.saturating_sub(EPOCH);
            (
                (since / 10_000_000) as i64,
                (since % 10_000_000 * 100) as i64,
            )
        }

        /// What Windows tells of a file, in the shape Unix's `stat` has.
        pub trait MetadataExt {
            fn dev(&self) -> u64;
            fn ino(&self) -> u64;
            fn mode(&self) -> u32;
            fn nlink(&self) -> u64;
            fn size(&self) -> u64;
            fn rdev(&self) -> u64;
            fn atime(&self) -> i64;
            fn atime_nsec(&self) -> i64;
            fn mtime(&self) -> i64;
            fn mtime_nsec(&self) -> i64;
        }

        impl MetadataExt for Metadata {
            fn dev(&self) -> u64 {
                0
            }

            /// The file's creation time, which no other file in a directory
            /// shares in practice: Windows' file index is not in std.
            fn ino(&self) -> u64 {
                self.creation_time()
            }

            fn mode(&self) -> u32 {
                let attributes = self.file_attributes();
                let write = if attributes & FILE_ATTRIBUTE_READONLY != 0 {
                    0
                } else {
                    0o222
                };
                if attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 && self.is_symlink() {
                    S_IFLNK | 0o777
                } else if self.is_dir() {
                    S_IFDIR | 0o555 | write
                } else {
                    S_IFREG | 0o555 | write
                }
            }

            fn nlink(&self) -> u64 {
                1
            }

            fn size(&self) -> u64 {
                self.file_size()
            }

            fn rdev(&self) -> u64 {
                0
            }

            fn atime(&self) -> i64 {
                unix_time(self.last_access_time()).0
            }

            fn atime_nsec(&self) -> i64 {
                unix_time(self.last_access_time()).1
            }

            fn mtime(&self) -> i64 {
                unix_time(self.last_write_time()).0
            }

            fn mtime_nsec(&self) -> i64 {
                unix_time(self.last_write_time()).1
            }
        }

        pub trait PermissionsExt {
            fn mode(&self) -> u32;
        }

        impl PermissionsExt for std::fs::Permissions {
            fn mode(&self) -> u32 {
                if self.readonly() { 0o444 } else { 0o644 }
            }
        }

        /// Reads and writes at an offset, as `pread` and `pwrite` do.
        pub trait FileExt {
            fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;
            fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize>;
        }

        impl FileExt for File {
            fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
                self.seek_read(buf, offset)
            }

            fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
                self.seek_write(buf, offset)
            }
        }

        /// What has no meaning on Windows, and is ignored there: the mode a
        /// directory or file is made with.
        pub trait DirBuilderExt {
            fn mode(&mut self, mode: u32) -> &mut Self;
        }

        impl DirBuilderExt for std::fs::DirBuilder {
            fn mode(&mut self, _mode: u32) -> &mut Self {
                self
            }
        }

        pub trait OpenOptionsExt {
            fn mode(&mut self, mode: u32) -> &mut Self;
        }

        impl OpenOptionsExt for std::fs::OpenOptions {
            fn mode(&mut self, _mode: u32) -> &mut Self {
                self
            }
        }

        /// A directory entry's inode: its name's hash, since Windows keeps
        /// none that a listing gives.
        pub trait DirEntryExt {
            fn ino(&self) -> u64;
        }

        impl DirEntryExt for std::fs::DirEntry {
            fn ino(&self) -> u64 {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                self.file_name().hash(&mut hasher);
                hasher.finish()
            }
        }

        /// Windows has no FIFOs, sockets or devices as files.
        pub trait FileTypeExt {
            fn is_block_device(&self) -> bool;
            fn is_char_device(&self) -> bool;
            fn is_fifo(&self) -> bool;
            fn is_socket(&self) -> bool;
        }

        impl FileTypeExt for std::fs::FileType {
            fn is_block_device(&self) -> bool {
                false
            }

            fn is_char_device(&self) -> bool {
                false
            }

            fn is_fifo(&self) -> bool {
                false
            }

            fn is_socket(&self) -> bool {
                false
            }
        }

        /// A symlink to `target`, which Windows only lets a user make with
        /// the right to, or in developer mode.
        pub fn symlink(target: impl AsRef<Path>, link: impl AsRef<Path>) -> io::Result<()> {
            let (target, link) = (target.as_ref(), link.as_ref());
            // A link to a directory is another call there than a link to a file
            let to_dir = link
                .parent()
                .map(|parent| parent.join(target))
                .is_some_and(|path| path.is_dir());
            if to_dir {
                std::os::windows::fs::symlink_dir(target, link)
            } else {
                std::os::windows::fs::symlink_file(target, link)
            }
        }
    }
}

/// An exclusive lock on `file`, waited for, held until the file is closed.
pub fn lock(file: &std::fs::File) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: the descriptor is open for the call
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{LOCKFILE_EXCLUSIVE_LOCK, LockFileEx};
        use windows_sys::Win32::System::IO::OVERLAPPED;
        // SAFETY: the handle is open for the call, and the zeroed OVERLAPPED
        // names offset 0 and no event: the call waits for the lock
        let locked = unsafe {
            let mut overlapped: OVERLAPPED = std::mem::zeroed();
            LockFileEx(
                file.as_raw_handle(),
                LOCKFILE_EXCLUSIVE_LOCK,
                0,
                u32::MAX,
                u32::MAX,
                &mut overlapped,
            )
        };
        if locked == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Open the lock file `path`, made if it is not there, and lock it: not
/// through a link a process of another user left in its place, and not
/// kept open by the programs shem starts.
pub fn lock_file(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).truncate(false).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .mode(0o600);
    }
    let file = options.open(path)?;
    lock(&file)?;
    Ok(file)
}

/// Unix domain sockets, which the neural vCPU server and the model server
/// are reached by. Windows has them, but std does not: there, connecting
/// fails, and a run goes without those servers.
pub mod net {
    #[cfg(unix)]
    pub use std::os::unix::net::UnixStream;

    #[cfg(windows)]
    pub use windows::UnixStream;

    #[cfg(windows)]
    mod windows {
        use std::io::{self, Read, Write};
        use std::path::Path;

        /// A stream that cannot be made: there is nothing to read or write.
        #[derive(Debug)]
        pub struct UnixStream(());

        impl UnixStream {
            pub fn connect(_: impl AsRef<Path>) -> io::Result<Self> {
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "Unix domain sockets are not available in this build of shem on Windows",
                ))
            }
        }

        impl Read for UnixStream {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::ErrorKind::Unsupported.into())
            }
        }

        impl Write for UnixStream {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::Unsupported.into())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
    }
}
