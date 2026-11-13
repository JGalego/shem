//! The state a container keeps in its root filesystem, under `/.shem`: a
//! frozen program, or a model's session, which `docker commit` takes with the
//! rest of the filesystem.
//!
//! The shim runs as root, and the root filesystem is the image's and the
//! guest's to shape: either may have left a symlink, a device or a FIFO where
//! the state goes. So the shim never follows a link there, reads only regular
//! files, and only as much as it expects.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use shem::image::WARM_DIR;

/// The most a state file may hold: a frozen machine's memory, which an RV32
/// address space bounds, and a little more.
pub const MAX_STATE: u64 = (4 << 30) + (1 << 20);

/// `/.shem` in `root`, if it is a directory and not a link to one.
pub fn dir(root: &Path) -> Option<PathBuf> {
    let dir = root.join(WARM_DIR);
    std::fs::symlink_metadata(&dir)
        .is_ok_and(|meta| meta.is_dir())
        .then_some(dir)
}

/// Whether `/.shem/name` in `root` is a regular file, reached through no link.
pub fn has(root: &Path, name: &str) -> bool {
    dir(root).is_some_and(|dir| {
        std::fs::symlink_metadata(dir.join(name)).is_ok_and(|meta| meta.file_type().is_file())
    })
}

/// The contents of `/.shem/name` in `root`, if there is such a file: a
/// regular file of at most `max` bytes, reached through no link.
pub fn read(root: &Path, name: &str, max: u64) -> anyhow::Result<Option<Vec<u8>>> {
    let Some(dir) = dir(root) else {
        return Ok(None);
    };
    let path = dir.join(name);
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
    {
        Ok(file) => file,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("opening {}", path.display())),
    };
    let meta = file.metadata()?;
    if !meta.file_type().is_file() {
        bail!("{} is no regular file", path.display());
    }
    if meta.size() > max {
        bail!(
            "{} holds {} bytes, more than {max}",
            path.display(),
            meta.size()
        );
    }
    let mut data = Vec::with_capacity(meta.size() as usize);
    file.take(max + 1).read_to_end(&mut data)?;
    if data.len() as u64 > max {
        bail!("{} grew past {max} bytes", path.display());
    }
    Ok(Some(data))
}

/// `/.shem` in `root`, made a directory: whatever else is there, such as a
/// link, is removed first.
pub fn make_dir(root: &Path) -> anyhow::Result<PathBuf> {
    let dir = root.join(WARM_DIR);
    match std::fs::symlink_metadata(&dir) {
        Ok(meta) if meta.is_dir() => return Ok(dir),
        Ok(_) => std::fs::remove_file(&dir)?,
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }
    std::fs::create_dir(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// Write `data` as `/.shem/name` in `root`, a new file: whatever was there,
/// such as a link, is removed rather than written through.
pub fn write(root: &Path, name: &str, data: &[u8]) -> anyhow::Result<()> {
    if name.contains('/') || name == "." || name == ".." || name.is_empty() {
        bail!("{name:?} is no name for a state file");
    }
    let path = make_dir(root)?.join(name);
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => return Err(err).with_context(|| format!("replacing {}", path.display())),
    }
    let mut file: File = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o644)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)
        .with_context(|| format!("writing {}", path.display()))?;
    file.write_all(data)?;
    Ok(())
}

/// Remove `/.shem` from `root`, a link itself rather than what it leads to.
pub fn remove(root: &Path) -> std::io::Result<()> {
    let dir = root.join(WARM_DIR);
    match std::fs::symlink_metadata(&dir) {
        Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(&dir),
        Ok(_) => std::fs::remove_file(&dir),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}
