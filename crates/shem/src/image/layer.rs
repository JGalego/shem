//! Filesystem layers as reproducible tar archives, and unpacking them.
//!
//! Entries are sorted by path, with mtime 0 and root ownership, so the same
//! files always make the same bytes. Deletions are OCI whiteouts: an empty
//! `.wh.NAME` entry next to the deleted `NAME`.

use crate::sys::fs::MetadataExt;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};

const WHITEOUT: &str = ".wh.";

/// What a path in a tree is, compared to find what changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Dir { mode: u32 },
    File { mode: u32, hash: Vec<u8> },
    Link { target: PathBuf },
}

/// Every path under `root`, relative to it, with what it is.
pub fn scan(root: &Path) -> anyhow::Result<BTreeMap<PathBuf, Node>> {
    let mut nodes = BTreeMap::new();
    walk(root, Path::new(""), &mut nodes)?;
    Ok(nodes)
}

fn walk(root: &Path, relative: &Path, nodes: &mut BTreeMap<PathBuf, Node>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(root.join(relative))? {
        let entry = entry?;
        let path = relative.join(entry.file_name());
        let meta = entry.metadata()?;
        let node = if meta.is_symlink() {
            Node::Link {
                target: std::fs::read_link(entry.path())?,
            }
        } else if meta.is_dir() {
            walk(root, &path, nodes)?;
            Node::Dir {
                mode: meta.mode() & 0o7777,
            }
        } else if meta.is_file() {
            Node::File {
                mode: meta.mode() & 0o7777,
                hash: Sha256::digest(std::fs::read(entry.path())?).to_vec(),
            }
        } else {
            continue; // devices, sockets and pipes have no place in an image
        };
        nodes.insert(path, node);
    }
    Ok(())
}

/// A layer holding `paths` from `root`, and whiteouts for `deleted`.
pub fn create(root: &Path, paths: &[PathBuf], deleted: &[PathBuf]) -> anyhow::Result<Vec<u8>> {
    let mut entries: Vec<(PathBuf, Option<PathBuf>)> = paths
        .iter()
        .map(|p| (p.clone(), Some(root.join(p))))
        .collect();
    for path in deleted {
        let name = path
            .file_name()
            .context("cannot delete the root")?
            .to_string_lossy();
        entries.push((path.with_file_name(format!("{WHITEOUT}{name}")), None));
    }
    entries.sort();
    entries.dedup();
    let mut builder = tar::Builder::new(Vec::new());
    for (path, source) in entries {
        let mut header = tar::Header::new_ustar();
        header.set_mtime(0);
        header.set_uid(0);
        header.set_gid(0);
        let Some(source) = source else {
            header.set_entry_type(tar::EntryType::Regular);
            header.set_mode(0o644);
            header.set_size(0);
            builder.append_data(&mut header, &path, std::io::empty())?;
            continue;
        };
        let meta = std::fs::symlink_metadata(&source)?;
        header.set_mode(meta.mode() & 0o7777);
        if meta.is_symlink() {
            header.set_entry_type(tar::EntryType::Symlink);
            header.set_size(0);
            builder.append_link(&mut header, &path, std::fs::read_link(&source)?)?;
        } else if meta.is_dir() {
            header.set_entry_type(tar::EntryType::Directory);
            header.set_size(0);
            builder.append_data(&mut header, &path, std::io::empty())?;
        } else {
            header.set_entry_type(tar::EntryType::Regular);
            let data = std::fs::read(&source)?;
            header.set_size(data.len() as u64);
            builder.append_data(&mut header, &path, data.as_slice())?;
        }
    }
    Ok(builder.into_inner()?)
}

/// The paths that are new or changed in `after`, and those deleted from
/// `before`, ready for `create`.
pub fn diff(
    before: &BTreeMap<PathBuf, Node>,
    after: &BTreeMap<PathBuf, Node>,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let changed = after
        .iter()
        .filter(|(path, node)| before.get(*path) != Some(node))
        .map(|(path, _)| path.clone())
        .collect();
    // Only the topmost of a deleted subtree needs a whiteout.
    let mut deleted: Vec<PathBuf> = Vec::new();
    for path in before.keys().filter(|path| !after.contains_key(*path)) {
        if !deleted.iter().any(|parent| path.starts_with(parent)) {
            deleted.push(path.clone());
        }
    }
    (changed, deleted)
}

/// Apply a layer on top of the tree at `root`.
pub fn unpack(layer: &[u8], root: &Path) -> anyhow::Result<()> {
    unpack_sharing(layer, root, None)
}

/// The size from which a file is kept once for all the trees that hold it.
pub const SHARED_FROM: u64 = 1 << 20;

/// Apply a layer on top of the tree at `root`, as `unpack` does, with the
/// files of `SHARED_FROM` bytes and more, if there is a `pool` to keep them
/// in, hard links to files there that other trees link to as well: the same
/// weights in two images are on the disk once. A pooled file is read-only,
/// since a write to it would change every tree it is in.
pub fn unpack_sharing(layer: &[u8], root: &Path, pool: Option<&Path>) -> anyhow::Result<()> {
    std::fs::create_dir_all(root)?;
    let mut archive = tar::Archive::new(layer);
    archive.set_preserve_permissions(true);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && let Some(target) = name.strip_prefix(WHITEOUT)
        {
            // A deletion outside the root deletes nothing
            if let Some(doomed) = inside(root, &path.with_file_name(target)) {
                if doomed.is_dir() && !doomed.is_symlink() {
                    std::fs::remove_dir_all(&doomed).ok();
                } else {
                    std::fs::remove_file(&doomed).ok();
                }
            }
            continue;
        }
        // A file replacing a directory, or the other way around, replaces it.
        if let Some(target) = inside(root, &path)
            && let Ok(meta) = std::fs::symlink_metadata(&target)
        {
            let incoming_dir = entry.header().entry_type().is_dir();
            if meta.is_dir() && !incoming_dir {
                std::fs::remove_dir_all(&target)?;
            } else if !meta.is_dir() && !incoming_dir {
                std::fs::remove_file(&target)?;
            }
        }
        if let Some(pool) = pool
            && entry.header().entry_type() == tar::EntryType::Regular
            && entry.size() >= SHARED_FROM
        {
            let start = entry.raw_file_position() as usize;
            let mode = entry.header().mode().unwrap_or(0o644) & 0o7555;
            let data = layer.get(start..start + entry.size() as usize);
            if let (Some(data), Some(parent)) = (data, plain_parent(root, &path))
                && share(
                    data,
                    mode,
                    pool,
                    &parent.join(path.file_name().unwrap_or_default()),
                )
                .is_ok()
            {
                continue;
            }
        }
        // unpack_in refuses paths that would land outside `root`.
        if !entry.unpack_in(root)? {
            bail!("layer entry {} escapes the root", path.display());
        }
    }
    Ok(())
}

/// The directory the layer entry `path` is in under `root`, made if it is
/// not, when every part of it is a plain directory: None if the path climbs
/// with `..` or goes through a link, which the tar crate makes sure of
/// where this does not.
fn plain_parent(root: &Path, path: &Path) -> Option<PathBuf> {
    let mut names = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => names.push(part),
            Component::CurDir | Component::RootDir => {}
            Component::ParentDir | Component::Prefix(_) => return None,
        }
    }
    names.pop()?;
    let mut dir = root.to_path_buf();
    for name in names {
        dir.push(name);
        match std::fs::symlink_metadata(&dir) {
            Ok(meta) if meta.is_dir() => {}
            Ok(_) => return None,
            Err(_) => std::fs::create_dir(&dir).ok()?,
        }
    }
    Some(dir)
}

/// Put `data` at `target` as a hard link to its file in `pool`, which is
/// named for the content and the mode, and made if no tree has put it
/// there yet. The file is made read-only: it is shared.
fn share(data: &[u8], mode: u32, pool: &Path, target: &Path) -> anyhow::Result<()> {
    use std::io::Write;
    let kept = pool.join(format!("{}-{mode:o}", hex(&Sha256::digest(data))));
    let linked = |from: &Path| std::fs::hard_link(from, target);
    if target.symlink_metadata().is_ok() {
        std::fs::remove_file(target)?;
    }
    if linked(&kept).is_ok() {
        return Ok(());
    }
    // Made under another name first: a name in the pool is only ever a whole file
    std::fs::create_dir_all(pool)?;
    let mut staged = tempfile::Builder::new()
        .prefix(".staged-")
        .tempfile_in(pool)?;
    staged.write_all(data)?;
    crate::sys::fs::set_mode(staged.path(), mode & !0o222)?;
    linked(staged.path())?;
    // Another unpack may have made it first, which is the same file
    let _ = std::fs::hard_link(staged.path(), &kept);
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Where the layer entry `path` is under `root`, which already exists: None
/// if it would be outside, as a path that climbs out with `..` is, or one
/// that goes through a link that leads out. A leading `/` is dropped, as tar
/// does.
fn inside(root: &Path, path: &Path) -> Option<PathBuf> {
    let mut relative = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => relative.push(part),
            Component::CurDir | Component::RootDir => {}
            Component::ParentDir | Component::Prefix(_) => return None,
        }
    }
    let target = root.join(&relative);
    let parent = target.parent()?.canonicalize().ok()?;
    let root = root.canonicalize().ok()?;
    (parent.starts_with(&root) && target != root).then_some(target)
}

/// The regular files in a layer, by path, for small metadata layers.
pub fn files(layer: &[u8]) -> anyhow::Result<BTreeMap<String, Vec<u8>>> {
    let mut files = BTreeMap::new();
    for entry in tar::Archive::new(layer).entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().into_owned();
        let mut data = Vec::new();
        entry.read_to_end(&mut data)?;
        files.insert(path, data);
    }
    Ok(files)
}

/// A layer of in-memory files, for metadata layers such as frozen state.
/// The files of the flat layer `files` (as `from_files` makes) moved under
/// the directory `dir`, which the layer also creates.
pub fn under(flat: &[u8], dir: &str) -> anyhow::Result<Vec<u8>> {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_ustar();
    header.set_entry_type(tar::EntryType::Directory);
    header.set_mtime(0);
    header.set_mode(0o700);
    header.set_size(0);
    builder.append_data(&mut header, format!("{dir}/"), std::io::empty())?;
    for (name, data) in files(flat)? {
        let mut header = tar::Header::new_ustar();
        header.set_mtime(0);
        header.set_mode(0o600);
        header.set_size(data.len() as u64);
        builder.append_data(&mut header, format!("{dir}/{name}"), &data[..])?;
    }
    Ok(builder.into_inner()?)
}

/// The reverse of `under`: if every entry of `layer` is `dir` or in it, the
/// files in it as a flat layer; otherwise None.
pub fn strip(layer: &[u8], dir: &str) -> anyhow::Result<Option<Vec<u8>>> {
    let mut flat = Vec::new();
    for entry in tar::Archive::new(layer).entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().into_owned();
        let path = path.trim_start_matches("./").trim_end_matches('/');
        if path == dir {
            continue;
        }
        let Some(name) = path.strip_prefix(&format!("{dir}/")) else {
            return Ok(None);
        };
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut data)?;
        flat.push((name.to_string(), data));
    }
    if flat.is_empty() {
        return Ok(None);
    }
    let files: Vec<(&str, &[u8])> = flat
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect();
    Ok(Some(from_files(&files)?))
}

pub fn from_files(files: &[(&str, &[u8])]) -> anyhow::Result<Vec<u8>> {
    let mut builder = tar::Builder::new(Vec::new());
    for (name, data) in files {
        let mut header = tar::Header::new_ustar();
        header.set_mtime(0);
        header.set_mode(0o644);
        header.set_size(data.len() as u64);
        builder.append_data(&mut header, name, *data)?;
    }
    Ok(builder.into_inner()?)
}
