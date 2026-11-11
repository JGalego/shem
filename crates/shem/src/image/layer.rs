//! Filesystem layers as reproducible tar archives, and unpacking them.
//!
//! Entries are sorted by path, with mtime 0 and root ownership, so the same
//! files always make the same bytes. Deletions are OCI whiteouts: an empty
//! `.wh.NAME` entry next to the deleted `NAME`.

use std::collections::BTreeMap;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

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
    std::fs::create_dir_all(root)?;
    let mut archive = tar::Archive::new(layer);
    archive.set_preserve_permissions(true);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && let Some(target) = name.strip_prefix(WHITEOUT)
        {
            let doomed = root.join(path.with_file_name(target));
            if doomed.is_dir() && !doomed.is_symlink() {
                std::fs::remove_dir_all(&doomed).ok();
            } else {
                std::fs::remove_file(&doomed).ok();
            }
            continue;
        }
        // A file replacing a directory, or the other way around, replaces it.
        let target = root.join(&path);
        if let Ok(meta) = std::fs::symlink_metadata(&target) {
            let incoming_dir = entry.header().entry_type().is_dir();
            if meta.is_dir() && !incoming_dir {
                std::fs::remove_dir_all(&target)?;
            } else if !meta.is_dir() && !incoming_dir {
                std::fs::remove_file(&target)?;
            }
        }
        // unpack_in refuses paths that would land outside `root`.
        if !entry.unpack_in(root)? {
            bail!("layer entry {} escapes the root", path.display());
        }
    }
    Ok(())
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
