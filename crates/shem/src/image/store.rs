//! Removing what the store holds: images by reference or ID, and the copies
//! it keeps to run them, which it can make again.

use std::collections::{HashMap, HashSet};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

use super::{Image, Store, parse_reference};

impl Store {
    /// The references `name` stands for: an image's reference, as
    /// `name[:tag]`, or else a prefix of an image ID, with or without
    /// `sha256:`, and every reference of the image it names.
    pub fn resolve(&self, name: &str) -> anyhow::Result<Vec<String>> {
        if let Ok((repository, tag)) = parse_reference(name)
            && self.path(name)?.join("oci-layout").is_file()
        {
            return Ok(vec![format!("{repository}:{tag}")]);
        }
        let prefix = name.strip_prefix("sha256:").unwrap_or(name);
        if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("{name}: no such image (see `shem images`)");
        }
        let prefix = prefix.to_ascii_lowercase();
        let matches: Vec<(String, String)> = self
            .list()?
            .into_iter()
            .filter(|(_, digest)| digest.trim_start_matches("sha256:").starts_with(&prefix))
            .collect();
        let digests: HashSet<&str> = matches.iter().map(|(_, d)| d.as_str()).collect();
        match digests.len() {
            0 => bail!("{name}: no such image (see `shem images`)"),
            1 => Ok(matches
                .into_iter()
                .map(|(reference, _)| reference)
                .collect()),
            n => bail!("{name}: the start of {n} images' IDs; give more of it"),
        }
    }

    /// Untag `reference`, and if no other reference holds its image, delete
    /// the copy of it unpacked to run, returning its digest as deleted.
    pub fn remove(&self, reference: &str) -> anyhow::Result<Option<String>> {
        let dir = self.path(reference)?;
        if !dir.join("oci-layout").is_file() {
            bail!("{reference}: no such image (see `shem images`)");
        }
        let digest = Image::layout_digest(&dir)?;
        self.discard(&dir)?;
        // The directories of its name, now that they may hold nothing
        let mut parent = dir.parent();
        while let Some(p) = parent.filter(|p| *p != self.root && p.starts_with(&self.root)) {
            if std::fs::remove_dir(p).is_err() {
                break;
            }
            parent = p.parent();
        }
        if self.list()?.iter().any(|(_, d)| *d == digest) {
            return Ok(None);
        }
        let unpacked = self
            .cache("rootfs")
            .join(digest.trim_start_matches("sha256:"));
        if unpacked.exists() {
            self.discard(&unpacked)?;
        }
        Ok(Some(digest))
    }

    /// Delete the copies the store keeps to run images that it no longer
    /// holds, or with `all` every copy, of images and of ModelPack artifacts
    /// alike, to be made again when next run. Returns each path deleted and
    /// the bytes that freed.
    pub fn prune(&self, all: bool) -> anyhow::Result<Vec<(PathBuf, u64)>> {
        let mut kept: HashSet<String> = HashSet::new();
        let mut doomed = Vec::new();
        for twin in entries(&self.cache("twins"))? {
            match all {
                true => doomed.push(twin),
                // Kept, so the copy unpacked from it is in use too
                false => {
                    if let Ok(digest) = Image::layout_digest(&twin) {
                        kept.insert(digest);
                    }
                }
            }
        }
        if !all {
            kept.extend(self.list()?.into_iter().map(|(_, digest)| digest));
        }
        for unpacked in entries(&self.cache("rootfs"))? {
            let name = unpacked.file_name().unwrap().to_string_lossy();
            if !kept.contains(&format!("sha256:{name}")) {
                doomed.push(unpacked);
            }
        }
        let mut deleted = Vec::new();
        for path in doomed {
            let size = freed_by(&path)?;
            self.discard(&path)?;
            deleted.push((path, size));
        }
        Ok(deleted)
    }

    /// The directory beside the images where the store keeps `name`.
    fn cache(&self, name: &str) -> PathBuf {
        self.root.with_file_name(name)
    }

    /// Delete `path`, first moving it out of where it is seen, so that a
    /// deletion cut short never leaves half an image behind.
    fn discard(&self, path: &Path) -> anyhow::Result<()> {
        let tmp = self.cache("tmp");
        std::fs::create_dir_all(&tmp)?;
        let staging = tempfile::tempdir_in(&tmp)?;
        let target = staging.path().join("deleted");
        if std::fs::rename(path, &target).is_err() {
            std::fs::remove_dir_all(path)
                .with_context(|| format!("deleting {}", path.display()))?;
        }
        staging.close()?;
        Ok(())
    }
}

/// The entries of `dir` that are not left over from making one: a staging
/// directory's name starts with a dot.
fn entries(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let Ok(listing) = std::fs::read_dir(dir) else {
        return Ok(Vec::new());
    };
    let mut paths = Vec::new();
    for entry in listing {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with('.') {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

/// The bytes deleting `path` frees: those of the files under it that no
/// link outside it keeps, such as weights linked from an image's layers.
fn freed_by(path: &Path) -> anyhow::Result<u64> {
    let mut files: HashMap<(u64, u64), (u64, u64, u64)> = HashMap::new();
    let mut pending = vec![path.to_path_buf()];
    while let Some(path) = pending.pop() {
        let meta = std::fs::symlink_metadata(&path)?;
        if meta.is_dir() {
            for entry in std::fs::read_dir(&path)? {
                pending.push(entry?.path());
            }
        } else {
            let file =
                files
                    .entry((meta.dev(), meta.ino()))
                    .or_insert((meta.len(), meta.nlink(), 0));
            file.2 += 1;
        }
    }
    Ok(files
        .values()
        .filter(|(_, links, seen)| seen >= links)
        .map(|(size, ..)| size)
        .sum())
}
