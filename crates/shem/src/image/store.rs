//! Removing what the store holds: images by reference or ID, and the copies
//! it keeps to run them, which it can make again.

use std::collections::{HashMap, HashSet};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

use super::{Image, REF_NAME, Store, parse_reference};

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

    /// Store the image in the layout `source` as `target` too, as `docker
    /// tag` does, in place of any image that was `target`. The copy links
    /// to the source's blobs where it can.
    pub fn tag(&self, source: &Path, target: &str) -> anyhow::Result<()> {
        let (_, tag) = parse_reference(target)?;
        let dir = self.path(target)?;
        if dir.canonicalize().ok() == Some(source.canonicalize()?) {
            return Ok(());
        }
        let tmp = self.cache("tmp");
        std::fs::create_dir_all(&tmp)?;
        let staging = tempfile::tempdir_in(&tmp)?;
        let copy = staging.path().join("layout");
        link_tree(source, &copy)?;
        // The index names the tag
        let mut index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(source.join("index.json"))?)?;
        for entry in index["manifests"].as_array_mut().into_iter().flatten() {
            entry["annotations"][REF_NAME] = tag.into();
        }
        // A new file, not one linked to the source's index
        std::fs::remove_file(copy.join("index.json"))?;
        std::fs::write(copy.join("index.json"), serde_json::to_vec(&index)?)?;
        if dir.exists() {
            self.discard(&dir)?;
        }
        std::fs::create_dir_all(dir.parent().context("no parent")?)?;
        std::fs::rename(&copy, &dir)?;
        Ok(())
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
    /// alike, to be made again when next run, and the build cache. Returns each path deleted and
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
        if all {
            // The build cache: models imported and sessions primed, made again by the next build
            doomed.extend(entries(&self.cache("imports"))?);
            doomed.extend(entries(&self.cache("warms"))?);
        }
        let mut deleted = Vec::new();
        for path in doomed {
            let size = freed_by(&path)?;
            self.discard(&path)?;
            deleted.push((path, size));
        }
        Ok(deleted)
    }

    /// Whether the build cache is on, which `SHEM_NO_CACHE` or `shem build
    /// --no-cache` turns off.
    pub fn caching() -> bool {
        std::env::var_os("SHEM_NO_CACHE").is_none_or(|value| value.is_empty())
    }

    /// Link the model a build cached under `key` into `to`, if it did.
    pub fn cached_import(&self, key: &str, to: &Path) -> anyhow::Result<bool> {
        let cached = self.cache("imports").join(key);
        if !cached.join("shem.json").is_file() {
            return Ok(false);
        }
        link_tree(&cached, to)?;
        Ok(true)
    }

    /// Keep the model in `from` for builds that import the same checkpoint.
    /// Its files are links, not copies, so the cache costs the weights once
    /// until the build's own copies are gone.
    pub fn cache_import(&self, key: &str, from: &Path) -> anyhow::Result<()> {
        self.keep("imports", key, |staging| link_tree(from, staging))
    }

    /// The session a build cached under `key`.
    pub fn cached_warm(&self, key: &str) -> Option<Vec<u8>> {
        std::fs::read(self.cache("warms").join(key).join("state.tar")).ok()
    }

    /// Keep the state layer a `WARM` made, for the build that would make it again.
    pub fn cache_warm(&self, key: &str, state: &[u8]) -> anyhow::Result<()> {
        self.keep("warms", key, |staging| {
            std::fs::create_dir_all(staging)?;
            std::fs::write(staging.join("state.tar"), state)?;
            Ok(())
        })
    }

    /// Make the entry `key` of the cache `name` with `fill`, in a staging
    /// directory renamed into place, so that a build cut short leaves none.
    fn keep(
        &self,
        name: &str,
        key: &str,
        fill: impl FnOnce(&Path) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let dir = self.cache(name);
        let target = dir.join(key);
        if target.exists() {
            return Ok(());
        }
        std::fs::create_dir_all(&dir)?;
        let staging = tempfile::Builder::new().prefix(".").tempdir_in(&dir)?;
        let inner = staging.path().join("entry");
        fill(&inner)?;
        // Another build may have made it meanwhile: either is the same
        if std::fs::rename(&inner, &target).is_err() && !target.exists() {
            bail!("keeping {} in the cache", target.display());
        }
        Ok(())
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

/// Copy the directory `from` to `to`, hard-linking its files where it can.
fn link_tree(from: &Path, to: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            link_tree(&entry.path(), &target)?;
        } else if std::fs::hard_link(entry.path(), &target).is_err() {
            std::fs::copy(entry.path(), &target)
                .with_context(|| format!("copying {}", entry.path().display()))?;
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_a_build_keeps_comes_back_and_prune_all_empties_it() {
        let home = tempfile::tempdir().unwrap();
        let store = Store::at(home.path());
        assert_eq!(store.cached_warm("k"), None);
        store.cache_warm("k", b"state").unwrap();
        store.cache_warm("k", b"other").unwrap(); // the first is kept
        assert_eq!(store.cached_warm("k").as_deref(), Some(&b"state"[..]));

        let model = tempfile::tempdir().unwrap();
        std::fs::write(model.path().join("shem.json"), "{}").unwrap();
        std::fs::write(model.path().join("w.safetensors"), "weights").unwrap();
        let into = tempfile::tempdir().unwrap();
        assert!(!store.cached_import("m", into.path()).unwrap());
        store.cache_import("m", model.path()).unwrap();
        assert!(store.cached_import("m", into.path()).unwrap());
        assert_eq!(
            std::fs::read(into.path().join("w.safetensors")).unwrap(),
            b"weights"
        );

        assert!(
            store.prune(false).unwrap().is_empty(),
            "a build cache is not a stale copy"
        );
        assert_eq!(store.prune(true).unwrap().len(), 2);
        assert_eq!(store.cached_warm("k"), None);
    }
}
