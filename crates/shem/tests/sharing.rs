#![cfg(unix)]

//! What images that hold the same layer, or the same large file, keep on the
//! disk once: the blobs of their layouts, and the files of their unpacked copies.

use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use shem::image::{self, BuildOptions, Image, Store};

/// Bytes that no other file of the test has, as large as weights are, in miniature.
fn weights(seed: u8) -> Vec<u8> {
    (0..2u32 << 20)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8 ^ seed)
        .collect()
}

/// An image of the weights and a file of its own, installed in the store at `home` as `name`.
fn install(home: &Path, name: &str, weights: &[u8], own: &str) -> (Image, PathBuf, String) {
    let context = tempfile::tempdir().unwrap();
    fs::write(context.path().join("weights"), weights).unwrap();
    fs::write(context.path().join("own"), own).unwrap();
    fs::write(
        context.path().join("Shemfile"),
        "FROM scratch\nCOPY weights /model/weights\nCOPY own /own\n",
    )
    .unwrap();
    let image = image::build(&BuildOptions {
        shemfile: context.path().join("Shemfile"),
        context: context.path().to_path_buf(),
        build_args: Default::default(),
        mem: Some(1 << 20),
        home: Some(tempfile::tempdir().unwrap().keep()),
    })
    .unwrap();
    let store = Store::at(home);
    let staging = tempfile::tempdir_in(home).unwrap();
    let layout = staging.path().join("layout");
    let digest = image.write_layout(&layout, "latest").unwrap();
    let dir = store.path(name).unwrap();
    store.install(&layout, &dir).unwrap();
    (image, dir, digest)
}

/// The files directly in `dir`, if it is there.
fn files_in(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .map(|listing| listing.flatten().map(|entry| entry.path()).collect())
        .unwrap_or_default()
}

/// The blob of `dir`'s layout that is the weights' layer: the largest.
fn largest_blob(dir: &Path) -> PathBuf {
    files_in(&dir.join("blobs/sha256"))
        .into_iter()
        .max_by_key(|path| fs::metadata(path).unwrap().len())
        .unwrap()
}

#[test]
fn images_with_a_layer_in_common_keep_its_blob_once() {
    let home = tempfile::tempdir().unwrap();
    let store = Store::at(home.path());
    let same = weights(1);
    let (_, first, _) = install(home.path(), "example.com/a", &same, "first");
    let (_, second, _) = install(home.path(), "example.com/b", &same, "second");
    let (blob, other) = (largest_blob(&first), largest_blob(&second));
    assert_eq!(blob.file_name(), other.file_name(), "the same layer");
    assert_eq!(
        fs::metadata(&blob).unwrap().ino(),
        fs::metadata(&other).unwrap().ino()
    );
    // The store's own file, and the one of each layout
    assert_eq!(fs::metadata(&blob).unwrap().nlink(), 3);
    // Images with other weights do not share them
    let (_, third, _) = install(home.path(), "example.com/c", &weights(2), "third");
    assert_ne!(
        fs::metadata(largest_blob(&third)).unwrap().ino(),
        fs::metadata(&blob).unwrap().ino()
    );
    // Removing one leaves the other whole; removing both, nothing behind
    store.remove("example.com/a").unwrap();
    assert_eq!(
        fs::read(&other).unwrap().len(),
        fs::metadata(&other).unwrap().len() as usize
    );
    store.remove("example.com/b").unwrap();
    store.remove("example.com/c").unwrap();
    assert!(files_in(&home.path().join("blobs")).is_empty());
}

#[test]
fn the_copies_unpacked_to_run_images_keep_a_large_file_once() {
    let home = tempfile::tempdir().unwrap();
    let store = Store::at(home.path());
    let same = weights(3);
    let (one, _, one_digest) = install(home.path(), "example.com/a", &same, "first");
    let (two, _, two_digest) = install(home.path(), "example.com/b", &same, "second");
    let (a, b) = (
        store.rootfs(&one, &one_digest).unwrap(),
        store.rootfs(&two, &two_digest).unwrap(),
    );
    let (mine, yours) = (a.join("model/weights"), b.join("model/weights"));
    assert_eq!(fs::read(&mine).unwrap(), same);
    assert_eq!(
        fs::metadata(&mine).unwrap().ino(),
        fs::metadata(&yours).unwrap().ino()
    );
    // Read-only, since a write would change both
    assert_eq!(
        fs::metadata(&mine).unwrap().permissions().mode() & 0o777,
        0o444
    );
    // Small files are each image's own
    assert_ne!(
        fs::metadata(a.join("own")).unwrap().ino(),
        fs::metadata(b.join("own")).unwrap().ino()
    );
    assert_eq!(fs::read_to_string(b.join("own")).unwrap(), "second");
    // The copy that is left is whole, and when both are gone so is the file
    store.remove("example.com/a").unwrap();
    assert!(!a.exists());
    assert_eq!(fs::read(&yours).unwrap(), same);
    store.remove("example.com/b").unwrap();
    assert!(files_in(&home.path().join("files")).is_empty());
}

#[test]
fn prune_clears_what_no_image_holds() {
    let home = tempfile::tempdir().unwrap();
    let store = Store::at(home.path());
    let (image, _, digest) = install(home.path(), "example.com/a", &weights(4), "x");
    store.rootfs(&image, &digest).unwrap();
    assert_eq!(files_in(&home.path().join("files")).len(), 1);
    store.prune(true).unwrap();
    assert!(files_in(&home.path().join("files")).is_empty());
    // The image itself is still there, and its blobs
    assert!(!files_in(&home.path().join("blobs")).is_empty());
    assert!(store.rootfs(&image, &digest).unwrap().join("own").is_file());
}
