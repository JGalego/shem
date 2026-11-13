//! An image's layer, a tar, unpacked into a root filesystem: whatever its
//! paths, links and whiteouts say, nothing outside the root changes.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|layer: &[u8]| {
    let dir = tempfile::tempdir().unwrap();
    let (root, outside) = (dir.path().join("root"), dir.path().join("outside"));
    std::fs::create_dir_all(outside.join("dir")).unwrap();
    std::fs::write(outside.join("file"), "kept").unwrap();
    std::fs::create_dir(&root).unwrap();
    let _ = shem::image::unpack(layer, &root);
    let mut around: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    around.sort();
    assert_eq!(around, ["outside", "root"]);
    assert_eq!(
        std::fs::read_to_string(outside.join("file")).unwrap(),
        "kept"
    );
    assert!(outside.join("dir").is_dir());
});
