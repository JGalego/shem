//! An archive for `shem load`, as `docker save` writes it: the images it
//! holds go into the store under their names, and nowhere else.
#![no_main]

use libfuzzer_sys::fuzz_target;
use shem::image::Store;

fuzz_target!(|archive: &[u8]| {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let store = Store::at(&home);
    if let Ok(names) = store.load(archive, None) {
        for name in names {
            assert!(store.path(&name).unwrap().starts_with(&home), "{name}");
        }
    }
    let around: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert!(around.iter().all(|name| name == "home"), "{around:?}");
});
