//! The state the shim, as root, reads and writes under a container's `/.shem`
//! never goes through links the image or the guest left there.

use std::fs;
use std::os::unix::fs::symlink;

use containerd_shim_shem::state;

#[test]
fn state_is_written_as_new_files_not_through_links() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("victim"), b"host data").unwrap();

    // /.shem itself a link out of the root
    symlink(outside.path(), root.path().join(".shem")).unwrap();
    state::write(root.path(), "machine.state", b"frozen").unwrap();
    assert!(!outside.path().join("machine.state").exists());
    assert!(
        fs::symlink_metadata(root.path().join(".shem"))
            .unwrap()
            .is_dir()
    );

    // A state file a link to a host file
    symlink(
        outside.path().join("victim"),
        root.path().join(".shem/container.json"),
    )
    .unwrap();
    state::write(root.path(), "container.json", b"{}").unwrap();
    assert_eq!(
        fs::read(outside.path().join("victim")).unwrap(),
        b"host data"
    );
    assert_eq!(
        fs::read(root.path().join(".shem/container.json")).unwrap(),
        b"{}"
    );

    assert!(state::write(root.path(), "../escape", b"").is_err());
    state::remove(root.path()).unwrap();
    assert!(!root.path().join(".shem").exists());
}

#[test]
fn state_is_read_from_regular_files_of_a_bounded_size_only() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("secret"), b"host data").unwrap();
    assert_eq!(
        state::read(root.path(), "container.json", 64).unwrap(),
        None
    );

    // Through a link at /.shem: as if there were no state
    symlink(outside.path(), root.path().join(".shem")).unwrap();
    fs::write(outside.path().join("container.json"), b"{}").unwrap();
    assert_eq!(
        state::read(root.path(), "container.json", 64).unwrap(),
        None
    );
    assert!(!state::has(root.path(), "container.json"));
    fs::remove_file(root.path().join(".shem")).unwrap();

    fs::create_dir(root.path().join(".shem")).unwrap();
    let dir = root.path().join(".shem");
    symlink(outside.path().join("secret"), dir.join("linked")).unwrap();
    assert!(state::read(root.path(), "linked", 64).is_err());
    assert!(!state::has(root.path(), "linked"));

    let fifo = std::ffi::CString::new(dir.join("fifo").to_str().unwrap()).unwrap();
    // SAFETY: a NUL-terminated path
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o644) }, 0);
    assert!(state::read(root.path(), "fifo", 64).is_err());

    fs::write(dir.join("big"), vec![0; 65]).unwrap();
    assert!(state::read(root.path(), "big", 64).is_err());
    fs::write(dir.join("small"), b"fits").unwrap();
    assert_eq!(
        state::read(root.path(), "small", 64).unwrap(),
        Some(b"fits".to_vec())
    );
    assert!(state::has(root.path(), "small"));
}
