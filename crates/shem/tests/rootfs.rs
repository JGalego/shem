use std::fs;
use std::os::unix::fs::symlink;
use std::path::PathBuf;

use shem::linux::{Errno, RootFs};
use tempfile::TempDir;

struct Root {
    _dir: TempDir,
    host: PathBuf,
    fs: RootFs,
}

/// A small rootfs with the kinds of symlinks a real one has, and some hostile ones.
fn root() -> Root {
    let dir = tempfile::tempdir().unwrap();
    let host = fs::canonicalize(dir.path()).unwrap();
    fs::create_dir(host.join("bin")).unwrap();
    fs::create_dir(host.join("etc")).unwrap();
    fs::write(host.join("bin/busybox"), b"").unwrap();
    fs::write(host.join("file"), b"").unwrap();
    symlink("/bin/busybox", host.join("bin/ls")).unwrap(); // as `busybox --install -s` makes
    symlink("../../../../../..", host.join("bin/up")).unwrap();
    symlink("/etc/passwd", host.join("passwd")).unwrap();
    symlink("loop", host.join("loop")).unwrap();
    let fs = RootFs::new(&host).unwrap();
    Root {
        _dir: dir,
        host,
        fs,
    }
}

impl Root {
    fn guest(&self, base: &str, path: &str, follow: bool) -> Result<String, Errno> {
        let resolved = self.fs.resolve(base.as_bytes(), path.as_bytes(), follow)?;
        assert!(
            resolved.host.starts_with(&self.host),
            "{resolved:?} left the root"
        );
        Ok(String::from_utf8(resolved.guest).unwrap())
    }
}

#[test]
fn dotdot_stops_at_the_root() {
    let root = root();
    assert_eq!(
        root.guest("/", "../../../bin/busybox", true),
        Ok("/bin/busybox".into())
    );
    assert_eq!(root.guest("/bin", "../../..", true), Ok("/".into()));
}

#[test]
fn a_base_that_became_a_symlink_resolves_inside_the_root() {
    // The guest's working directory, or a directory it holds open, removed
    // and replaced by a link since it got there: the base is walked again
    let root = root();
    symlink("/", root.host.join("was-a-dir")).unwrap();
    symlink("../../../../../..", root.host.join("was-another")).unwrap();
    assert_eq!(root.guest("/was-a-dir", "etc", true), Ok("/etc".into()));
    assert_eq!(
        root.guest("/was-another", "bin/busybox", true),
        Ok("/bin/busybox".into())
    );
    assert_eq!(
        root.guest("/was-a-dir/bin", "../file", true),
        Ok("/file".into())
    );
    // A base that climbs out, as a forged one could
    assert_eq!(root.guest("/../../..", "file", true), Ok("/file".into()));
    // A base that is gone
    assert_eq!(root.guest("/gone", "file", true), Err(Errno::ENOENT));
}

#[test]
fn absolute_symlinks_resolve_inside_the_root() {
    let root = root();
    assert_eq!(root.guest("/", "/bin/ls", true), Ok("/bin/busybox".into()));
    assert_eq!(root.guest("/", "/bin/ls", false), Ok("/bin/ls".into()));
    // Names the guest's /etc/passwd, which does not exist, never the host's.
    assert_eq!(root.guest("/", "/passwd", true), Ok("/etc/passwd".into()));
}

#[test]
fn relative_symlinks_cannot_climb_out() {
    let root = root();
    assert_eq!(
        root.guest("/", "/bin/up/bin/busybox", true),
        Ok("/bin/busybox".into())
    );
    assert_eq!(root.guest("/", "/bin/up", true), Ok("/".into()));
}

#[test]
fn relative_paths_start_at_the_base_directory() {
    let root = root();
    assert_eq!(
        root.guest("/bin", "busybox", true),
        Ok("/bin/busybox".into())
    );
    assert_eq!(root.guest("/bin", "./ls", true), Ok("/bin/busybox".into()));
}

#[test]
fn symlink_loops_are_detected() {
    let root = root();
    assert_eq!(root.guest("/", "/loop", true), Err(Errno::ELOOP));
    assert_eq!(root.guest("/", "/loop", false), Ok("/loop".into()));
}

#[test]
fn only_the_last_component_may_be_missing_or_a_file() {
    let root = root();
    assert_eq!(root.guest("/", "/missing", true), Ok("/missing".into()));
    assert_eq!(root.guest("/", "/missing/x", true), Err(Errno::ENOENT));
    assert_eq!(root.guest("/", "/file/x", true), Err(Errno::ENOTDIR));
    assert_eq!(root.guest("/", "/file/", true), Err(Errno::ENOTDIR));
}

#[test]
fn rejects_empty_and_overlong_paths() {
    let root = root();
    assert_eq!(root.guest("/", "", true), Err(Errno::ENOENT));
    assert_eq!(
        root.guest("/", &"a".repeat(256), true),
        Err(Errno::ENAMETOOLONG)
    );
    assert_eq!(
        root.guest("/", &"a/".repeat(2048), true),
        Err(Errno::ENAMETOOLONG)
    );
}
