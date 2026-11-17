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

#[cfg(target_os = "linux")]
mod beneath {
    use super::*;
    use std::io::Read;

    /// `open_beneath` for a read, as the O_RDONLY a guest's open becomes.
    fn read(root: &Root, guest: &str) -> Option<std::io::Result<String>> {
        root.fs
            .open_beneath(guest.as_bytes(), libc::O_RDONLY | libc::O_NOFOLLOW, 0)
            .map(|file| {
                let mut text = String::new();
                file?.read_to_string(&mut text)?;
                Ok(text)
            })
    }

    #[test]
    fn a_file_is_opened_beneath_the_root_in_one_step() {
        let root = root();
        fs::create_dir(root.host.join("data")).unwrap();
        fs::write(root.host.join("data/note"), "inside").unwrap();
        let Some(opened) = read(&root, "/data/note") else {
            return; // a kernel before openat2
        };
        assert_eq!(opened.unwrap(), "inside");
    }

    #[test]
    fn a_link_swapped_in_after_the_path_was_resolved_leads_nowhere() {
        let root = root();
        let secret = tempfile::tempdir().unwrap();
        fs::write(secret.path().join("note"), "the host's").unwrap();
        fs::create_dir(root.host.join("data")).unwrap();
        fs::write(root.host.join("data/note"), "inside").unwrap();
        // The guest's open resolved /data/note, a file in a directory. Then another
        // process on the host replaces the directory with a link to a place outside.
        let resolved = root.fs.resolve(b"/", b"/data/note", true).unwrap();
        fs::rename(root.host.join("data"), root.host.join("data.old")).unwrap();
        symlink(secret.path(), root.host.join("data")).unwrap();
        // Opening the host path, as an open does without openat2, would read the host's file
        assert_eq!(fs::read_to_string(&resolved.host).unwrap(), "the host's");
        let Some(opened) = read(&root, std::str::from_utf8(&resolved.guest).unwrap()) else {
            return;
        };
        let err = opened.expect_err("a link in the way is refused, not followed");
        assert_eq!(err.raw_os_error(), Some(libc::ELOOP));
        // Nor does a path that climbs out get anywhere
        let climbs = read(&root, "/../../etc/hostname").unwrap();
        assert!(climbs.is_err());
    }

    #[test]
    fn a_directory_swapped_for_a_link_is_not_made_into_or_removed_from() {
        let root = root();
        let outside = tempfile::tempdir().unwrap();
        fs::create_dir(root.host.join("data")).unwrap();
        fs::write(outside.path().join("kept"), "the host's").unwrap();
        let resolved = root.fs.resolve(b"/", b"/data/kept", true).unwrap();
        fs::rename(root.host.join("data"), root.host.join("data.old")).unwrap();
        symlink(outside.path(), root.host.join("data")).unwrap();
        let guest = std::str::from_utf8(&resolved.guest).unwrap();
        let made = root.fs.at_parent(guest.as_bytes(), |dir, name| {
            // SAFETY: the descriptor is open and the name a C string, for the call
            match unsafe { libc::mkdirat(dir, name.as_ptr(), 0o755 as libc::mode_t) } {
                0 => Ok(()),
                _ => Err(std::io::Error::last_os_error()),
            }
        });
        let Some(made) = made else { return };
        // The link is refused, as a link (ELOOP) or as no directory (ENOTDIR, which O_DIRECTORY says of one)
        let refused = made.unwrap_err().raw_os_error();
        assert!(
            refused == Some(libc::ELOOP) || refused == Some(libc::ENOTDIR),
            "{refused:?}"
        );
        assert!(!outside.path().join("kept").is_dir() && outside.path().join("kept").is_file());
        // A parent that is a plain directory works as before
        let ok = root.fs.at_parent(b"/bin/new", |dir, name| {
            match unsafe { libc::mkdirat(dir, name.as_ptr(), 0o755 as libc::mode_t) } {
                0 => Ok(()),
                _ => Err(std::io::Error::last_os_error()),
            }
        });
        assert!(ok.unwrap().is_ok() && root.host.join("bin/new").is_dir());
    }

    #[test]
    fn stat_readlink_times_and_listings_stay_beneath_the_root() {
        let root = root();
        fs::create_dir(root.host.join("data")).unwrap();
        fs::write(root.host.join("data/note"), "inside").unwrap();
        symlink("note", root.host.join("data/link")).unwrap();
        let Some(stat) = root.fs.stat_beneath(b"/data/note") else {
            return; // a kernel before openat2
        };
        assert_eq!(stat.unwrap().st_size, 6);
        // A link is itself, not its target
        let link = root.fs.stat_beneath(b"/data/link").unwrap().unwrap();
        assert_eq!(link.st_mode & 0o170000, 0o120000);
        assert_eq!(
            root.fs.readlink_beneath(b"/data/link").unwrap().unwrap(),
            b"note"
        );
        let now = [libc::timespec {
            tv_sec: 1_000_000,
            tv_nsec: 0,
        }; 2];
        root.fs
            .set_times_beneath(b"/data/note", &now)
            .unwrap()
            .unwrap();
        assert_eq!(
            root.fs
                .stat_beneath(b"/data/note")
                .unwrap()
                .unwrap()
                .st_mtime,
            1_000_000
        );
        let mut names: Vec<_> = root
            .fs
            .list_beneath(b"/data")
            .unwrap()
            .unwrap()
            .into_iter()
            .map(|(name, _, _)| String::from_utf8(name).unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["link", "note"]);

        // A directory swapped for a link to somewhere else is refused by all four
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("note"), "the host's").unwrap();
        fs::rename(root.host.join("data"), root.host.join("data.old")).unwrap();
        symlink(outside.path(), root.host.join("data")).unwrap();
        assert!(root.fs.stat_beneath(b"/data/note").unwrap().is_err());
        assert!(root.fs.readlink_beneath(b"/data/link").unwrap().is_err());
        assert!(
            root.fs
                .set_times_beneath(b"/data/note", &now)
                .unwrap()
                .is_err()
        );
        assert!(root.fs.list_beneath(b"/data").unwrap().is_err());
        assert_ne!(
            fs::metadata(outside.path().join("note"))
                .unwrap()
                .modified()
                .unwrap(),
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000)
        );
    }
}
