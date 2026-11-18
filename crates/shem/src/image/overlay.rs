//! A run's filesystem as an overlay of the image's, with the kernel's overlayfs.
//!
//! An image's filesystem is unpacked once and kept (see `Store::rootfs`),
//! and a run mounts an overlay over it: the unpacked image below, never
//! written to, and above it an empty directory that holds what the guest
//! creates, changes and deletes, and is removed when the run ends. Nothing
//! is unpacked or copied for a run, however large the image, and a guest's
//! writes cannot reach the copy the next run starts from.
//!
//! The mount is made without privileges, in namespaces of the shem process
//! of the run: a user namespace that maps the user to itself, which gives
//! the process the right to mount there, and a mount namespace, which keeps
//! the mount from being seen outside it. Both end with the process, which
//! takes the mount with them. A process can only enter a user namespace
//! while it has no other threads, so this is done before the run makes any.
//! Where it cannot be done (another system than Linux, a kernel before 5.11
//! or one that does not let a user mount, a filesystem that cannot keep the
//! overlay's extended attributes, a path overlayfs cannot be told of), the
//! run unpacks the image into a directory of its own, as it did before.
//! `SHEM_NO_OVERLAY=1` does so too.

use std::path::{Path, PathBuf};

/// Whether the user turned overlays off.
#[cfg(target_os = "linux")]
fn disabled() -> bool {
    std::env::var_os("SHEM_NO_OVERLAY").is_some_and(|value| !value.is_empty() && value != "0")
}

/// Why an overlay cannot be had, for `shem doctor` and the log.
pub type Why = String;

/// A mounted overlay, and the directory that holds what is above the image.
/// Dropping it removes what the run wrote; the mount itself ends with the
/// process.
pub struct Mounted {
    scratch: PathBuf,
    merged: PathBuf,
}

impl Mounted {
    /// The filesystem the guest sees, with the image below and the run's
    /// changes on it.
    pub fn path(&self) -> &Path {
        &self.merged
    }
}

impl Drop for Mounted {
    fn drop(&mut self) {
        // The mount goes first: `merged` is the image with the run's changes
        // on it, which nothing is to walk through and delete from
        #[cfg(target_os = "linux")]
        if let Ok(target) = std::ffi::CString::new(self.merged.as_os_str().as_encoded_bytes()) {
            // SAFETY: a NUL-terminated path, and flags only
            unsafe { libc::umount2(target.as_ptr(), libc::MNT_DETACH) };
        }
        make_removable(&self.scratch);
        // Only what is below `scratch` that the overlay's mount is not: if
        // the mount is still there, `merged` stays, and so does `scratch`
        let _ = std::fs::remove_dir_all(self.scratch.join("upper"));
        let _ = std::fs::remove_dir_all(self.scratch.join("work"));
        if std::fs::remove_dir(&self.merged).is_ok() {
            let _ = std::fs::remove_dir(&self.scratch);
        }
    }
}

/// Let the directory a run left, `scratch`, be removed: the kernel makes
/// its work directory with no permissions at all.
pub fn make_removable(scratch: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let work = scratch.join("work/work");
        if work.is_dir() {
            let _ = std::fs::set_permissions(&work, std::fs::Permissions::from_mode(0o700));
        }
    }
    #[cfg(not(unix))]
    let _ = scratch;
}

/// Make the namespaces an overlay can be mounted in, and be in them. Where
/// this fails part of the way (a kernel that lets a process make a user
/// namespace and then refuses it the right to map its ids, as Ubuntu 24.04
/// does by default), the process is left in one it cannot use: `enter`
/// tries it in a process of its own first.
#[cfg(target_os = "linux")]
fn attempt() -> Result<(), Why> {
    // SAFETY: geteuid and getegid read the process's ids
    let (uid, gid) = unsafe { (libc::geteuid(), libc::getegid()) };
    // SAFETY: unshare takes flags only; it fails where there are other threads
    if unsafe { libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNS) } != 0 {
        return Err(format!(
            "cannot make namespaces: {}",
            std::io::Error::last_os_error()
        ));
    }
    // The user stays who they are, so the files the run makes are theirs
    let write = |file: &str, text: String| {
        std::fs::write(format!("/proc/self/{file}"), text)
            .map_err(|error| format!("cannot write {file}: {error}"))
    };
    write("setgroups", "deny".into())?;
    write("gid_map", format!("{gid} {gid} 1"))?;
    write("uid_map", format!("{uid} {uid} 1"))?;
    // What is mounted here is not shared with the host's namespace
    // SAFETY: remounting "/" with flags only, and no data
    unsafe {
        libc::mount(
            std::ptr::null(),
            c"/".as_ptr(),
            std::ptr::null(),
            libc::MS_REC | libc::MS_PRIVATE,
            std::ptr::null(),
        );
    }
    Ok(())
}

/// Run `attempt` in a process of its own, which ends with it, and say what
/// it found: the caller's namespaces stay as they are.
#[cfg(target_os = "linux")]
fn in_child(attempt: impl FnOnce() -> Result<(), Why>) -> Result<(), Why> {
    use std::io::Read;
    use std::os::fd::FromRawFd;
    let mut fds = [0; 2];
    // SAFETY: an array of two descriptors for pipe to fill in
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    // SAFETY: fork; the child makes the attempt, says what happened and ends
    // without returning to the caller's code
    match unsafe { libc::fork() } {
        -1 => Err(std::io::Error::last_os_error().to_string()),
        0 => {
            let said = match attempt() {
                Ok(()) => String::new(),
                Err(why) => why,
            };
            // SAFETY: the write end of the pipe, a buffer and its length, then exit
            unsafe {
                libc::write(fds[1], said.as_ptr().cast(), said.len());
                libc::_exit(0)
            }
        }
        _ => {
            // SAFETY: the read end of the pipe, which nothing else owns; the
            // write end is closed here so that reading ends with the child's
            let mut reader = unsafe {
                libc::close(fds[1]);
                std::fs::File::from_raw_fd(fds[0])
            };
            let mut said = String::new();
            let _ = reader.read_to_string(&mut said);
            // SAFETY: reaping the child
            unsafe { libc::wait(std::ptr::null_mut()) };
            if said.is_empty() { Ok(()) } else { Err(said) }
        }
    }
}

/// Enter the namespaces an overlay can be mounted in, once for the process.
/// Fails, and leaves the process as it was, where it cannot be done.
#[cfg(target_os = "linux")]
pub fn enter() -> Result<(), Why> {
    use std::sync::OnceLock;
    static ENTERED: OnceLock<Result<(), Why>> = OnceLock::new();
    ENTERED
        .get_or_init(|| {
            if disabled() {
                return Err("SHEM_NO_OVERLAY is set".into());
            }
            // Where a child can, this process can
            in_child(attempt)?;
            attempt()
        })
        .clone()
}

#[cfg(not(target_os = "linux"))]
pub fn enter() -> Result<(), Why> {
    Err("overlays are Linux's".into())
}

/// Mount an overlay of the unpacked image `lower`, with its own directory
/// made in `scratch_in`. Needs `enter` to have worked.
#[cfg(target_os = "linux")]
pub fn mount(lower: &Path, scratch_in: &Path) -> Result<Mounted, Why> {
    let scratch = tempfile::Builder::new()
        .prefix("overlay-")
        .tempdir_in(scratch_in)
        .map_err(|error| {
            format!(
                "cannot make a directory in {}: {error}",
                scratch_in.display()
            )
        })?
        .keep();
    let mounted = Mounted {
        merged: scratch.join("merged"),
        scratch,
    };
    for name in ["upper", "work", "merged"] {
        std::fs::create_dir(mounted.scratch.join(name)).map_err(|error| error.to_string())?;
    }
    let (upper, work) = (mounted.scratch.join("upper"), mounted.scratch.join("work"));
    // The options are a list split at commas, and the lower directories at colons
    for path in [lower, &upper, &work] {
        let text = path.to_string_lossy();
        if text.contains([',', ':', '\\', '\0']) {
            return Err(format!("overlayfs cannot be given the path {text}"));
        }
    }
    let options = format!(
        "lowerdir={},upperdir={},workdir={},userxattr",
        lower.display(),
        upper.display(),
        work.display()
    );
    let target = std::ffi::CString::new(mounted.merged.as_os_str().as_encoded_bytes())
        .map_err(|error| error.to_string())?;
    let options = std::ffi::CString::new(options).map_err(|error| error.to_string())?;
    // SAFETY: NUL-terminated strings that live through the call
    let status = unsafe {
        libc::mount(
            c"overlay".as_ptr(),
            target.as_ptr(),
            c"overlay".as_ptr(),
            0,
            options.as_ptr().cast(),
        )
    };
    if status != 0 {
        return Err(format!(
            "cannot mount an overlay: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(mounted)
}

#[cfg(not(target_os = "linux"))]
pub fn mount(_lower: &Path, _scratch_in: &Path) -> Result<Mounted, Why> {
    Err("overlays are Linux's".into())
}

/// Whether an overlay can be mounted here, found out in a process of its own
/// so that the caller's namespaces stay as they are, and why not if it
/// cannot: what `shem doctor` reports. `scratch_in` is where the probe's
/// directories go, on the filesystem a run's would.
#[cfg(target_os = "linux")]
pub fn probe(scratch_in: &Path) -> Result<(), Why> {
    if disabled() {
        return Err("SHEM_NO_OVERLAY is set".into());
    }
    let lower = tempfile::tempdir_in(scratch_in).map_err(|error| error.to_string())?;
    in_child(|| attempt().and_then(|()| mount(lower.path(), scratch_in).map(drop)))
}

#[cfg(not(target_os = "linux"))]
pub fn probe(_scratch_in: &Path) -> Result<(), Why> {
    Err("overlays are Linux's".into())
}
