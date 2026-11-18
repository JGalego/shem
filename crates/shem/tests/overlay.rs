//! A run's filesystem as an overlay of the image's unpacked copy: the guest
//! reads the image's files and changes them, and the copy the next run
//! starts from does not change. Run through the `shem` binary, since a
//! process can only enter the namespaces an overlay is mounted in while it
//! has no other threads, which the tests' own harness has.

mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Output};

use common::*;
use shem::image::{self, BuildOptions, Store};

const AT_FDCWD: u32 = -100i32 as u32;
const O_WRONLY: i32 = 1;
const O_CREAT: i32 = 0o100;
const O_TRUNC: i32 = 0o1000;
const MEM: u32 = 1 << 20;
/// `K`, and what the guest's changes add to it when they all work
const STATUS: i32 = b'K' as i32 + 7 + 5;

/// What the guest does to its filesystem: reads the first byte of `/keep`,
/// deletes `/gone`, truncates `/mod` and writes `changed` in it, creates
/// `/new` holding `fresh`, and exits with the byte it read and what the
/// calls that changed things returned (0, 7 and 5 when they did), which
/// make a status of `STATUS`.
fn guest() -> Vec<u8> {
    let mut data = Vec::new();
    let mut put = |text: &[u8]| {
        let at = DATA + data.len() as u32;
        data.extend(text);
        data.push(0);
        data.resize(data.len().next_multiple_of(8), 0);
        at
    };
    let (keep, gone, modified) = (put(b"/keep"), put(b"/gone"), put(b"/mod"));
    let (changed, new, fresh) = (put(b"changed"), put(b"/new"), put(b"fresh"));
    let buf = put(b"");
    data.resize(data.len() + 8, 0);
    let open = |path: u32, flags: i32| {
        program![
            li(A0, AT_FDCWD),
            li(A1, path),
            li(A2, flags as u32),
            addi(A3, ZERO, 0o644),
            addi(A7, ZERO, 56),
            ECALL,
        ]
    };
    let write = |text: u32, len: i32| {
        // The descriptor `open` returned is still in a0
        program![
            addi(T1, A0, 0),
            li(A1, text),
            addi(A2, ZERO, len),
            addi(A7, ZERO, 64),
            ECALL,
            // The bytes written: an error would change the status
            add(S0, S0, A0),
            // Nothing is left open, which a WARM could not freeze
            addi(A0, T1, 0),
            addi(A7, ZERO, 57),
            ECALL,
        ]
    };
    let code = [
        open(keep, 0),
        program![
            addi(T2, A0, 0),
            li(A1, buf),
            addi(A2, ZERO, 1),
            addi(A7, ZERO, 63),
            ECALL,
            li(T0, buf),
            lbu(S0, T0, 0),
            addi(A0, T2, 0),
            addi(A7, ZERO, 57),
            ECALL,
            // unlinkat(AT_FDCWD, "/gone", 0)
            li(A0, AT_FDCWD),
            li(A1, gone),
            addi(A2, ZERO, 0),
            addi(A7, ZERO, 35),
            ECALL,
            // 0 if it was deleted
            add(S0, S0, A0),
        ],
        open(modified, O_WRONLY | O_TRUNC),
        write(changed, 7),
        open(new, O_WRONLY | O_CREAT),
        write(fresh, 5),
        program![addi(A0, S0, 0), addi(A7, ZERO, 93), ECALL],
    ]
    .concat();
    elf(&code, &data)
}

/// A store in `home` holding the image `guest` runs as, with the files it works on.
fn installed(home: &Path) -> (String, std::path::PathBuf) {
    installed_as(home, "prog")
}

/// As `installed`, under the name `name`.
fn installed_as(home: &Path, name: &str) -> (String, std::path::PathBuf) {
    let context = tempfile::tempdir().unwrap();
    let files: [(&str, Vec<u8>); 4] = [
        ("prog", guest()),
        ("keep", b"Keep".to_vec()),
        ("gone", b"Gone".to_vec()),
        ("mod", b"Original contents".to_vec()),
    ];
    for (name, data) in &files {
        let path = context.path().join(name);
        fs::write(&path, data).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    fs::write(
        context.path().join("Shemfile"),
        "FROM scratch\nCOPY prog /bin/prog\nCOPY keep /keep\nCOPY gone /gone\nCOPY mod /mod\nENTRYPOINT [\"/bin/prog\"]\n",
    )
    .unwrap();
    let image = image::build(&BuildOptions {
        shemfile: context.path().join("Shemfile"),
        context: context.path().to_path_buf(),
        build_args: Default::default(),
        mem: Some(MEM),
        home: Some(tempfile::tempdir().unwrap().keep()),
    })
    .unwrap();
    let store = Store::at(home);
    let digest = image
        .write_layout(&store.path(name).unwrap(), "latest")
        .unwrap();
    (
        digest.clone(),
        home.join("rootfs").join(&digest["sha256:".len()..]),
    )
}

fn run(home: &Path, overlay: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_shem"));
    command.args(["run", "prog"]).env("SHEM_HOME", home);
    if !overlay {
        command.env("SHEM_NO_OVERLAY", "1");
    }
    command.output().unwrap()
}

#[test]
fn a_guest_changes_its_filesystem_and_the_image_does_not() {
    let home = tempfile::tempdir().unwrap();
    let (_, unpacked) = installed(home.path());
    for turn in 0..3 {
        let output = run(home.path(), true);
        assert_eq!(
            output.status.code(),
            Some(STATUS),
            "run {turn}: the guest read the image's /keep: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    // Where the system mounts overlays, runs were overlays of the unpacked image
    let doctor = Command::new(env!("CARGO_BIN_EXE_shem"))
        .arg("doctor")
        .env("SHEM_HOME", home.path())
        .output()
        .unwrap();
    let overlays = String::from_utf8_lossy(&doctor.stdout).contains("ok    overlayfs");
    assert_eq!(
        unpacked.is_dir(),
        overlays,
        "the image's copy is kept for overlays"
    );
    // and that copy is what each started from, still
    if unpacked.is_dir() {
        assert_eq!(fs::read(unpacked.join("keep")).unwrap(), b"Keep");
        assert_eq!(fs::read(unpacked.join("gone")).unwrap(), b"Gone");
        assert_eq!(
            fs::read(unpacked.join("mod")).unwrap(),
            b"Original contents"
        );
        assert!(!unpacked.join("new").exists());
    }
    // Nothing a run made is left behind
    let scratch = home.path().join("tmp");
    let left = fs::read_dir(&scratch).map_or(0, Iterator::count);
    assert_eq!(left, 0, "the runs' own directories are removed");
}

#[test]
fn a_run_with_the_overlay_does_what_one_without_does() {
    let home = tempfile::tempdir().unwrap();
    let (_, unpacked) = installed(home.path());
    let without = run(home.path(), false);
    assert_eq!(without.status.code(), Some(STATUS));
    // Turned off, a run does not keep a copy of the image: it unpacks one for itself
    assert!(!unpacked.exists());
    let with = run(home.path(), true);
    assert_eq!(with.status.code(), without.status.code());
    assert_eq!(with.stdout, without.stdout);
}

/// `shem build` of `context` in the store at `home`: the digest it says it made.
fn build(home: &Path, context: &Path, overlay: bool) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_shem"));
    command
        .args(["build", "-t", "made"])
        .arg(context)
        .env("SHEM_HOME", home);
    if !overlay {
        command.env("SHEM_NO_OVERLAY", "1");
    }
    let output = command.output().unwrap();
    let said = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "{said}");
    said.split_whitespace()
        .find(|word| word.starts_with("sha256:"))
        .unwrap_or_else(|| panic!("{said}"))
        .to_string()
}

#[test]
fn a_build_on_an_overlay_of_its_base_makes_the_image_one_that_unpacks_it_does() {
    let base = "example.com/team/prog";
    let context = tempfile::tempdir().unwrap();
    fs::write(context.path().join("extra"), b"extra file").unwrap();
    fs::write(
        context.path().join("Shemfile"),
        format!(
            "FROM {base}:latest\nCOPY extra /extra\nENV BUILT=yes\nWARM --until=syscall:exit\n"
        ),
    )
    .unwrap();
    let (with, without) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (_, unpacked) = installed_as(with.path(), base);
    installed_as(without.path(), base);
    // The guest deletes, rewrites and creates files while the build warms it: the
    // layer that makes holds what changed, and is the same whichever way the base is there
    let on = build(with.path(), context.path(), true);
    let off = build(without.path(), context.path(), false);
    assert_eq!(on, off, "the same image, however the base's files were");
    // Where the build was on an overlay, the unpacked base is what it began with, still
    if unpacked.is_dir() {
        assert_eq!(fs::read(unpacked.join("gone")).unwrap(), b"Gone");
        assert_eq!(
            fs::read(unpacked.join("mod")).unwrap(),
            b"Original contents"
        );
        assert!(!unpacked.join("new").exists());
        assert!(!unpacked.join("extra").exists());
    }
    let left = fs::read_dir(with.path().join("tmp")).map_or(0, Iterator::count);
    assert_eq!(left, 0, "the build's own directory is removed");
}

/// A directory holding the files `guest` works on, and the program itself beside it.
fn directory() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("root");
    fs::create_dir(&root).unwrap();
    for (name, data) in [
        ("keep", &b"Keep"[..]),
        ("gone", b"Gone"),
        ("mod", b"Original contents"),
    ] {
        fs::write(root.join(name), data).unwrap();
    }
    let program = dir.path().join("prog");
    fs::write(&program, guest()).unwrap();
    fs::set_permissions(&program, fs::Permissions::from_mode(0o755)).unwrap();
    (dir, root)
}

fn run_in(home: &Path, root: &Path, program: &Path, overlay: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_shem"));
    command
        .arg("run")
        .arg("--root")
        .arg(root)
        .args(overlay.then_some("--root-overlay"))
        .args(["--mem", "1M"])
        .arg(program)
        .env("SHEM_HOME", home);
    command.output().unwrap()
}

#[test]
fn a_root_given_to_an_executable_is_changed_unless_it_is_an_overlay() {
    let home = tempfile::tempdir().unwrap();
    let (dir, root) = directory();
    let program = dir.path().join("prog");
    // Without the flag, the guest's changes are made to the directory itself
    let plain = run_in(home.path(), &root, &program, false);
    assert_eq!(plain.status.code(), Some(STATUS), "{plain:?}");
    assert!(!root.join("gone").exists());
    assert_eq!(fs::read(root.join("new")).unwrap(), b"fresh");
    // With it, they are made to a layer above, where the system mounts overlays
    let (dir, root) = directory();
    let program = dir.path().join("prog");
    let layered = run_in(home.path(), &root, &program, true);
    assert_eq!(layered.status.code(), Some(STATUS), "{layered:?}");
    let said = String::from_utf8_lossy(&layered.stderr);
    if !said.contains("no overlay of") {
        assert_eq!(fs::read(root.join("gone")).unwrap(), b"Gone");
        assert_eq!(fs::read(root.join("mod")).unwrap(), b"Original contents");
        assert!(!root.join("new").exists());
    }
    let left = fs::read_dir(home.path().join("tmp")).map_or(0, Iterator::count);
    assert_eq!(left, 0, "the run's own directory is removed");
}
