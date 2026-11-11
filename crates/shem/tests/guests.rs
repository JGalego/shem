//! Guest programs built with the toolchain image: the official riscv-tests ISA
//! suites, C programs linked against musl, and BusyBox. Building them needs
//! Docker, so these tests are ignored by default; `just guest-tests` builds and
//! runs them.

mod common;

use std::path::{Path, PathBuf};

use common::start;
use shem::container::RunError;

const MEM: u32 = 16 << 20;
const MAX_STEPS: u64 = 100_000_000;

fn built(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(path)
}

struct Run {
    result: Result<u8, RunError>,
    stdout: String,
    unsupported: Vec<u32>,
}

fn run(path: &Path, argv: &[&str]) -> Run {
    run_in(None, path, argv)
}

/// Run `path` with `root`, if given, as the guest's root filesystem.
fn run_in(root: Option<&Path>, path: &Path, argv: &[&str]) -> Run {
    let elf = std::fs::read(path)
        .unwrap_or_else(|err| panic!("{}: {err} (run `just guest-tests`)", path.display()));
    let (mut container, stdout) = start(&elf, argv, &[], MEM).unwrap();
    if let Some(root) = root {
        container.set_root(root).unwrap();
    }
    let result = container.run(Some(MAX_STEPS));
    Run {
        result,
        stdout: String::from_utf8_lossy(&stdout.contents()).into_owned(),
        unsupported: container.linux.unsupported.keys().copied().collect(),
    }
}

#[test]
#[ignore = "needs guest binaries: run `just guest-tests`"]
fn riscv_tests_pass() {
    let dir = built("riscv-tests");
    let mut tests: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("{}: {err} (run `just guest-tests`)", dir.display()))
        .map(|entry| entry.unwrap().path())
        .collect();
    tests.sort();
    assert!(
        tests.len() >= 60,
        "expected the rv32ui, rv32um and rv32ua suites"
    );

    let failures: Vec<String> = tests
        .iter()
        .filter_map(|path| {
            let name = path.file_name().unwrap().to_string_lossy();
            match run(path, &[&name]).result {
                Ok(0) => None,
                // The test environment exits with (case << 1) | 1 on failure.
                Ok(status) => Some(format!("{name}: case {} failed", status >> 1)),
                Err(err) => Some(format!("{name}: {err}")),
            }
        })
        .collect();
    assert!(
        failures.is_empty(),
        "riscv-tests failures:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs guest binaries: run `just guest-tests`"]
fn c_program_runs_against_musl() {
    let out = run(&built("guests/hello"), &["hello", "one", "two"]);
    assert_eq!(out.result, Ok(3));
    assert_eq!(
        out.stdout,
        "argv[0] = hello\nargv[1] = one\nargv[2] = two\n\
         sum of squares below 100000 = 333328333350000\n"
    );
    assert_eq!(out.unsupported, [] as [u32; 0], "unsupported system calls");
}

#[test]
#[ignore = "needs guest binaries: run `just guest-tests`"]
fn busybox_applets_run_in_a_root_filesystem() {
    let busybox = built("busybox/bin/busybox");
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("fruit.txt"), "banana\napple\ncherry\n").unwrap();

    let cases: &[(&[&str], &str)] = &[
        (&["sort", "/fruit.txt"], "apple\nbanana\ncherry\n"),
        (&["grep", "-c", "an", "/fruit.txt"], "1\n"),
        (
            &["sha256sum", "/fruit.txt"],
            "64112a2c204881f4aac7da9ffd84a2b0412a193ae9b3773cbab04ff947d2b92c  /fruit.txt\n",
        ),
        (
            &["awk", "{ n += length($0) } END { print n }", "/fruit.txt"],
            "17\n",
        ),
        (&["sh", "-c", "x=$((6 * 7)); echo answer=$x"], "answer=42\n"),
        (&["mkdir", "-p", "/tmp/a/b"], ""),
        (&["cp", "/fruit.txt", "/tmp/a/b/copy"], ""),
        (&["touch", "/tmp/a/f"], ""),
        (
            &["find", "/tmp"],
            "/tmp\n/tmp/a\n/tmp/a/b\n/tmp/a/b/copy\n/tmp/a/f\n",
        ),
        (&["cat", "/tmp/a/b/copy"], "banana\napple\ncherry\n"),
        (&["date", "+%Y-%m-%d"], "2026-01-01\n"),
        (&["uname", "-m"], "riscv32\n"),
    ];
    for (args, expected) in cases {
        let argv: Vec<&str> = std::iter::once("busybox")
            .chain(args.iter().copied())
            .collect();
        let out = run_in(Some(root.path()), &busybox, &argv);
        assert_eq!(out.result, Ok(0), "busybox {args:?}");
        assert_eq!(out.stdout, *expected, "busybox {args:?}");
        assert_eq!(
            out.unsupported,
            [] as [u32; 0],
            "busybox {args:?}: unsupported system calls"
        );
    }
}

/// Copy BusyBox's installed rootfs, symlinks and all, into a fresh directory.
fn busybox_root() -> tempfile::TempDir {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            let kind = entry.file_type().unwrap();
            if kind.is_symlink() {
                let link = std::fs::read_link(entry.path()).unwrap();
                std::os::unix::fs::symlink(link, &target).unwrap();
            } else if kind.is_dir() {
                copy(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).unwrap();
            }
        }
    }
    let root = tempfile::tempdir().unwrap();
    copy(&built("busybox"), root.path());
    root
}

#[test]
#[ignore = "needs guest binaries: run `just guest-tests`"]
fn busybox_shell_runs_pipelines_jobs_and_traps() {
    let root = busybox_root();
    let busybox = root.path().join("bin/busybox");
    let cases: &[(&str, &str)] = &[
        ("echo piped | tr a-z A-Z", "PIPED\n"),
        ("seq 1 5 | sort -r | head -3", "5\n4\n3\n"),
        ("yes | head -2", "y\ny\n"),
        ("sh -c 'exit 7'; echo $?", "7\n"),
        ("x=$(echo captured); echo $x", "captured\n"),
        ("for i in 1 2 3; do echo $i; done | xargs echo", "1 2 3\n"),
        (
            "mkdir -p /tmp && echo hello > /tmp/out && cat /tmp/out",
            "hello\n",
        ),
        ("sleep 1 & wait; echo done", "done\n"),
        (
            "trap 'echo caught' TERM; kill $$; echo after",
            "caught\nafter\n",
        ),
        ("trap 'echo bye' EXIT; echo body", "body\nbye\n"),
    ];
    for (script, expected) in cases {
        let out = run_in(Some(root.path()), &busybox, &["sh", "-c", script]);
        assert_eq!(out.result, Ok(0), "sh -c {script:?}");
        assert_eq!(out.stdout, *expected, "sh -c {script:?}");
        assert_eq!(
            out.unsupported,
            [] as [u32; 0],
            "sh -c {script:?}: unsupported system calls"
        );
    }
    let killed = run_in(
        Some(root.path()),
        &busybox,
        &["sh", "-c", "kill $$; echo survived"],
    );
    assert_eq!(killed.result, Ok(128 + 15), "killed by SIGTERM");
    assert_eq!(killed.stdout, "");
}

fn build_example(name: &str) -> shem::image::Image {
    let context = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name);
    shem::image::build(&shem::image::BuildOptions {
        shemfile: context.join("Shemfile"),
        context,
        build_args: Default::default(),
        mem: MEM,
    })
    .unwrap_or_else(|err| panic!("building examples/{name}: {err:#} (run `just guest-tests`)"))
}

fn run_image(image: &shem::image::Image, args: &[&str], stdin: &[u8]) -> (u8, String) {
    let root = tempfile::tempdir().unwrap();
    let stdout = common::SharedBuf::default();
    let stdio = shem::linux::Stdio {
        stdin: Box::new(std::io::Cursor::new(stdin.to_vec())),
        stdout: Box::new(stdout.clone()),
        stderr: Box::new(std::io::sink()),
    };
    let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
    let make = |mem| Ok(shem::reference::ReferenceVcpu::new(mem));
    let mut container =
        shem::image::start(image, make, &args, &[], stdio, root.path(), MEM).unwrap();
    let status = container.run(Some(MAX_STEPS)).unwrap();
    (
        status,
        String::from_utf8_lossy(&stdout.contents()).into_owned(),
    )
}

#[test]
#[ignore = "needs Docker and the toolchain images: run `just guest-tests`"]
fn example_images_build_and_run() {
    let hello = build_example("hello");
    assert_eq!(
        run_image(&hello, &[], b""),
        (0, "hello from a shem image\n".into())
    );
    assert_eq!(
        run_image(&hello, &["again"], b""),
        (0, "hello again\n".into())
    );

    let shell = build_example("warm-shell");
    assert!(shell.warm_state().unwrap().is_some());
    let script = b"echo hello from a warm shell\nuname -m\necho $((6 * 7))\n";
    let expected = "hello from a warm shell\nriscv32\n42\n";
    assert_eq!(run_image(&shell, &[], script), (0, expected.into()));
}
