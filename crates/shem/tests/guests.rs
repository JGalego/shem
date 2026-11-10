//! Guest programs built with the toolchain image: the official riscv-tests ISA
//! suites, and C programs linked against musl. Building them needs Docker, so
//! these tests are ignored by default; `just guest-tests` builds and runs them.

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
    let elf = std::fs::read(path)
        .unwrap_or_else(|err| panic!("{}: {err} (run `just guest-tests`)", path.display()));
    let (mut container, stdout) = start(&elf, argv, &[], MEM).unwrap();
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
