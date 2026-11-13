//! Seeds for the fuzz targets in `fuzz/`: a small, valid input of each kind
//! for the fuzzer to start from. Write them again when a format changes:
//!
//!     cargo test -p shem --test fuzz_seeds -- --ignored

mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use common::*;
use shem::container::{Pause, Stop};
use shem::image::{self, BuildOptions};
use shem::vcpu::MachineState;

const MEM: u32 = 1 << 20;

/// write(1, "ready\n"), then echo one read of stdin, then exit(0)
fn echo() -> Vec<u8> {
    let mut data = b"ready\n".to_vec();
    data.resize(16 + 64, 0);
    let buf = DATA + 16;
    let code = program![
        addi(A0, ZERO, 1),
        li(A1, DATA),
        addi(A2, ZERO, 6),
        addi(A7, ZERO, 64),
        ECALL,
        addi(A0, ZERO, 0),
        li(A1, buf),
        addi(A2, ZERO, 64),
        addi(A7, ZERO, 63),
        ECALL,
        addi(A2, A0, 0),
        addi(A0, ZERO, 1),
        li(A1, buf),
        addi(A7, ZERO, 64),
        ECALL,
        addi(A0, ZERO, 0),
        addi(A7, ZERO, 93),
        ECALL,
    ];
    elf(&code, &data)
}

#[test]
#[ignore = "writes fuzz/seeds, when a format changes"]
fn write_fuzz_seeds() {
    let seeds = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/seeds");
    let write = |target: &str, name: &str, data: &[u8]| {
        let dir = seeds.join(target);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(name), data).unwrap();
    };
    let program = echo();
    write("elf", "echo", &program);

    // The echo program frozen at its read: its host state, and its machine
    // state with a little of its memory
    let (mut container, _) = start(&program, &["echo"], b"", MEM).unwrap();
    let paused = container.run_until(Some(10_000), Some(Pause::Syscall(63)));
    assert_eq!(paused, Ok(Stop::Paused));
    let frozen = container.freeze().unwrap();
    write(
        "frozen",
        "echo.json",
        &serde_json::to_vec(&frozen.host).unwrap(),
    );
    let small = MachineState {
        mem: frozen.machine.mem[..4096].to_vec(),
        ..frozen.machine.clone()
    };
    write("machine", "small.state", &small.encode());

    // The same with its counts near their limits, as an image's own state
    // files may have them
    let mut host = frozen.host.clone();
    host.steps = u64::MAX - 1;
    host.process.cpu_ns = u64::MAX - 1;
    host.linux.slept_ns = u64::MAX - 1;
    write(
        "frozen",
        "near-limits.json",
        &serde_json::to_vec(&host).unwrap(),
    );
    let near_limits = MachineState {
        instret: u64::MAX - 1,
        ..small.clone()
    };
    write("machine", "near-limits.state", &near_limits.encode());

    // A warm image of it, as `shem save` writes it for `docker load`
    let context = tempfile::tempdir().unwrap();
    let path = context.path().join("echo");
    fs::write(&path, &program).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    let shemfile = "FROM scratch\nCOPY echo /bin/echo\nENTRYPOINT [\"/bin/echo\"]\nWARM --until=syscall:read\n";
    fs::write(context.path().join("Shemfile"), shemfile).unwrap();
    let warm = image::build(&BuildOptions {
        shemfile: context.path().join("Shemfile"),
        context: context.path().to_path_buf(),
        build_args: Default::default(),
        mem: Some(MEM),
    })
    .unwrap();
    write(
        "archive",
        "echo.tar",
        &warm.docker_archive("echo", &[]).unwrap(),
    );

    // A layer with a directory, a file in it, a link and a whiteout
    let mut tar = tar::Builder::new(Vec::new());
    let mut entry = |kind: tar::EntryType, name: &str, data: &[u8], link: Option<&str>| {
        let mut header = tar::Header::new_ustar();
        header.set_entry_type(kind);
        header.set_size(data.len() as u64);
        header.set_mode(if kind.is_dir() { 0o755 } else { 0o644 });
        header.set_mtime(0);
        match link {
            Some(target) => tar.append_link(&mut header, name, target).unwrap(),
            None => tar.append_data(&mut header, name, data).unwrap(),
        }
    };
    entry(tar::EntryType::Directory, "etc/", b"", None);
    entry(tar::EntryType::Regular, "etc/motd", b"hello\n", None);
    entry(tar::EntryType::Symlink, "motd", b"", Some("etc/motd"));
    entry(tar::EntryType::Regular, "etc/.wh.issue", b"", None);
    write("layer", "layer.tar", &tar.into_inner().unwrap());
}
