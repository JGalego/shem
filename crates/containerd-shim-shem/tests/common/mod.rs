//! RV32 programs and OCI bundles for the shim's tests.
#![allow(dead_code)]

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::sync::{Arc, Mutex};

pub const BASE: u32 = 0x1_0000;

/// A static RV32 executable: `code` after `data`, in one segment.
pub fn elf(code: &[u32], data: &[u8]) -> Vec<u8> {
    let mut body = data.to_vec();
    body.resize(data.len().next_multiple_of(4), 0); // keep the entry point aligned
    let entry = BASE + 84 + body.len() as u32;
    body.extend(code.iter().flat_map(|w| w.to_le_bytes()));
    let size = 84 + body.len() as u32;
    let mut out = b"\x7fELF\x01\x01\x01".to_vec();
    out.resize(16, 0);
    out.extend([2u16, 243].iter().flat_map(|h| h.to_le_bytes()));
    out.extend(
        [1, entry, 52, 0, 0]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes()),
    );
    out.extend(
        [52u16, 32, 1, 40, 0, 0]
            .iter()
            .flat_map(|h| h.to_le_bytes()),
    );
    out.extend(
        [1, 0, BASE, BASE, size, size, 7, 0x1000]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes()),
    );
    out.extend(body);
    out
}

pub fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    (imm as u32 & 0xfff) << 20 | rs1 << 15 | rd << 7 | 0x13
}

pub fn lui(rd: u32, upper: u32) -> u32 {
    upper & 0xffff_f000 | rd << 7 | 0x37
}

pub const ECALL: u32 = 0x73;

#[derive(Clone, Default)]
pub struct Captured(pub Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn bundle_with(program: Vec<u8>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let config = r#"{
        "process": {"args": ["/bin/hello", "x"], "env": ["PATH=/bin"], "cwd": "/"},
        "root": {"path": "rootfs"},
        "linux": {"resources": {"memory": {"limit": 8388608}}}
    }"#;
    std::fs::write(dir.path().join("config.json"), config).unwrap();
    let bin = dir.path().join("rootfs/bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(bin.join("hello"), program).unwrap();
    std::fs::set_permissions(bin.join("hello"), std::fs::Permissions::from_mode(0o755)).unwrap();
    dir
}

/// `lui` and `addi` that load `value` into `rd`.
pub fn li(rd: u32, value: u32) -> [u32; 2] {
    [
        lui(rd, value + 0x800),
        addi(rd, rd, ((value & 0xfff) as i32 ^ 0x800) - 0x800),
    ]
}

/// A program that writes "ready", echoes one read of stdin, and exits.
pub fn echo() -> Vec<u8> {
    let data_at = BASE + 84;
    let buf = data_at + 8;
    let mut data = b"ready\n".to_vec();
    data.resize(8 + 64, 0);
    let mut code = vec![addi(10, 0, 1)]; // write(1, "ready\n", 6)
    code.extend(li(11, data_at));
    code.extend([addi(12, 0, 6), addi(17, 0, 64), ECALL]);
    code.push(addi(10, 0, 0)); // n = read(0, buf, 64)
    code.extend(li(11, buf));
    code.extend([addi(12, 0, 64), addi(17, 0, 63), ECALL]);
    code.extend([addi(12, 10, 0), addi(10, 0, 1)]); // write(1, buf, n)
    code.extend(li(11, buf));
    code.extend([addi(17, 0, 64), ECALL]);
    code.extend([addi(10, 0, 0), addi(17, 0, 93), ECALL]); // exit(0)
    elf(&code, &data)
}
