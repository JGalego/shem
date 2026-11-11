use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

use containerd_shim_shem::bundle::{self, Bundle};
use shem::linux::Stdio;

const BASE: u32 = 0x1_0000;

/// A static RV32 executable: `code` after `data`, in one segment.
fn elf(code: &[u32], data: &[u8]) -> Vec<u8> {
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

fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    (imm as u32 & 0xfff) << 20 | rs1 << 15 | rd << 7 | 0x13
}

fn lui(rd: u32, upper: u32) -> u32 {
    upper & 0xffff_f000 | rd << 7 | 0x37
}

const ECALL: u32 = 0x73;

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn bundle_with(program: Vec<u8>) -> tempfile::TempDir {
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

#[test]
fn bundles_run_their_process_on_shem() {
    let msg = b"hello from containerd\n";
    let data_at = BASE + 84;
    let code = [
        addi(10, 0, 1), // write(1, msg, len)
        lui(11, data_at + 0x800),
        addi(11, 11, ((data_at & 0xfff) as i32 ^ 0x800) - 0x800),
        addi(12, 0, msg.len() as i32),
        addi(17, 0, 64),
        ECALL,
        addi(10, 0, 3), // exit(3)
        addi(17, 0, 93),
        ECALL,
    ];
    let dir = bundle_with(elf(&code, msg));
    let bundle = Bundle::load(dir.path()).unwrap();
    assert_eq!(bundle.args, ["/bin/hello", "x"]);
    assert_eq!(bundle.memory, 8 << 20);
    let out = Captured::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(out.clone()),
        stderr: Box::new(std::io::sink()),
    };
    assert_eq!(bundle::run(&bundle, stdio, &AtomicU32::new(0)).unwrap(), 3);
    assert_eq!(out.0.lock().unwrap().as_slice(), msg);
}

#[test]
fn a_kill_stops_a_running_process() {
    let spin = [0x0000_006f]; // jal zero, 0: loop forever
    let dir = bundle_with(elf(&spin, &[]));
    let bundle = Bundle::load(dir.path()).unwrap();
    let stdio = Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(std::io::sink()),
        stderr: Box::new(std::io::sink()),
    };
    assert_eq!(
        bundle::run(&bundle, stdio, &AtomicU32::new(15)).unwrap(),
        128 + 15
    );
}

/// A program that writes `msg` and exits with status 3.
fn greeter(msg: &[u8]) -> Vec<u8> {
    let data_at = BASE + 84;
    let code = [
        addi(10, 0, 1), // write(1, msg, len)
        lui(11, data_at + 0x800),
        addi(11, 11, ((data_at & 0xfff) as i32 ^ 0x800) - 0x800),
        addi(12, 0, msg.len() as i32),
        addi(17, 0, 64),
        ECALL,
        addi(10, 0, 3), // exit(3)
        addi(17, 0, 93),
        ECALL,
    ];
    elf(&code, msg)
}

#[test]
fn annotations_choose_the_vcpu() {
    use containerd_shim_shem::bundle::{DEFAULT_NEURAL_SOCKET, VcpuKind};

    let dir = bundle_with(greeter(b"hi\n"));
    assert_eq!(Bundle::load(dir.path()).unwrap().vcpu, VcpuKind::Reference);
    let config = |annotations: &str| {
        let text = format!(
            r#"{{"process": {{"args": ["/bin/hello"]}}, "annotations": {{{annotations}}}}}"#
        );
        std::fs::write(dir.path().join("config.json"), text).unwrap();
        Bundle::load(dir.path())
    };
    let neural = config(r#""dev.shem.vcpu": "neural""#).unwrap();
    assert_eq!(neural.vcpu, VcpuKind::Neural(DEFAULT_NEURAL_SOCKET.into()));
    let elsewhere = config(r#""dev.shem.vcpu": "neural", "dev.shem.neural-socket": "/run/m.sock""#);
    assert_eq!(
        elsewhere.unwrap().vcpu,
        VcpuKind::Neural("/run/m.sock".into())
    );
    assert!(config(r#""dev.shem.vcpu": "gpu""#).is_err());
}

#[test]
fn warm_images_resume_from_their_state_in_the_root() {
    use shem::container::{Container, Pause, Stop};
    use shem::reference::ReferenceVcpu;

    let msg = b"resumed\n";
    let dir = bundle_with(greeter(msg));
    // Freeze the program just before its write, as WARM would
    let program = greeter(msg);
    let quiet = Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(std::io::sink()),
        stderr: Box::new(std::io::sink()),
    };
    let argv = ["/bin/hello".to_string()];
    let mut container =
        Container::from_elf(ReferenceVcpu::new(1 << 20), &program, &argv, &[], quiet).unwrap();
    assert_eq!(
        container.run_until(None, Some(Pause::Syscall(64))).unwrap(),
        Stop::Paused
    );
    let frozen = container.freeze().unwrap();
    let warm = dir.path().join("rootfs/.shem");
    std::fs::create_dir(&warm).unwrap();
    std::fs::write(
        warm.join("container.json"),
        serde_json::to_vec(&frozen.host).unwrap(),
    )
    .unwrap();
    std::fs::write(warm.join("machine.state"), frozen.machine.encode()).unwrap();

    let bundle = Bundle::load(dir.path()).unwrap();
    let out = Captured::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(out.clone()),
        stderr: Box::new(std::io::sink()),
    };
    assert_eq!(bundle::run(&bundle, stdio, &AtomicU32::new(0)).unwrap(), 3);
    assert_eq!(out.0.lock().unwrap().as_slice(), msg);
    assert!(!warm.exists(), "the guest never saw its own state");
}

#[test]
#[ignore = "needs a neural vCPU server: set SHEM_NEURAL_SOCKET to its socket"]
fn annotated_bundles_run_inside_the_model() {
    let socket = std::env::var("SHEM_NEURAL_SOCKET").expect("SHEM_NEURAL_SOCKET");
    let msg = b"hello from inside the model\n";
    let dir = bundle_with(greeter(msg));
    let config = format!(
        r#"{{"process": {{"args": ["/bin/hello"]}},
            "linux": {{"resources": {{"memory": {{"limit": 1048576}}}}}},
            "annotations": {{"dev.shem.vcpu": "neural", "dev.shem.neural-socket": "{socket}"}}}}"#
    );
    std::fs::write(dir.path().join("config.json"), config).unwrap();
    let bundle = Bundle::load(dir.path()).unwrap();
    let out = Captured::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(out.clone()),
        stderr: Box::new(std::io::sink()),
    };
    assert_eq!(bundle::run(&bundle, stdio, &AtomicU32::new(0)).unwrap(), 3);
    assert_eq!(out.0.lock().unwrap().as_slice(), msg);
}

#[test]
#[ignore = "needs a model server and an unpacked model image: set SHEM_NEURAL_SOCKET and SHEM_MODEL_ROOT"]
fn model_images_run_on_the_model_server() {
    let socket = std::env::var("SHEM_NEURAL_SOCKET").expect("SHEM_NEURAL_SOCKET");
    let root = std::env::var("SHEM_MODEL_ROOT").expect("SHEM_MODEL_ROOT, the root of an unpacked model image");
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "process": {"args": ["shem:generate", "--max-tokens", "12", "Say hi in Portuguese."]},
        "root": {"path": root},
        "annotations": {"dev.shem.vcpu": "neural", "dev.shem.neural-socket": socket},
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let bundle = Bundle::load(dir.path()).unwrap();
    let out = Captured::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(out.clone()),
        stderr: Box::new(std::io::sink()),
    };
    assert_eq!(bundle::run(&bundle, stdio, &AtomicU32::new(0)).unwrap(), 0);
    let reply = String::from_utf8(out.0.lock().unwrap().clone()).unwrap();
    assert!(reply.contains("Ol"), "{reply}");
}
