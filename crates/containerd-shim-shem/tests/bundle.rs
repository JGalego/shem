mod common;

use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use common::*;
use containerd_shim_shem::bundle::{self, Bundle, Control, Wakeable};
use shem::linux::Stdio;

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
    assert_eq!(
        bundle::run(&bundle, stdio, &Control::default(), None).unwrap(),
        3
    );
    assert_eq!(out.0.lock().unwrap().as_slice(), msg);
}

/// A control that has been asked to stop the program with `signal`.
fn killed(signal: u32) -> Control {
    let control = Control::default();
    control.kill(signal);
    control
}

/// Start the echo program in `dir` on a thread, its stdin a pipe that a
/// kill or a pause wakes, and wait until it has said it is ready.
fn start_echo(
    dir: &std::path::Path,
) -> (
    Arc<Control>,
    std::io::PipeWriter,
    Captured,
    std::thread::JoinHandle<u8>,
) {
    let bundle = Bundle::load(dir).unwrap();
    let control = Arc::new(Control::default());
    let (reader, writer) = std::io::pipe().unwrap();
    let out = Captured::default();
    let stdio = Stdio {
        stdin: Box::new(Wakeable::new(reader, control.clone())),
        stdout: Box::new(out.clone()),
        stderr: Box::new(std::io::sink()),
    };
    let run = {
        let control = control.clone();
        std::thread::spawn(move || bundle::run(&bundle, stdio, &control, None).unwrap())
    };
    while out.0.lock().unwrap().as_slice() != b"ready\n" {
        std::thread::sleep(Duration::from_millis(10));
    }
    (control, writer, out, run)
}

#[test]
fn a_kill_stops_a_program_waiting_for_input() {
    let dir = bundle_with(echo());
    let (control, _stdin, _, run) = start_echo(dir.path());
    std::thread::sleep(Duration::from_millis(200)); // well into its read
    control.kill(15);
    assert_eq!(run.join().unwrap(), 128 + 15);
}

#[test]
fn a_paused_program_is_frozen_for_docker_commit_and_resumes() {
    let dir = bundle_with(echo());
    let (control, mut stdin, out, run) = start_echo(dir.path());
    std::thread::sleep(Duration::from_millis(200)); // well into its read

    // docker commit's pause: the program, frozen at its read, in /.shem
    assert!(control.pause(Duration::from_secs(10)));
    let state = dir.path().join("rootfs/.shem");
    assert!(state.join("container.json").is_file());
    assert!(state.join("machine.state").is_file());

    // What the commit takes: the container's filesystem, the state and all
    let committed = tempfile::tempdir().unwrap();
    let copy = |from: &std::path::Path, to: &std::path::Path| {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
            }
        }
    };
    std::fs::copy(
        dir.path().join("config.json"),
        committed.path().join("config.json"),
    )
    .unwrap();
    copy(
        &dir.path().join("rootfs/bin"),
        &committed.path().join("rootfs/bin"),
    );
    copy(&state, &committed.path().join("rootfs/.shem"));

    // Resumed, the program reads on, and its state is gone again
    control.resume();
    stdin.write_all(b"ping\n").unwrap();
    assert_eq!(run.join().unwrap(), 0);
    assert_eq!(out.0.lock().unwrap().as_slice(), b"ready\nping\n");
    assert!(!state.exists());

    // The committed image's container picks up at the read, ready said before
    let bundle = Bundle::load(committed.path()).unwrap();
    let out = Captured::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::Cursor::new(b"pong\n".to_vec())),
        stdout: Box::new(out.clone()),
        stderr: Box::new(std::io::sink()),
    };
    assert_eq!(
        bundle::run(&bundle, stdio, &Control::default(), None).unwrap(),
        0
    );
    assert_eq!(out.0.lock().unwrap().as_slice(), b"pong\n");
}

#[test]
fn a_kill_ends_a_paused_program() {
    let dir = bundle_with(echo());
    let (control, _stdin, _, run) = start_echo(dir.path());
    assert!(control.pause(Duration::from_secs(10)));
    control.kill(9);
    assert_eq!(run.join().unwrap(), 128 + 9);
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
        bundle::run(&bundle, stdio, &killed(15), None).unwrap(),
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
fn kubernetes_sandboxes_wait_to_be_stopped() {
    let dir = bundle_with(elf(&[0x0000_006f], &[]));
    let config = r#"{"process": {"args": ["/pause"]}, "root": {"path": "rootfs"},
        "annotations": {"io.kubernetes.cri.container-type": "sandbox"}}"#;
    std::fs::write(dir.path().join("config.json"), config).unwrap();
    let bundle = Bundle::load(dir.path()).unwrap();
    assert!(bundle.sandbox);
    let stdio = Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(std::io::sink()),
        stderr: Box::new(std::io::sink()),
    };
    // /pause is no RV32 program, and the sandbox never runs it
    assert_eq!(
        bundle::run(&bundle, stdio, &killed(9), None).unwrap(),
        128 + 9
    );
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
    assert_eq!(
        bundle::run(&bundle, stdio, &Control::default(), None).unwrap(),
        3
    );
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
    assert_eq!(
        bundle::run(&bundle, stdio, &Control::default(), None).unwrap(),
        3
    );
    assert_eq!(out.0.lock().unwrap().as_slice(), msg);
}

#[test]
#[ignore = "needs a model server and a model: set SHEM_NEURAL_SOCKET, and SHEM_MODEL to a model directory"]
fn model_containers_hold_live_sessions() {
    let socket = std::env::var("SHEM_NEURAL_SOCKET").expect("SHEM_NEURAL_SOCKET");
    let model =
        std::env::var("SHEM_MODEL").expect("SHEM_MODEL, a model directory holding shem.json");
    // The bundle's directory name is the container's ID, and so the session's
    let dir = tempfile::tempdir().unwrap();
    let bundle_dir = dir.path().join("shem-session-test");
    std::fs::create_dir_all(bundle_dir.join("rootfs")).unwrap();
    std::os::unix::fs::symlink(&model, bundle_dir.join("rootfs/model")).unwrap();
    let config = serde_json::json!({
        "process": {"args": ["shem:generate", "--max-tokens", "24", "--keep",
                             "--system", "Answer in one short sentence."]},
        "root": {"path": "rootfs"},
        "annotations": {"dev.shem.vcpu": "neural", "dev.shem.neural-socket": socket},
    });
    std::fs::write(bundle_dir.join("config.json"), config.to_string()).unwrap();
    let bundle = Bundle::load(&bundle_dir).unwrap();
    assert_eq!(bundle.id, "shem-session-test");

    // The main process: a turn from stdin, then the session stays open
    let out = Captured::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::Cursor::new(b"My name is Joana.\n".to_vec())),
        stdout: Box::new(out.clone()),
        stderr: Box::new(std::io::sink()),
    };
    let control = Arc::new(Control::default());
    let main = {
        let (bundle, control) = (bundle.clone(), control.clone());
        std::thread::spawn(move || bundle::run(&bundle, stdio, &control, None).unwrap())
    };
    while out.0.lock().unwrap().is_empty() {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    std::thread::sleep(std::time::Duration::from_secs(5)); // the rest of the first reply

    // docker exec: another turn of the same session, which remembers the first
    let mut reply = Vec::new();
    let args = ["shem:generate", "What did I just tell you my name is?"].map(String::from);
    containerd_shim_shem::session::exec(&bundle, &args, &mut reply).unwrap();
    let reply = String::from_utf8(reply).unwrap();
    assert!(reply.contains("Joana"), "{reply}");

    // docker commit's pause: the session, conversation and all, into /.shem
    containerd_shim_shem::session::save(&bundle).unwrap();
    let meta = std::fs::read_to_string(bundle_dir.join("rootfs/.shem/session.json")).unwrap();
    assert!(
        meta.contains("What did I just tell you my name is?"),
        "{meta}"
    );
    assert!(
        bundle_dir
            .join("rootfs/.shem/session.safetensors")
            .is_file()
    );

    control.kill(15);
    assert_eq!(main.join().unwrap(), 128 + 15);
}
