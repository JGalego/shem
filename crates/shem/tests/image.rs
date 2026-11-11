mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use common::*;
use shem::container::Container;
use shem::image::shemfile::{self, Instruction, Until};
use shem::image::{self, BuildOptions, Image, STATE_TYPE, parse_reference};
use shem::linux::Stdio;
use shem::reference::ReferenceVcpu;

const MEM: u32 = 1 << 20;
const AT_FDCWD: u32 = -100i32 as u32;

/// A context directory holding `files` (executables) and a Shemfile.
fn context(shemfile: &str, files: &[(&str, Vec<u8>)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("Shemfile"), shemfile).unwrap();
    for (name, data) in files {
        let path = dir.path().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, data).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    dir
}

fn build(context: &Path) -> anyhow::Result<Image> {
    image::build(&BuildOptions {
        shemfile: context.join("Shemfile"),
        context: context.to_path_buf(),
        build_args: Default::default(),
        mem: Some(MEM),
    })
}

/// Start `image` with `stdin`, run it, and return the status and stdout.
fn run(image: &Image, args: &[&str], stdin: &[u8]) -> (u8, Vec<u8>) {
    let root = tempfile::tempdir().unwrap();
    let stdout = SharedBuf::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::Cursor::new(stdin.to_vec())),
        stdout: Box::new(stdout.clone()),
        stderr: Box::new(std::io::sink()),
    };
    let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
    let mut container: Container<ReferenceVcpu> = image::start(
        image,
        |mem| Ok(ReferenceVcpu::new(mem)),
        &args,
        &[],
        stdio,
        root.path(),
        MEM,
    )
    .unwrap();
    let status = container.run(Some(10_000_000)).unwrap();
    (status, stdout.contents())
}

/// write(1, "hello image\n"), exit(0)
fn hello() -> Vec<u8> {
    let msg = b"hello image\n";
    let code = program![
        addi(A0, ZERO, 1),
        li(A1, DATA),
        addi(A2, ZERO, msg.len() as i32),
        addi(A7, ZERO, 64),
        ECALL,
        addi(A0, ZERO, 0),
        addi(A7, ZERO, 93),
        ECALL,
    ];
    elf(&code, msg)
}

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
fn scratch_images_build_and_run() {
    let ctx = context(
        "FROM scratch\nCOPY hello /bin/hello\nENV GREETING=hi\nWORKDIR /work\nENTRYPOINT [\"/bin/hello\"]\n",
        &[("hello", hello())],
    );
    let image = build(ctx.path()).unwrap();
    assert_eq!(image.config.entrypoint, ["/bin/hello"]);
    assert_eq!(image.config.env, ["GREETING=hi"]);
    assert_eq!(image.config.workdir, "/work");
    assert_eq!(image.layers.len(), 2, "one layer for COPY, one for WORKDIR");
    assert_eq!(run(&image, &[], b""), (0, b"hello image\n".to_vec()));
}

#[test]
fn builds_are_reproducible_and_layouts_round_trip() {
    let shemfile = "FROM scratch\nCOPY hello /bin/hello\nCMD [\"/bin/hello\"]\n";
    let out = tempfile::tempdir().unwrap();
    let mut digests = Vec::new();
    for n in 0..2 {
        // Fresh contexts, so file timestamps differ between the builds.
        let ctx = context(shemfile, &[("hello", hello())]);
        let image = build(ctx.path()).unwrap();
        let dir = out.path().join(n.to_string());
        digests.push(image.write_layout(&dir, "hello:latest").unwrap());
        assert_eq!(Image::read_layout(&dir).unwrap(), image);
        assert_eq!(Image::layout_digest(&dir).unwrap(), digests[n]);
    }
    assert_eq!(digests[0], digests[1]);
}

#[test]
fn warm_images_resume_at_the_frozen_system_call() {
    let shemfile = "FROM scratch\nCOPY echo /bin/echo\nENTRYPOINT [\"/bin/echo\"]\nWARM --until=syscall:read\n";
    let ctx = context(shemfile, &[("echo", echo())]);
    let image = build(ctx.path()).unwrap();
    assert_eq!(image.layers.last().unwrap().media_type, STATE_TYPE);
    assert_eq!(image.config.memory, Some(MEM));
    let frozen = image.warm_state().unwrap().unwrap();
    assert!(frozen.host.steps > 0);

    // "ready" was printed at build time; the run starts at the read.
    assert_eq!(run(&image, &[], b"ping\n"), (0, b"ping\n".to_vec()));
    assert_eq!(run(&image, &[], b"pong\n"), (0, b"pong\n".to_vec()));

    let again = build(ctx.path()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        image.write_layout(&dir.path().join("a"), "w").unwrap(),
        again.write_layout(&dir.path().join("b"), "w").unwrap(),
        "WARM is deterministic, so warm images are reproducible too"
    );
}

#[test]
fn warm_images_take_no_arguments() {
    let shemfile = "FROM scratch\nCOPY echo /bin/echo\nENTRYPOINT [\"/bin/echo\"]\nWARM --until=syscall:read\n";
    let ctx = context(shemfile, &[("echo", echo())]);
    let image = build(ctx.path()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let stdio = Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(std::io::sink()),
        stderr: Box::new(std::io::sink()),
    };
    let args = vec!["extra".to_string()];
    let started = image::start(
        &image,
        |mem| Ok(ReferenceVcpu::new(mem)),
        &args,
        &[],
        stdio,
        root.path(),
        MEM,
    );
    assert!(started.is_err());
}

#[test]
fn files_written_before_the_freeze_become_a_layer() {
    let mut data = b"/made\0\0\0data".to_vec();
    data.resize(32, 0);
    let code = program![
        li(A0, AT_FDCWD), // fd = openat(AT_FDCWD, "/made", O_CREAT | O_WRONLY, 0644)
        li(A1, DATA),
        addi(A2, ZERO, 0o101),
        addi(A3, ZERO, 0o644),
        addi(A7, ZERO, 56),
        ECALL,
        addi(S0, A0, 0),
        li(A1, DATA + 8), // write(fd, "data", 4)
        addi(A2, ZERO, 4),
        addi(A7, ZERO, 64),
        ECALL,
        addi(A0, S0, 0), // close(fd)
        addi(A7, ZERO, 57),
        ECALL,
        addi(A0, ZERO, 0), // read(0, buf, 16): WARM stops here
        li(A1, DATA + 16),
        addi(A2, ZERO, 16),
        addi(A7, ZERO, 63),
        ECALL,
        addi(A0, ZERO, 0),
        addi(A7, ZERO, 93),
        ECALL,
    ];
    let shemfile =
        "FROM scratch\nCOPY maker /maker\nENTRYPOINT [\"/maker\"]\nWARM --until=syscall:read\n";
    let ctx = context(shemfile, &[("maker", elf(&code, &data))]);
    let image = build(ctx.path()).unwrap();
    let root = tempfile::tempdir().unwrap();
    image.unpack_rootfs(root.path()).unwrap();
    assert_eq!(fs::read(root.path().join("made")).unwrap(), b"data");
}

#[test]
fn copy_follows_docker_rules_for_directories() {
    let ctx = context(
        "FROM scratch\nCOPY tree /opt/\nCOPY tree/a /single\nCOPY hello tree/a /many/\nCMD [\"/x\"]\n",
        &[
            ("tree/a", b"a".to_vec()),
            ("tree/sub/b", b"b".to_vec()),
            ("hello", hello()),
        ],
    );
    let image = build(ctx.path()).unwrap();
    let root = tempfile::tempdir().unwrap();
    image.unpack_rootfs(root.path()).unwrap();
    for (path, contents) in [
        ("opt/a", "a"),
        ("opt/sub/b", "b"),
        ("single", "a"),
        ("many/a", "a"),
    ] {
        assert_eq!(
            fs::read_to_string(root.path().join(path)).unwrap(),
            contents,
            "{path}"
        );
    }
    assert!(root.path().join("many/hello").is_file());
}

#[test]
fn builds_refuse_what_a_shem_stage_cannot_do() {
    let escape = context("FROM scratch\nCOPY ../outside /x\n", &[]);
    assert!(build(escape.path()).is_err());
    let run_in_shem = context("FROM scratch\nRUN make\n", &[]);
    let err = build(run_in_shem.path()).unwrap_err().to_string();
    assert!(err.contains("RUN is not allowed in a shem stage"), "{err}");
    let docker_last = context("FROM alpine\nRUN true\n", &[]);
    assert!(build(docker_last.path()).is_err());
}

#[test]
fn shemfiles_parse_into_stages() {
    let text = r#"
# a comment
FROM shem-toolchain AS build
RUN rv32-cc -o hello \
    hello.c

FROM scratch
COPY --from=build /src/hello /bin/
ENV A=1 B="two words"
ENV LEGACY value with spaces
ENTRYPOINT /bin/hello --flag
CMD ["x", "y"]
ARG VERSION=1.0
WARM --until=syscall:read --max-steps=1000
"#;
    let stages = shemfile::parse(text).unwrap();
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].name.as_deref(), Some("build"));
    assert_eq!(stages[0].lines[1], "RUN rv32-cc -o hello  hello.c");
    let ins = &stages[1].instructions;
    assert_eq!(
        ins[0],
        Instruction::Copy {
            from: Some("build".into()),
            sources: vec!["/src/hello".into()],
            dest: "/bin/".into()
        }
    );
    assert_eq!(
        ins[1],
        Instruction::Env(vec![
            ("A".into(), "1".into()),
            ("B".into(), "two words".into())
        ])
    );
    assert_eq!(
        ins[2],
        Instruction::Env(vec![("LEGACY".into(), "value with spaces".into())])
    );
    assert_eq!(
        ins[3],
        Instruction::Entrypoint(vec![
            "/bin/sh".into(),
            "-c".into(),
            "/bin/hello --flag".into()
        ])
    );
    assert_eq!(ins[4], Instruction::Cmd(vec!["x".into(), "y".into()]));
    assert_eq!(
        ins[5],
        Instruction::Arg {
            name: "VERSION".into(),
            default: Some("1.0".into())
        }
    );
    assert_eq!(
        ins[6],
        Instruction::Warm {
            until: Until::Syscall(63),
            max_steps: Some(1000)
        }
    );
    for bad in [
        "COPY a\n",
        "WARM\n",
        "WARM --until=syscall:nonsense\n",
        "WARM --until=output:\n",
        "WARM --until=output:\"unterminated\n",
        "FROM a b c\n",
    ] {
        let text = if bad.starts_with("FROM") {
            bad.to_string()
        } else {
            format!("FROM scratch\n{bad}")
        };
        assert!(shemfile::parse(&text).is_err(), "{bad:?} should not parse");
    }
    assert!(
        shemfile::parse("COPY a b\n").is_err(),
        "an instruction before FROM"
    );
}

#[test]
fn image_references_are_checked() {
    assert_eq!(parse_reference("hello").unwrap(), ("hello", "latest"));
    assert_eq!(
        parse_reference("me/hello:1.0").unwrap(),
        ("me/hello", "1.0")
    );
    let with_registry = parse_reference("localhost:5000/me/hello:1.0").unwrap();
    assert_eq!(with_registry, ("localhost:5000/me/hello", "1.0"));
    let untagged = parse_reference("localhost:5000/me/hello").unwrap();
    assert_eq!(untagged, ("localhost:5000/me/hello", "latest"));
    for bad in [
        "",
        "../x",
        "a//b",
        "x:../y",
        "x:",
        "has space",
        "host:port/x",
        "a/b:c/d:1",
    ] {
        assert!(parse_reference(bad).is_err(), "{bad:?}");
    }
}

#[test]
#[ignore = "needs oras and a registry: run `just registry-test`"]
fn warm_images_round_trip_through_a_registry() {
    let registry = std::env::var("SHEM_TEST_REGISTRY").unwrap_or_else(|_| "localhost:5000".into());
    let shemfile = "FROM scratch\nCOPY echo /bin/echo\nENTRYPOINT [\"/bin/echo\"]\nWARM --until=syscall:read\n";
    let ctx = context(shemfile, &[("echo", echo())]);
    let image = build(ctx.path()).unwrap();
    let dirs = tempfile::tempdir().unwrap();
    let (pushed, pulled) = (dirs.path().join("pushed"), dirs.path().join("pulled"));
    let digest = image.write_layout(&pushed, "latest").unwrap();

    let remote = format!("{registry}/shem-tests/echo:warm");
    assert_eq!(image::push(&pushed, &remote, true).unwrap(), digest);
    assert_eq!(image::pull(&remote, &pulled, true).unwrap(), digest);
    let back = Image::read_layout(&pulled).unwrap();
    assert_eq!(back, image);
    assert_eq!(
        run(&back, &[], b"over the wire\n"),
        (0, b"over the wire\n".to_vec())
    );
}
