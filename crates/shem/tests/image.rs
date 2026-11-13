mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use common::*;
use shem::container::Container;
use shem::image::shemfile::{self, Instruction, Until};
use shem::image::{self, BuildOptions, Image, STATE_TYPE, Store, parse_reference};
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
fn frozen_counts_near_their_limits_go_on_without_overflowing() {
    let shemfile = "FROM scratch\nCOPY echo /bin/echo\nENTRYPOINT [\"/bin/echo\"]\nWARM --until=syscall:read\n";
    let ctx = context(shemfile, &[("echo", echo())]);
    let image = build(ctx.path()).unwrap();
    // As a state from an image's own files may set them
    let mut frozen = image.warm_state().unwrap().unwrap();
    frozen.machine.instret = u64::MAX - 1;
    frozen.host.steps = u64::MAX - 1;
    frozen.host.process.cpu_ns = u64::MAX - 1;
    frozen.host.linux.slept_ns = u64::MAX - 1;
    let stdout = SharedBuf::default();
    let stdio = Stdio {
        stdin: Box::new(std::io::Cursor::new(b"ping\n".to_vec())),
        stdout: Box::new(stdout.clone()),
        stderr: Box::new(std::io::sink()),
    };
    let mut container = Container::thaw(ReferenceVcpu::new(MEM), &frozen, stdio).unwrap();
    assert_eq!(container.run(None).unwrap(), 0);
    assert_eq!(stdout.contents(), b"ping\n");
}

#[test]
fn images_save_as_docker_archives_with_their_state_as_files() {
    use std::collections::BTreeMap;
    use std::io::Read;

    let shemfile = "FROM scratch\nCOPY echo /bin/echo\nENTRYPOINT [\"/bin/echo\"]\nWARM --until=syscall:read\n";
    let ctx = context(shemfile, &[("echo", echo())]);
    let image = build(ctx.path()).unwrap();
    let archive = image.docker_archive("echo", &[]).unwrap();
    assert_eq!(
        archive,
        image.docker_archive("echo", &[]).unwrap(),
        "reproducible"
    );

    let mut files = BTreeMap::new();
    for entry in tar::Archive::new(&archive[..]).entries().unwrap() {
        let mut entry = entry.unwrap();
        let name = entry.path().unwrap().to_string_lossy().into_owned();
        let mut data = Vec::new();
        entry.read_to_end(&mut data).unwrap();
        files.insert(name, data);
    }
    let json = |data: &[u8]| serde_json::from_slice::<serde_json::Value>(data).unwrap();
    let blob = |digest: &serde_json::Value| {
        let hex = digest.as_str().unwrap().strip_prefix("sha256:").unwrap();
        files[&format!("blobs/sha256/{hex}")].clone()
    };
    let index = json(&files["index.json"]);
    let entry = &index["manifests"][0];
    assert_eq!(entry["platform"]["architecture"], "riscv32");
    assert_eq!(
        entry["annotations"]["io.containerd.image.name"],
        "docker.io/library/echo:latest"
    );
    let manifest = json(&blob(&entry["digest"]));
    let config = json(&blob(&manifest["config"]["digest"]));
    assert_eq!(
        (config["os"].as_str(), config["architecture"].as_str()),
        (Some("linux"), Some("riscv32"))
    );
    assert_eq!(config["config"]["Entrypoint"][0], "/bin/echo");

    // Each layer decompresses to its diff ID, and the last holds the state
    let root = tempfile::tempdir().unwrap();
    let layers = manifest["layers"].as_array().unwrap();
    for (layer, diff_id) in layers
        .iter()
        .zip(config["rootfs"]["diff_ids"].as_array().unwrap())
    {
        let tar = zstd::decode_all(&blob(&layer["digest"])[..]).unwrap();
        assert_eq!(image::digest(&tar), diff_id.as_str().unwrap());
        image::unpack(&tar, root.path()).unwrap();
    }
    assert!(root.path().join("bin/echo").is_file());
    let frozen = image.warm_state().unwrap().unwrap();
    let state = fs::read(root.path().join(".shem/machine.state")).unwrap();
    assert_eq!(state, frozen.machine.encode());
    assert!(root.path().join(".shem/container.json").is_file());
}

#[test]
fn docker_archives_load_into_the_store() {
    let shemfile = "FROM scratch\nCOPY echo /bin/echo\nENTRYPOINT [\"/bin/echo\"]\nWARM --until=syscall:read\n";
    let ctx = context(shemfile, &[("echo", echo())]);
    let image = build(ctx.path()).unwrap();
    let home = tempfile::tempdir().unwrap();
    let store = Store::at(home.path());

    // Listed under another platform too, so the image is in a nested index
    let archive = image
        .docker_archive("echo", &["linux/amd64".to_string()])
        .unwrap();
    assert_eq!(store.load(&archive[..], None).unwrap(), ["echo:latest"]);
    let loaded = Image::read_layout(&store.path("echo").unwrap()).unwrap();
    assert_eq!(loaded.warm_state().unwrap(), image.warm_state().unwrap());
    assert_eq!(run(&loaded, &[], b"ping\n"), (0, b"ping\n".to_vec()));

    // Under a name of its own, which a second load replaces
    store.load(&archive[..], Some("team/echo:1")).unwrap();
    store.load(&archive[..], Some("team/echo:1")).unwrap();
    let names: Vec<String> = store.list().unwrap().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["echo:latest", "team/echo:1"]);
}

/// A tar of `files`, as an archive to load.
fn tar_of(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut tar = tar::Builder::new(Vec::new());
    for (name, data) in files {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(&mut header, name, *data).unwrap();
    }
    tar.into_inner().unwrap()
}

#[test]
fn loading_refuses_other_platforms_and_damaged_blobs() {
    let home = tempfile::tempdir().unwrap();
    let store = Store::at(home.path());
    let blob = |data: &[u8]| format!("blobs/sha256/{}", &image::digest(data)[7..]);
    let json = |value: serde_json::Value| serde_json::to_vec(&value).unwrap();

    // An image built without --platform linux/riscv32
    let config = json(serde_json::json!({ "architecture": "amd64", "os": "linux" }));
    let manifest = json(serde_json::json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": image::digest(&config),
            "size": config.len(),
        },
        "layers": [],
    }));
    let index = json(serde_json::json!({
        "schemaVersion": 2,
        "manifests": [{
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "digest": image::digest(&manifest),
            "size": manifest.len(),
            "annotations": { "io.containerd.image.name": "docker.io/library/web:latest" },
        }],
    }));
    let amd64 = tar_of(&[
        ("index.json", &index),
        (&blob(&config), &config),
        (&blob(&manifest), &manifest),
    ]);
    let err = store.load(&amd64[..], None).unwrap_err().to_string();
    assert!(err.contains("--platform linux/riscv32"), "{err}");

    // A blob that is not what its name says
    let damaged = tar_of(&[("index.json", &index), (&blob(&config), b"{}")]);
    let err = store.load(&damaged[..], None).unwrap_err().to_string();
    assert!(err.contains("does not match its digest"), "{err}");

    // Neither an archive without a name, unless the load gives it one
    let unnamed = String::from_utf8(index.clone())
        .unwrap()
        .replace("io.containerd.image.name", "other");
    let anonymous = tar_of(&[("index.json", unnamed.as_bytes())]);
    let err = store.load(&anonymous[..], None).unwrap_err().to_string();
    assert!(err.contains("--tag"), "{err}");
    assert!(store.list().unwrap().is_empty());
}

/// Append an entry named `name` as it is, as a hostile layer would have it:
/// tar's builder refuses names that climb out or start with `/`.
fn raw_entry(tar: &mut tar::Builder<Vec<u8>>, name: &str, kind: tar::EntryType, data: &[u8]) {
    let mut header = tar::Header::new_old();
    header.as_old_mut().name[..name.len()].copy_from_slice(name.as_bytes());
    header.set_entry_type(kind);
    header.set_size(data.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    tar.append(&header, data).unwrap();
}

#[test]
fn layers_cannot_reach_outside_their_root() {
    use tar::EntryType::{Regular, Symlink};

    let dir = tempfile::tempdir().unwrap();
    let (root, outside) = (dir.path().join("root"), dir.path().join("outside"));
    fs::create_dir_all(&root).unwrap();
    fs::create_dir_all(outside.join("dir")).unwrap();
    fs::write(outside.join("file"), "kept").unwrap();
    let away = outside.display().to_string();

    // Deletions by a path from /, by one that climbs out, and through a link
    // that leads out delete nothing
    let mut tar = tar::Builder::new(Vec::new());
    raw_entry(&mut tar, &format!("{away}/.wh.file"), Regular, b"");
    raw_entry(&mut tar, "../outside/.wh.dir", Regular, b"");
    let mut link = tar::Header::new_old();
    link.set_entry_type(Symlink);
    link.set_size(0);
    link.set_link_name(&outside).unwrap();
    link.set_mode(0o777);
    tar.append_link(&mut link, "out", &outside).unwrap();
    raw_entry(&mut tar, "out/.wh.file", Regular, b"");
    image::unpack(&tar.into_inner().unwrap(), &root).unwrap();
    assert!(outside.join("file").is_file() && outside.join("dir").is_dir());

    // Nor does a file through the link replace the one outside
    let mut tar = tar::Builder::new(Vec::new());
    raw_entry(&mut tar, "out/file", Regular, b"replaced");
    assert!(image::unpack(&tar.into_inner().unwrap(), &root).is_err());
    assert_eq!(fs::read_to_string(outside.join("file")).unwrap(), "kept");
}

#[test]
fn a_model_image_session_cannot_write_outside_it() {
    let dir = tempfile::tempdir().unwrap();
    let victim = dir.path().join("victim");
    let mut tar = tar::Builder::new(Vec::new());
    raw_entry(
        &mut tar,
        &victim.display().to_string(),
        tar::EntryType::Regular,
        b"written",
    );
    let image = Image {
        config: image::Config {
            model: Some("/model".into()),
            ..Default::default()
        },
        layers: vec![image::Layer::new(STATE_TYPE, tar.into_inner().unwrap())],
        subject: None,
    };
    let store = Store::at(&dir.path().join("home"));
    let digest = format!("sha256:{}", "0".repeat(64));
    let err = store.rootfs(&image, &digest).unwrap_err().to_string();
    assert!(err.contains("outside"), "{err}");
    assert!(!victim.exists());
}

#[test]
fn images_can_be_listed_under_other_platforms_too() {
    let ctx = context(
        "FROM scratch\nCOPY hello /bin/hello\nENTRYPOINT [\"/bin/hello\"]\n",
        &[("hello", hello())],
    );
    let image = build(ctx.path()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let also = ["linux/amd64".to_string(), "linux/arm64/v8".to_string()];
    let digest = image.write_layout_for(dir.path(), "latest", &also).unwrap();
    let blob = |digest: &str| fs::read(dir.path().join("blobs/sha256").join(&digest[7..])).unwrap();
    let index: serde_json::Value = serde_json::from_slice(&blob(&digest)).unwrap();
    let platforms: Vec<String> = (index["manifests"].as_array().unwrap().iter())
        .map(|m| {
            format!(
                "{}/{}",
                m["platform"]["os"].as_str().unwrap(),
                m["platform"]["architecture"].as_str().unwrap()
            )
        })
        .collect();
    assert_eq!(platforms, ["linux/riscv32", "linux/amd64", "linux/arm64"]);
    let manifests: Vec<&serde_json::Value> = index["manifests"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| &m["digest"])
        .collect();
    assert!(
        manifests.windows(2).all(|pair| pair[0] == pair[1]),
        "one image, listed three times"
    );
    // shem reads it back as the image it is
    let read = Image::read_layout(dir.path()).unwrap();
    assert_eq!(read.config, image.config);
    assert_eq!(read.layers, image.layers);
    assert!(
        image
            .write_layout_for(dir.path(), "x", &["amd64".to_string()])
            .is_err()
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
    let flagged = shemfile::parse("FROM --platform=$BUILDPLATFORM alpine:3.22 AS build\n").unwrap();
    assert_eq!(
        (flagged[0].from.as_str(), flagged[0].name.as_deref()),
        ("alpine:3.22", Some("build"))
    );
    let warm = shemfile::parse("FROM hf.co/Qwen/Qwen3-0.6B\nWARM --until=prompt\n").unwrap();
    assert_eq!(
        warm[0].instructions[0],
        Instruction::Warm {
            until: Until::Prompt,
            max_steps: None
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
