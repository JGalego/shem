//! Artifacts, end to end, on a tiny random checkpoint: ModelPack artifacts
//! packed as modctl packs them, built on with `FROM`, run, primed with
//! `WARM`, and pulled from and pushed to a registry; model images published
//! as ModelPack artifacts; and sessions published beside their models.
//!
//! These need uv and the neural package's development dependencies, which
//! make the checkpoint with transformers: run them with `--ignored`. The
//! registry test also needs oras and a registry, at SHEM_TEST_REGISTRY.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde_json::{Value, json};
use shem::image::digest;

const NEURAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../neural");
const PROMPT: [&str; 3] = ["t5", "t6", "t7"];

/// A tiny random Llama with a word-level tokenizer, made once for every test.
fn checkpoint() -> &'static Path {
    static DIR: OnceLock<tempfile::TempDir> = OnceLock::new();
    DIR.get_or_init(|| make_checkpoint(0)).path()
}

/// A tiny random Llama, its weights from `seed`.
fn make_checkpoint(seed: u32) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let script = r#"
import json, sys
from pathlib import Path
import torch, transformers
from tokenizers import Tokenizer, models, pre_tokenizers
out = Path(sys.argv[1])
torch.manual_seed(int(sys.argv[2]))
config = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
    pad_token_id=0)
model = transformers.LlamaForCausalLM(config)
with torch.no_grad():
    for p in model.parameters():
        p.add_(torch.randn_like(p) * 0.1)
model.save_pretrained(out)
# The words a primed session's end is found with, as a real tokenizer has them
vocab = {f"t{i}": i for i in range(62)} | {"A": 62, "?": 63}
tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.save(str(out / "tokenizer.json"))
template = "{{ bos_token }} {% for m in messages %}{{ m.content }} {% endfor %}"
(out / "tokenizer_config.json").write_text(json.dumps({"chat_template": template, "bos_token": "t1", "eos_token": "t2"}))
(out / "LICENSE").write_text("Apache-2.0, for a test\n")
"#;
    let seed = seed.to_string();
    let made = python(&["-c", script, dir.path().to_str().unwrap(), &seed]);
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
    dir
}

fn python(args: &[&str]) -> std::process::Output {
    Command::new("uv")
        .args(["run", "--quiet", "--project", NEURAL, "python"])
        .args(args)
        .output()
        .expect("running uv")
}

/// What the checkpoint replies to the prompt, imported as `FROM hf.co/...`
/// imports a model and run by the driver directly.
fn reference(options: &[&str]) -> String {
    let imported = tempfile::tempdir().unwrap();
    let importer = [
        "-m",
        "shem_neural.model.importers",
        checkpoint().to_str().unwrap(),
    ];
    let done = python(&[&importer[..], &[imported.path().to_str().unwrap()]].concat());
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    let mut args = vec!["-m", "shem_neural.model.generate"];
    args.extend(options);
    args.push(imported.path().to_str().unwrap());
    args.push("--");
    args.extend(PROMPT);
    let replied = python(&args);
    assert!(
        replied.status.success(),
        "{}",
        String::from_utf8_lossy(&replied.stderr)
    );
    String::from_utf8(replied.stdout).unwrap()
}

/// Pack the checkpoint as an OCI layout in `out`, as modctl does: a layer
/// for each file, raw or in a tar, typed by what the file is, with the
/// media types and annotations of the namespace `ns` (`cncf` in the spec,
/// `cnai` in modctl). Returns the manifest's digest and its layers' digests.
fn pack(out: &Path, ns: &str, raw: bool) -> (String, Vec<String>) {
    let blobs = out.join("blobs/sha256");
    std::fs::create_dir_all(&blobs).unwrap();
    let put = |data: &[u8]| -> (String, usize) {
        let digest = digest(data);
        std::fs::write(blobs.join(&digest["sha256:".len()..]), data).unwrap();
        (digest, data.len())
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(checkpoint())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    let (mut layers, mut diff_ids) = (Vec::new(), Vec::new());
    for path in files {
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let data = std::fs::read(&path).unwrap();
        let kind = match name.as_str() {
            n if n.ends_with(".safetensors") => "weight",
            "LICENSE" => "doc",
            _ => "weight.config",
        };
        let blob = if raw {
            data.clone()
        } else {
            let mut builder = tar::Builder::new(Vec::new());
            let mut header = tar::Header::new_ustar();
            header.set_mode(0o644);
            header.set_uid(1000);
            header.set_mtime(1_794_480_844);
            header.set_size(data.len() as u64);
            builder.append_data(&mut header, &name, &data[..]).unwrap();
            builder.into_inner().unwrap()
        };
        let (digest, size) = put(&blob);
        diff_ids.push(digest.clone());
        let metadata = json!({"name": name, "mode": 420, "uid": 1000, "gid": 1000, "size": data.len(), "typeflag": 0});
        let format = if raw { "raw" } else { "tar" };
        layers.push(json!({
            "mediaType": format!("application/vnd.{ns}.model.{kind}.v1.{format}"),
            "digest": digest,
            "size": size,
            "annotations": {
                format!("org.{ns}.model.filepath"): name,
                format!("org.{ns}.model.file.metadata+json"): metadata.to_string(),
            },
        }));
    }
    let config = json!({
        "descriptor": {"name": "tiny-llama", "family": "llama", "licenses": ["Apache-2.0"]},
        "config": {"architecture": "transformer", "format": "safetensors"},
        "modelfs": {"type": "layers", "diffIds": diff_ids},
    });
    let (config_digest, config_size) = put(&serde_json::to_vec(&config).unwrap());
    let manifest = json!({
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "artifactType": format!("application/vnd.{ns}.model.manifest.v1+json"),
        "config": {
            "mediaType": format!("application/vnd.{ns}.model.config.v1+json"),
            "digest": config_digest,
            "size": config_size,
        },
        "layers": layers,
    });
    let (manifest_digest, manifest_size) = put(&serde_json::to_vec(&manifest).unwrap());
    let index = json!({
        "schemaVersion": 2,
        "manifests": [{
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "digest": manifest_digest,
            "size": manifest_size,
            "annotations": {"org.opencontainers.image.ref.name": "latest"},
        }],
    });
    std::fs::write(out.join("index.json"), index.to_string()).unwrap();
    std::fs::write(out.join("oci-layout"), r#"{"imageLayoutVersion":"1.0.0"}"#).unwrap();
    (manifest_digest, diff_ids)
}

fn shem(home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_shem"))
        .args(args)
        .env("SHEM_HOME", home)
        .env("SHEM_NEURAL_PROJECT", NEURAL)
        .output()
        .expect("running shem")
}

fn ok(output: std::process::Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// The manifest and config of the image in the layout at `dir`.
fn read_image(dir: &Path) -> (Value, Value) {
    let blob = |digest: &str| -> Value {
        let path = dir.join("blobs/sha256").join(&digest["sha256:".len()..]);
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    };
    let index: Value =
        serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
    let manifest = blob(index["manifests"][0]["digest"].as_str().unwrap());
    let config = blob(manifest["config"]["digest"].as_str().unwrap());
    (manifest, config)
}

fn layer_digests(manifest: &Value) -> Vec<String> {
    manifest["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["digest"].as_str().unwrap().to_string())
        .collect()
}

/// Build `shemfile` on the artifact in the context, as an OCI layout in the
/// returned directory, and run it with the prompt.
fn build_and_run(context: &Path, shemfile: &str, home: &Path) -> (PathBuf, String, String) {
    std::fs::write(context.join("Shemfile"), shemfile).unwrap();
    let out = home.join("built");
    let built = shem(
        home,
        &[
            "build",
            "-o",
            out.to_str().unwrap(),
            context.to_str().unwrap(),
        ],
    );
    let log = String::from_utf8_lossy(&built.stderr).into_owned();
    ok(built);
    let mut args = vec!["run", out.to_str().unwrap()];
    args.extend(PROMPT);
    let reply = ok(shem(home, &args));
    (out, reply, log)
}

#[test]
#[ignore = "needs uv and the neural package's dev dependencies"]
fn twins_share_the_artifacts_tar_layers_and_run_its_checkpoint() {
    let context = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let (artifact, blobs) = pack(&context.path().join("artifact"), "cncf", false);
    let shemfile = "FROM artifact\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\"]\n";
    let (out, reply, _) = build_and_run(context.path(), shemfile, home.path());

    let (manifest, config) = read_image(&out);
    let layers = layer_digests(&manifest);
    assert_eq!(
        layers[..blobs.len()],
        blobs[..],
        "the artifact's blobs, unchanged"
    );
    assert_eq!(layers.len(), blobs.len() + 1, "and one layer of shem's");
    assert_eq!(manifest["subject"]["digest"], artifact.as_str());
    let labels = &config["config"]["Labels"];
    assert_eq!(labels["dev.shem.model"], "/");
    assert_eq!(labels["org.opencontainers.image.title"], "tiny-llama");
    assert_eq!(labels["org.opencontainers.image.licenses"], "Apache-2.0");
    assert_eq!(
        labels["org.opencontainers.image.base.digest"],
        artifact.as_str()
    );
    assert_eq!(reply, reference(&["--max-tokens", "8"]));

    // An artifact's layout runs as is, as its twin, with the driver's defaults
    let layout = context.path().join("artifact");
    let mut args = vec!["run", layout.to_str().unwrap()];
    args.extend(PROMPT);
    assert_eq!(ok(shem(home.path(), &args)), reference(&[]));
}

#[test]
#[ignore = "needs uv and the neural package's dev dependencies"]
fn twins_wrap_raw_layers_in_tars_of_their_own() {
    let context = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let (_, blobs) = pack(&context.path().join("artifact"), "cnai", true);
    let shemfile = "FROM artifact\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\"]\n";
    let (out, reply, _) = build_and_run(context.path(), shemfile, home.path());

    let (manifest, _) = read_image(&out);
    let layers = layer_digests(&manifest);
    assert!(
        blobs.iter().all(|b| !layers.contains(b)),
        "a raw file is no layer"
    );
    assert_eq!(layers.len(), blobs.len() + 1);
    assert_eq!(reply, reference(&["--max-tokens", "8"]));
}

#[test]
#[ignore = "needs uv and the neural package's dev dependencies"]
fn twins_warm_as_imported_models_do() {
    let context = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    pack(&context.path().join("artifact"), "cncf", false);
    let shemfile = "FROM artifact\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\", \"--system\", \"t9 t10\"]\nWARM --until=prompt\n";
    let (_, reply, log) = build_and_run(context.path(), shemfile, home.path());
    assert!(log.contains("primed a session"), "{log}");
    assert_eq!(
        reply,
        reference(&["--max-tokens", "8", "--system", "t9 t10"])
    );
}

#[test]
#[ignore = "needs uv and the neural package's dev dependencies"]
fn images_built_on_twins_publish_as_uncompressed_artifacts() {
    // A twin's files are at the root already, so the layers an uncompressed
    // export writes are its tars as they were: blobs the export has linked
    // already, read-only, as oras leaves them
    let context = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let (_, blobs) = pack(&context.path().join("artifact"), "cncf", false);
    for entry in std::fs::read_dir(context.path().join("artifact/blobs/sha256")).unwrap() {
        let path = entry.unwrap().path();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&path, permissions).unwrap();
    }
    let shemfile = "FROM artifact\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\"]\n";
    let (out, _, _) = build_and_run(context.path(), shemfile, home.path());
    let image = shem::image::Image::read_layout(&out).unwrap();
    let artifact = home.path().join("exported");
    export(&out, &image, &artifact, true);

    let layers = layer_digests(&read_image(&artifact).0);
    assert!(
        blobs.iter().all(|blob| layers.contains(blob)),
        "the twin's tars"
    );
    let dest = tempfile::tempdir().unwrap();
    extract(&artifact, dest.path());
    assert!(loads_as_the_checkpoint(dest.path()));
}

#[test]
#[ignore = "needs uv, oras and a registry: set SHEM_TEST_REGISTRY"]
fn artifacts_pull_as_twins_that_push_beside_them() {
    let registry = std::env::var("SHEM_TEST_REGISTRY").unwrap_or_else(|_| "localhost:5000".into());
    let staging = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let layout = staging.path().join("artifact");
    let (artifact, blobs) = pack(&layout, "cnai", false);
    let remote = format!("{registry}/shem-tests/tiny-llama");
    let pushed = Command::new(std::env::var("SHEM_ORAS").unwrap_or_else(|_| "oras".into()))
        .args(["cp", "--from-oci-layout"])
        .arg(format!("{}@{artifact}", layout.display()))
        .args(["--to-plain-http", &format!("{remote}:artifact")])
        .output()
        .expect("running oras");
    assert!(
        pushed.status.success(),
        "{}",
        String::from_utf8_lossy(&pushed.stderr)
    );

    // `shem run` pulls the artifact and keeps it as its twin
    let artifact_ref = format!("{remote}:artifact");
    let mut args = vec!["run", artifact_ref.as_str()];
    args.extend(PROMPT);
    assert_eq!(ok(shem(home.path(), &args)), reference(&[]));

    // The twin pushes beside the artifact, its layers the artifact's blobs
    let twin_ref = format!("{remote}:twin");
    ok(shem(
        home.path(),
        &["push", "--plain-http", &artifact_ref, &twin_ref],
    ));
    let pulled = home.path().join("twin");
    ok(shem(
        home.path(),
        &["pull", "--plain-http", "-t", "twin-copy", &twin_ref],
    ));
    let store = home.path().join("images/twin-copy/latest");
    std::fs::rename(&store, &pulled).unwrap();
    let (manifest, _) = read_image(&pulled);
    assert_eq!(layer_digests(&manifest)[..blobs.len()], blobs[..]);
    assert_eq!(manifest["subject"]["digest"], artifact.as_str());
}

/// A context holding a copy of the checkpoint, as `ckpt`, and a model image
/// built `FROM` it, which imports it as `hf.co/` imports a model: its layout
/// directory, and the image.
fn checkpoint_image(home: &Path) -> (tempfile::TempDir, PathBuf, shem::image::Image) {
    let context = tempfile::tempdir().unwrap();
    let ckpt = context.path().join("ckpt");
    std::fs::create_dir(&ckpt).unwrap();
    for entry in std::fs::read_dir(checkpoint()).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, ckpt.join(path.file_name().unwrap())).unwrap();
    }
    let shemfile = "FROM ckpt\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\"]\n";
    std::fs::write(context.path().join("Shemfile"), shemfile).unwrap();
    let out = home.join("image");
    let args = [
        "build",
        "-o",
        out.to_str().unwrap(),
        context.path().to_str().unwrap(),
    ];
    ok(shem(home, &args));
    let image = shem::image::Image::read_layout(&out).unwrap();
    (context, out, image)
}

/// Export the image in the layout `dir` as a ModelPack artifact in `out`.
fn export(dir: &Path, image: &shem::image::Image, out: &Path, uncompressed: bool) -> String {
    let root = tempfile::tempdir().unwrap();
    image.unpack_rootfs(root.path()).unwrap();
    shem::image::modelpack::export(image, dir, root.path(), out, "latest", uncompressed).unwrap()
}

/// The files an artifact's tar layers hold, unpacked into `dest`, as modctl
/// extracts them or a runtime mounts them.
fn extract(artifact: &Path, dest: &Path) {
    let (manifest, _) = read_image(artifact);
    for layer in manifest["layers"].as_array().unwrap() {
        let digest = layer["digest"].as_str().unwrap();
        let blob = std::fs::read(artifact.join("blobs/sha256").join(&digest[7..])).unwrap();
        let tar = match layer["mediaType"].as_str().unwrap() {
            t if t.ends_with("+zstd") => zstd::decode_all(&blob[..]).unwrap(),
            _ => blob,
        };
        shem::image::unpack(&tar, dest).unwrap();
    }
}

/// Whether transformers gives the same logits for the checkpoint at `dir`
/// as for the original.
fn loads_as_the_checkpoint(dir: &Path) -> bool {
    let script = r#"
import sys, torch, transformers
tokens = torch.arange(3, 12)[None, :]
logits = lambda path: transformers.AutoModelForCausalLM.from_pretrained(path).eval()(input_ids=tokens).logits
print(torch.equal(logits(sys.argv[1]), logits(sys.argv[2])))
"#;
    let out = python(&[
        "-c",
        script,
        dir.to_str().unwrap(),
        checkpoint().to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim() == "True"
}

#[test]
#[ignore = "needs uv and the neural package's dev dependencies"]
fn images_publish_as_artifacts_whose_layers_are_their_own() {
    let home = tempfile::tempdir().unwrap();
    let (_context, dir, image) = checkpoint_image(home.path());
    let (image_manifest, _) = read_image(&dir);
    let shared = home.path().join("shared");
    export(&dir, &image, &shared, false);

    let (manifest, config) = read_image(&shared);
    assert_eq!(
        manifest["artifactType"],
        "application/vnd.cncf.model.manifest.v1+json"
    );
    assert_eq!(
        layer_digests(&manifest),
        layer_digests(&image_manifest),
        "the image's own blobs"
    );
    assert_eq!(
        manifest["subject"]["digest"],
        shem::image::Image::layout_digest(&dir).unwrap().as_str()
    );
    let types: Vec<&str> = manifest["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["mediaType"].as_str().unwrap())
        .collect();
    assert_eq!(
        types[0],
        "application/vnd.cncf.model.weight.config.v1.tar+zstd"
    );
    assert!(
        types[1..]
            .iter()
            .all(|t| *t == "application/vnd.cncf.model.weight.v1.tar+zstd"),
        "{types:?}"
    );
    assert_eq!(config["descriptor"]["family"], "llama");
    assert_eq!(config["config"]["format"], "safetensors");
    assert_eq!(config["config"]["precision"], "float32");
    // 64 × 32 embeddings and head, 2 layers of 4 × 32 × 32 attention and 3 × 32 × 48 MLP, 5 norms of 32
    let parameters = 2 * 64 * 32 + 2 * (2 * 32 * 32 + 2 * 16 * 32 + 3 * 32 * 48) + 5 * 32;
    assert_eq!(
        config["config"]["paramSize"],
        shem::image::modelpack::param_size(parameters)
    );

    let plain = home.path().join("plain");
    export(&dir, &image, &plain, true);
    let (plain_manifest, plain_config) = read_image(&plain);
    assert!(
        plain_manifest["layers"]
            .as_array()
            .unwrap()
            .iter()
            .all(|l| l["mediaType"].as_str().unwrap().ends_with(".v1.tar"))
    );
    let plain_files = plain_manifest["layers"].as_array().unwrap();
    assert!(
        plain_files
            .iter()
            .all(|l| !l["annotations"]["org.cncf.model.filepath"]
                .as_str()
                .unwrap_or_default()
                .starts_with("model/")),
        "files at the root, as modctl packs them"
    );
    assert_eq!(
        plain_config["modelfs"]["diffIds"].as_array().unwrap().len(),
        config["modelfs"]["diffIds"].as_array().unwrap().len()
    );

    for (artifact, model) in [(&shared, "model"), (&plain, "")] {
        let files = tempfile::tempdir().unwrap();
        extract(artifact, files.path());
        assert!(loads_as_the_checkpoint(&files.path().join(model)));
    }

    // And back: the artifact is a base like any other, whose twin replies as the image does
    let context = tempfile::tempdir().unwrap();
    std::fs::rename(&shared, context.path().join("artifact")).unwrap();
    let shemfile = "FROM artifact\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\"]\n";
    let (_, reply, _) = build_and_run(context.path(), shemfile, home.path());
    assert_eq!(reply, reference(&["--max-tokens", "8"]));
}

#[test]
#[ignore = "needs uv, oras and a registry: set SHEM_TEST_REGISTRY"]
fn images_push_as_artifacts_that_run_as_twins() {
    let registry = std::env::var("SHEM_TEST_REGISTRY").unwrap_or_else(|_| "localhost:5000".into());
    let home = tempfile::tempdir().unwrap();
    let (_context, dir, _) = checkpoint_image(home.path());
    let remote = format!("{registry}/shem-tests/tiny-llama:exported");
    ok(shem(
        home.path(),
        &[
            "push",
            "--plain-http",
            "--modelpack",
            dir.to_str().unwrap(),
            &remote,
        ],
    ));
    let mut args = vec!["run", remote.as_str()];
    args.extend(PROMPT);
    // The twin of the artifact runs the model with the driver's defaults: an
    // artifact carries the model, not the image's entrypoint
    let fresh = tempfile::tempdir().unwrap();
    assert_eq!(ok(shem(fresh.path(), &args)), reference(&[]));
}

/// Two images of the checkpoint with a system prompt, `cold` and `warm`,
/// which differ only in `warm`'s session, primed with `WARM`.
fn cold_and_warm(home: &Path) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let context = tempfile::tempdir().unwrap();
    let ckpt = context.path().join("ckpt");
    std::fs::create_dir(&ckpt).unwrap();
    for entry in std::fs::read_dir(checkpoint()).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, ckpt.join(path.file_name().unwrap())).unwrap();
    }
    let entrypoint = "FROM ckpt\nENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\", \"--system\", \"t9 t10\"]\n";
    let mut dirs = Vec::new();
    for (name, shemfile) in [
        ("cold", entrypoint.to_string()),
        ("warm", format!("{entrypoint}WARM --until=prompt\n")),
    ] {
        std::fs::write(context.path().join("Shemfile"), shemfile).unwrap();
        let out = home.join(name);
        ok(shem(
            home,
            &[
                "build",
                "-o",
                out.to_str().unwrap(),
                context.path().to_str().unwrap(),
            ],
        ));
        dirs.push(out);
    }
    let warm = dirs.pop().unwrap();
    (context, dirs.pop().unwrap(), warm)
}

fn run_with_state(home: &Path, image: &Path, state: &str) -> std::process::Output {
    let mut args = vec!["run", "--state", state, image.to_str().unwrap()];
    args.extend(PROMPT);
    shem(home, &args)
}

#[test]
#[ignore = "needs uv and the neural package's dev dependencies"]
fn sessions_publish_beside_their_models_and_resume_on_them() {
    let home = tempfile::tempdir().unwrap();
    let (_context, cold, warm) = cold_and_warm(home.path());
    let image = shem::image::Image::read_layout(&warm).unwrap();
    let root = tempfile::tempdir().unwrap();
    image.unpack_rootfs(root.path()).unwrap();
    std::fs::create_dir_all(root.path().join(".shem")).unwrap();
    let state = home.path().join("state");
    shem::image::session::publish(&image, root.path(), &state, "latest").unwrap();

    let (manifest, config) = read_image(&state);
    assert_eq!(
        manifest["artifactType"],
        "application/vnd.shem.session.v1+json"
    );
    let cold_digest = shem::image::Image::layout_digest(&cold).unwrap();
    assert_eq!(
        manifest["subject"]["digest"],
        cold_digest.as_str(),
        "the image before WARM, byte for byte"
    );
    assert_eq!(config["format"], "shem.session/v1");
    assert!(config["tokens"].as_u64().unwrap() > 0);
    assert_eq!(config["precision"], "float32");
    assert_eq!(config["blocks"]["attention"], 1, "{config}");

    let reply = ok(run_with_state(home.path(), &cold, state.to_str().unwrap()));
    assert_eq!(
        reply,
        reference(&["--max-tokens", "8", "--system", "t9 t10"])
    );

    // The same model in another layout, its ModelPack twin, continues it too
    let shemfile =
        "ENTRYPOINT [\"shem:generate\", \"--max-tokens\", \"8\", \"--system\", \"t9 t10\"]\n";
    let packed = tempfile::tempdir().unwrap();
    pack(&packed.path().join("artifact"), "cncf", false);
    std::fs::write(
        packed.path().join("Shemfile"),
        format!("FROM artifact\n{shemfile}"),
    )
    .unwrap();
    let twin = home.path().join("twin");
    let built = [
        "build",
        "-o",
        twin.to_str().unwrap(),
        packed.path().to_str().unwrap(),
    ];
    ok(shem(home.path(), &built));
    let reply = ok(run_with_state(home.path(), &twin, state.to_str().unwrap()));
    assert_eq!(
        reply,
        reference(&["--max-tokens", "8", "--system", "t9 t10"])
    );

    // A model of the same shape with other weights refuses it, rather than run wrong
    let context = tempfile::tempdir().unwrap();
    let other = make_checkpoint(1);
    std::fs::rename(other.path(), context.path().join("ckpt")).unwrap();
    std::fs::write(
        context.path().join("Shemfile"),
        format!("FROM ckpt\n{shemfile}"),
    )
    .unwrap();
    let different = home.path().join("different");
    let built = [
        "build",
        "-o",
        different.to_str().unwrap(),
        context.path().to_str().unwrap(),
    ];
    ok(shem(home.path(), &built));
    let refused = run_with_state(home.path(), &different, state.to_str().unwrap());
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("another model"));
}

#[test]
#[ignore = "needs uv and the neural package's dev dependencies"]
fn runs_write_receipts_that_replay_on_their_image_alone() {
    let home = tempfile::tempdir().unwrap();
    let (_context, cold, warm) = cold_and_warm(home.path());
    ok(shem(home.path(), &["tag", warm.to_str().unwrap(), "golem"]));
    let receipt = home.path().join("receipt.json");
    let written = |image: &str, state: Option<&str>| -> (String, Value) {
        let mut args = vec!["run", "--receipt", receipt.to_str().unwrap()];
        if let Some(state) = state {
            args.extend(["--state", state]);
        }
        args.push(image);
        args.extend(PROMPT);
        let reply = ok(shem(home.path(), &args));
        let held = serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
        (reply, held)
    };
    let replay = |args: &[&str]| -> (Option<i32>, String) {
        let args = [&["replay", receipt.to_str().unwrap()], args].concat();
        let done = shem(home.path(), &args);
        (
            done.status.code(),
            String::from_utf8_lossy(&done.stderr).into_owned(),
        )
    };

    // A named image's receipt names it, and replays by itself
    let (reply, held) = written("golem", None);
    let digest = shem::image::Image::layout_digest(&warm).unwrap();
    assert_eq!(held["image"], format!("golem@{digest}"));
    assert_eq!(held["prompt"], PROMPT.join(" "));
    assert_eq!(held["reply"], reply.trim_end());
    assert_eq!(held["tokens"].as_array().unwrap().len(), 8);
    let (status, log) = replay(&[]);
    assert_eq!(status, Some(0), "{log}");
    assert!(log.contains("verified: the model chose the receipt's 8 tokens again"));
    // The same image by another name replays it, and no other image does:
    // not even the same model before it was primed
    let (status, log) = replay(&[warm.to_str().unwrap()]);
    assert_eq!(status, Some(0), "{log}");
    let (status, log) = replay(&[cold.to_str().unwrap()]);
    assert_ne!(status, Some(0));
    assert!(log.contains("not the receipt's sha256:"), "{log}");
    // A receipt that says the model replied otherwise is found out
    let mut forged = held.clone();
    forged["tokens"][0] = json!((held["tokens"][0].as_u64().unwrap() + 1) % 32);
    std::fs::write(&receipt, forged.to_string()).unwrap();
    let (status, log) = replay(&[]);
    assert!(
        matches!(status, Some(1 | 3)),
        "a mismatch, or a close call: {log}"
    );
    assert!(
        log.contains("parts from the receipt at token 1 of 8"),
        "{log}"
    );

    // A layout's receipt gives its digest alone, and its session's: the
    // replay is told where both are, and the session must be that one
    let image = shem::image::Image::read_layout(&warm).unwrap();
    let root = tempfile::tempdir().unwrap();
    image.unpack_rootfs(root.path()).unwrap();
    std::fs::create_dir_all(root.path().join(".shem")).unwrap();
    let state = home.path().join("state");
    shem::image::session::publish(&image, root.path(), &state, "latest").unwrap();
    let (_, held) = written(cold.to_str().unwrap(), Some(state.to_str().unwrap()));
    let cold_digest = shem::image::Image::layout_digest(&cold).unwrap();
    assert_eq!(held["image"], cold_digest.as_str());
    assert_eq!(
        held["state"],
        shem::image::Image::layout_digest(&state).unwrap().as_str()
    );
    let (status, log) = replay(&[]);
    assert_ne!(status, Some(0));
    assert!(log.contains("give the image to replay it on"), "{log}");
    let (status, log) = replay(&[cold.to_str().unwrap()]);
    assert_ne!(status, Some(0));
    assert!(log.contains("give the session with --state"), "{log}");
    let (status, log) = replay(&["--state", state.to_str().unwrap(), cold.to_str().unwrap()]);
    assert_eq!(status, Some(0), "{log}");
}

#[test]
#[ignore = "needs uv, oras and a registry: set SHEM_TEST_REGISTRY"]
fn sessions_push_with_their_models_and_resume_from_the_registry() {
    let registry = std::env::var("SHEM_TEST_REGISTRY").unwrap_or_else(|_| "localhost:5000".into());
    let home = tempfile::tempdir().unwrap();
    let (_context, cold, warm) = cold_and_warm(home.path());
    let remote = format!("{registry}/shem-tests/tiny-llama:session");
    ok(shem(
        home.path(),
        &[
            "push",
            "--plain-http",
            "--state",
            warm.to_str().unwrap(),
            &remote,
        ],
    ));

    // The model the session names came with it
    let oras = std::env::var("SHEM_ORAS").unwrap_or_else(|_| "oras".into());
    let fetch = |reference: &str| -> Value {
        let out = Command::new(&oras)
            .args(["manifest", "fetch", "--plain-http", reference])
            .output()
            .expect("running oras");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    };
    let subject = fetch(&remote)["subject"]["digest"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(subject, shem::image::Image::layout_digest(&cold).unwrap());
    fetch(&format!("{registry}/shem-tests/tiny-llama@{subject}"));

    let reply = ok(run_with_state(home.path(), &cold, &remote));
    assert_eq!(
        reply,
        reference(&["--max-tokens", "8", "--system", "t9 t10"])
    );
}
