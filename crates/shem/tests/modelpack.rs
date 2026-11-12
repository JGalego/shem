//! ModelPack artifacts through their image twins, end to end: a tiny random
//! checkpoint, packed as modctl packs one, built on with `FROM`, run, primed
//! with `WARM`, and pulled from and pushed to a registry.
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
    DIR.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let script = r#"
import json, sys
from pathlib import Path
import torch, transformers
from tokenizers import Tokenizer, models, pre_tokenizers
out = Path(sys.argv[1])
torch.manual_seed(0)
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
        let made = python(&["-c", script, dir.path().to_str().unwrap()]);
        assert!(made.status.success(), "{}", String::from_utf8_lossy(&made.stderr));
        dir
    })
    .path()
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
