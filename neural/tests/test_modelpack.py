"""Fetching ModelPack artifacts, as the BuildKit frontend does for `FROM`
an artifact. The round trip through a registry needs oras and a registry at
SHEM_TEST_REGISTRY, and transformers to make a tiny checkpoint."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import torch

from shem_neural.model.blocks import Model
from shem_neural.model.importers import import_checkpoint
from shem_neural.modelpack import fetch, parse_reference


def test_references_name_their_registry_repository_and_tag():
    assert parse_reference("ghcr.io/team/model:v1") == ("ghcr.io", "team/model", "v1")
    assert parse_reference("localhost:5000/team/model") == ("localhost:5000", "team/model", "latest")
    assert parse_reference("team/model:v1") == ("registry-1.docker.io", "team/model", "v1")
    assert parse_reference("model") == ("registry-1.docker.io", "library/model", "latest")
    digest = "sha256:" + "0" * 64
    assert parse_reference(f"ghcr.io/team/model@{digest}") == ("ghcr.io", "team/model", digest)


def pack(checkpoint: Path, out: Path, form: str) -> str:
    """Pack `checkpoint` as a ModelPack artifact in an OCI layout at `out`,
    a layer per file, in `form`: raw, tar or tar+zstd."""
    import zstandard

    blobs = out / "blobs" / "sha256"
    blobs.mkdir(parents=True)

    def put(data: bytes) -> tuple[str, int]:
        digest = hashlib.sha256(data).hexdigest()
        (blobs / digest).write_bytes(data)
        return "sha256:" + digest, len(data)

    layers = []
    for path in sorted(checkpoint.iterdir()):
        data = path.read_bytes()
        if form != "raw":
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                info = tarfile.TarInfo(path.name)
                info.size, info.mode = len(data), 0o644
                archive.addfile(info, io.BytesIO(data))
            data = buffer.getvalue()
            if form == "tar+zstd":
                data = zstandard.ZstdCompressor().compress(data)
        kind = "weight" if path.suffix == ".safetensors" else "weight.config"
        digest, size = put(data)
        layers.append(
            {
                "mediaType": f"application/vnd.cnai.model.{kind}.v1.{form}",
                "digest": digest,
                "size": size,
                "annotations": {"org.cnai.model.filepath": path.name},
            }
        )
    config_digest, config_size = put(json.dumps({"descriptor": {"name": "tiny"}, "config": {}}).encode())
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "artifactType": "application/vnd.cnai.model.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.cnai.model.config.v1+json",
            "digest": config_digest,
            "size": config_size,
        },
        "layers": layers,
    }
    digest, size = put(json.dumps(manifest).encode())
    index = {"schemaVersion": 2, "manifests": [{"mediaType": manifest["mediaType"], "digest": digest, "size": size}]}
    (out / "index.json").write_text(json.dumps(index))
    (out / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))
    return digest


@pytest.mark.skipif(
    not os.environ.get("SHEM_TEST_REGISTRY") or not shutil.which(os.environ.get("SHEM_ORAS", "oras")),
    reason="needs oras and a registry at SHEM_TEST_REGISTRY",
)
@pytest.mark.parametrize("form", ["raw", "tar", "tar+zstd"])
def test_artifacts_fetch_into_models_that_run_as_their_checkpoints(form, tmp_path):
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    checkpoint = tmp_path / "checkpoint"
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(checkpoint)
    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(checkpoint / "tokenizer.json"))
    (checkpoint / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))

    layout = tmp_path / "layout"
    digest = pack(checkpoint, layout, form)
    reference = f"{os.environ['SHEM_TEST_REGISTRY']}/shem-tests/fetched:{form.replace('+', '-')}"
    oras = os.environ.get("SHEM_ORAS", "oras")
    subprocess.run([oras, "cp", "--from-oci-layout", f"{layout}@{digest}", "--to-plain-http", reference], check=True)

    model_dir = fetch(reference, tmp_path / "fetched", plain_http=True)
    import_checkpoint(checkpoint, tmp_path / "imported")
    tokens = torch.arange(9)[None, :]
    fetched, imported = Model(model_dir), Model(tmp_path / "imported")
    assert torch.equal(
        fetched.forward(tokens, fetched.init_state(), 0), imported.forward(tokens, imported.init_state(), 0)
    )
