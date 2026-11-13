"""Hugging Face repos that hold a checkpoint, as ModelPack artifacts.

A repo that holds no image layout is served by `shem registry` as an
artifact made on request: a raw layer for each of its files, at its path, and
a model config from what the Hub says of the repo. A large file's layer is
the file itself, whose digest is the sha256 the Hub keeps for it, so its
bytes come from the Hub as they are. A small file is downloaded once to learn
its digest. The same commit always makes the same artifact.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
ARTIFACT_TYPE = "application/vnd.cncf.model.manifest.v1+json"
CONFIG_TYPE = "application/vnd.cncf.model.config.v1+json"
FILEPATH = "org.cncf.model.filepath"
#: The name ORAS gives a layer's file when it pulls it
TITLE = "org.opencontainers.image.title"
#: Files of a repo that are no part of its model
SKIPPED = {".gitattributes"}
WEIGHTS = (".safetensors", ".bin", ".gguf", ".onnx", ".pt", ".pth", ".ckpt", ".h5", ".msgpack", ".tflite")
DOCS = ("LICENSE", "NOTICE", "README")
#: safetensors' dtype names, as shem's manifests name precisions
PRECISIONS = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64"}


@dataclass
class File:
    path: str
    size: int
    #: Whether the Hub keeps it in large-file storage, behind its CDN
    large: bool


@dataclass
class Artifact:
    repo: str
    commit: str
    manifest: bytes
    #: The blobs the artifact makes, its manifest and config, by digest
    blobs: dict[str, bytes]
    #: The repo's files, by digest
    files: dict[str, File]

    @property
    def descriptor(self) -> dict:
        return {
            "mediaType": MANIFEST_TYPE,
            "artifactType": ARTIFACT_TYPE,
            "digest": "sha256:" + hashlib.sha256(self.manifest).hexdigest(),
            "size": len(self.manifest),
        }


def kind(path: str) -> str:
    """The ModelPack kind of a file: weights, a document, code, or the
    configs, tokenizers and the rest a model reads besides its weights."""
    name = path.rsplit("/", 1)[-1]
    if name.endswith(WEIGHTS):
        return "weight"
    if name.startswith(DOCS) or name.endswith(".md"):
        return "doc"
    if name.endswith(".py"):
        return "code"
    return "weight.config"


def param_size(count: int) -> str:
    """A parameter count as ModelPack writes sizes, such as `596m` or `1.5b`."""
    for scale, suffix in ((1e12, "t"), (1e9, "b"), (1e6, "m"), (1e3, "k")):
        if count >= scale:
            return f"{count / scale:.1f}".removesuffix(".0") + suffix
    return str(count)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build(api, repo: str, commit: str) -> Artifact:
    """The artifact of `repo` at `commit`."""
    info = api.model_info(repo, revision=commit, files_metadata=True)
    layers, files = [], {}
    for sibling in sorted(info.siblings, key=lambda s: s.rfilename):
        path = sibling.rfilename
        if path in SKIPPED:
            continue
        lfs = sibling.lfs
        if lfs:
            hex, size = lfs.sha256, lfs.size
        else:
            hex = sha256(Path(api.hf_hub_download(info.id, path, revision=info.sha)))
            size = sibling.size
        files[hex] = File(path, size, bool(lfs))
        layers.append(
            {
                "mediaType": f"application/vnd.cncf.model.{kind(path)}.v1.raw",
                "digest": f"sha256:{hex}",
                "size": size,
                "annotations": {FILEPATH: path, TITLE: path},
            }
        )
    if not layers:
        raise ValueError(f"{repo} holds no files")
    url = f"https://huggingface.co/{info.id}"
    org, _, name = info.id.rpartition("/")
    descriptor = {"name": name, "vendor": org, "revision": info.sha, "sourceURL": url, "docURL": url}
    card = info.card_data.to_dict() if info.card_data else {}
    if isinstance(card.get("license"), str):
        descriptor["licenses"] = [card["license"]]
    weights = [layer["annotations"][FILEPATH] for layer in layers if kind(layer["annotations"][FILEPATH]) == "weight"]
    formats = [ext.lstrip(".") for ext in WEIGHTS if any(w.endswith(ext) for w in weights)]
    settings: dict = {"architecture": "transformer"}
    if formats:
        settings["format"] = formats[0]
    if info.safetensors:
        settings["paramSize"] = param_size(info.safetensors.total)
        settings["precision"] = ",".join(PRECISIONS.get(d, d.lower()) for d in sorted(info.safetensors.parameters))
    config = {
        "descriptor": descriptor,
        "config": settings,
        # Raw layers are their own uncompressed content
        "modelfs": {"type": "layers", "diffIds": [layer["digest"] for layer in layers]},
    }
    config_bytes = json.dumps(config, separators=(",", ":")).encode()
    config_hex = hashlib.sha256(config_bytes).hexdigest()
    manifest = {
        "schemaVersion": 2,
        "mediaType": MANIFEST_TYPE,
        "artifactType": ARTIFACT_TYPE,
        "config": {"mediaType": CONFIG_TYPE, "digest": f"sha256:{config_hex}", "size": len(config_bytes)},
        "layers": layers,
        "annotations": {"org.opencontainers.image.source": url, "org.opencontainers.image.revision": info.sha},
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    blobs = {config_hex: config_bytes, hashlib.sha256(manifest_bytes).hexdigest(): manifest_bytes}
    return Artifact(info.id, info.sha, manifest_bytes, blobs, files)
