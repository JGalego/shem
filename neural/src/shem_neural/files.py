"""Hugging Face repos that hold a checkpoint, as images of their files.

Docker, containerd and Kubernetes take images, not artifacts: layers that
are tars, listed under the host's platform. So `shem registry` serves the
repo ORG/NAME of a checkpoint as an image too, at `files/ORG/NAME`: the
repo's files at the root of a filesystem, for a Kubernetes image volume or a
Docker image mount, or for a Dockerfile to `COPY --from`. No registry keeps
a copy of the weights for it, this one included.

A repo often holds its weights in several formats, and a runtime reads one.
A format's name in place of `files`, as in `safetensors/ORG/NAME`, is the
image without the files of the other formats.

The small files share a layer, which the registry makes and serves. Each
large file, of weights as a rule, is a layer of its own, or shares one with its neighbours
when there would be more layers than a filesystem stacks: a tar of the
file, which is the file between a header and padding. A tar's digest is not
the file's, the one the Hub knows, so the registry reads the file through
once to learn it, and from then on sends the layer as it reads the file from
the Hub again. It keeps the digests, by the files they are of, and nothing
else. The same commit makes the same image.
"""

import hashlib
import io
import json
import tarfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from shem_neural.checkpoints import FILEPATH, MANIFEST_TYPE, WEIGHTS, Artifact, File

INDEX_TYPE = "application/vnd.oci.image.index.v1+json"
CONFIG_TYPE = "application/vnd.oci.image.config.v1+json"
LAYER_TYPE = "application/vnd.oci.image.layer.v1.tar"
#: The platforms the image is listed under: it holds files, which run nowhere
ARCHITECTURES = ("amd64", "arm64")
#: What an image of a repo's files is named by, before the repo: `files`,
#: for all of them, or the format of weights it keeps
ALL = "files"
VIEWS = (ALL, *(extension.lstrip(".") for extension in WEIGHTS))
#: Overlay filesystems stack at most about 128 layers, so a repo with more
#: files of weights than this shares layers between neighbours
MAX_LAYERS = 96
BLOCK = 512
#: What ends a tar: two empty blocks
END = bytes(2 * BLOCK)


def header(path: str, size: int) -> bytes:
    """What a tar holds before a file's bytes: its path and size, owned by
    root and written at no time, so that it is the same wherever it is made."""
    info = tarfile.TarInfo(path)
    info.size, info.mode, info.mtime = size, 0o644, 0
    return info.tobuf(tarfile.PAX_FORMAT)


def padding(size: int) -> bytes:
    """What a tar holds after a file's bytes: the rest of its last block."""
    return bytes(-size % BLOCK)


@dataclass
class Layer:
    """A tar of some of a repo's files, in order: (the file's sha256, the file)."""

    files: list[tuple[str, File]]

    @property
    def key(self) -> tuple:
        """What the tar's bytes depend on, and so its digest."""
        return tuple((file.path, hex) for hex, file in self.files)

    def parts(self) -> list[bytes | File]:
        """The tar, as its own bytes and the files whose bytes go between."""
        parts: list[bytes | File] = []
        for _, file in self.files:
            parts += [header(file.path, file.size), file, padding(file.size)]
        return [*parts, END]

    @property
    def size(self) -> int:
        return sum(len(part) if isinstance(part, bytes) else part.size for part in self.parts())

    def read(self, read: Callable[[File, int, int], Iterator[bytes]], start: int = 0, end: int | None = None):
        """The tar's bytes from `start` to `end`, in pieces: `read` gives a
        file's from one offset to another."""
        end = self.size if end is None else end
        at = 0
        for part in self.parts():
            size = len(part) if isinstance(part, bytes) else part.size
            first, last = max(start - at, 0), min(end - at, size)
            if first < last:
                if isinstance(part, bytes):
                    yield part[first:last]
                else:
                    yield from read(part, first, last)
            at += size

    def digest(self, read: Callable[[File, int, int], Iterator[bytes]]) -> str:
        """The sha256 of the tar, which takes reading every file through."""
        digest = hashlib.sha256()
        for chunk in self.read(read):
            digest.update(chunk)
        return digest.hexdigest()


def formats(path: str) -> set[str]:
    """The formats of weights a file belongs to, by the extensions in its
    name: `model.onnx_data` is ONNX's, and `tokenizer.json` no format's."""
    name = path.rsplit("/", 1)[-1]
    return {extension.lstrip(".") for extension in WEIGHTS if extension in name}


def plan(artifact: Artifact, view: str = ALL, small: int = 1 << 20) -> tuple[Layer | None, list[Layer]]:
    """The layers of the image of `artifact`'s files: one of the small
    files, if there are any, and those of the files of weights, which are
    the files the Hub keeps in large-file storage that are larger than
    `small`. A `view` that names a format leaves out the files of the others."""
    # Every file the artifact lists: files with the same bytes are one blob of two paths
    files = []
    for layer in json.loads(artifact.manifest)["layers"]:
        hex, path = layer["digest"].removeprefix("sha256:"), layer["annotations"][FILEPATH]
        if view == ALL or not formats(path) or view in formats(path):
            files.append((hex, File(path, layer["size"], artifact.files[hex].large and layer["size"] > small)))
    if view != ALL and not any(view in formats(file.path) for _, file in files):
        raise ValueError(f"{artifact.repo} has no {view} weights")
    files.sort(key=lambda item: item[1].path)
    few = [item for item in files if not item[1].large]
    large = [item for item in files if item[1].large]
    per_layer = max(-(-len(large) // (MAX_LAYERS - 1)), 1)
    weights = [Layer(large[at : at + per_layer]) for at in range(0, len(large), per_layer)]
    return (Layer(few) if few else None), weights


@dataclass
class Image:
    repo: str
    commit: str
    #: The index that lists the image under each platform
    index: bytes
    #: The blobs the registry serves itself, by digest: the index, each
    #: platform's manifest and config, and the layer of small files
    blobs: dict[str, bytes]
    #: The layers of weights, by digest, sent as the Hub sends their files
    layers: dict[str, Layer]

    @property
    def descriptor(self) -> dict:
        return {"mediaType": INDEX_TYPE, "digest": "sha256:" + sha256(self.index), "size": len(self.index)}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: dict) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def labels(artifact: Artifact) -> dict[str, str]:
    """The standard OCI labels the artifact's model config fills in."""
    manifest = json.loads(artifact.manifest)
    config = json.loads(artifact.blobs[manifest["config"]["digest"].removeprefix("sha256:")])
    descriptor = config.get("descriptor") or {}
    fields = {
        "title": descriptor.get("name"),
        "vendor": descriptor.get("vendor"),
        "source": descriptor.get("sourceURL"),
        "revision": descriptor.get("revision"),
        "licenses": " AND ".join(descriptor.get("licenses") or []),
    }
    return {f"org.opencontainers.image.{key}": value for key, value in fields.items() if value}


def build(artifact: Artifact, small: tuple[Layer, bytes] | None, weights: list[tuple[Layer, str]]) -> Image:
    """The image of `artifact`'s files: its layer of `small` files, as the
    tar itself, and its layers of `weights`, each with its digest."""
    blobs: dict[str, bytes] = {}
    layers: dict[str, Layer] = {}
    descriptors = []
    if small:
        layer, data = small
        blobs[sha256(data)] = data
        descriptors.append({"mediaType": LAYER_TYPE, "digest": "sha256:" + sha256(data), "size": len(data)})
    for layer, hex in weights:
        layers[hex] = layer
        descriptors.append({"mediaType": LAYER_TYPE, "digest": f"sha256:{hex}", "size": layer.size})
    annotations = labels(artifact)
    entries = []
    for architecture in ARCHITECTURES:
        config = encode(
            {
                "architecture": architecture,
                "os": "linux",
                "config": {"Labels": annotations},
                # Uncompressed layers are their own content
                "rootfs": {"type": "layers", "diff_ids": [layer["digest"] for layer in descriptors]},
            }
        )
        manifest = encode(
            {
                "schemaVersion": 2,
                "mediaType": MANIFEST_TYPE,
                "config": {"mediaType": CONFIG_TYPE, "digest": "sha256:" + sha256(config), "size": len(config)},
                "layers": descriptors,
                "annotations": annotations,
            }
        )
        blobs[sha256(config)] = config
        blobs[sha256(manifest)] = manifest
        entries.append(
            {
                "mediaType": MANIFEST_TYPE,
                "digest": "sha256:" + sha256(manifest),
                "size": len(manifest),
                "platform": {"architecture": architecture, "os": "linux"},
            }
        )
    index = encode({"schemaVersion": 2, "mediaType": INDEX_TYPE, "manifests": entries, "annotations": annotations})
    blobs[sha256(index)] = index
    return Image(artifact.repo, artifact.commit, index, blobs, layers)


def tar(layer: Layer, read: Callable[[File, int, int], Iterator[bytes]]) -> bytes:
    """A layer's tar, whole: for the layer of small files."""
    out = io.BytesIO()
    for chunk in layer.read(read):
        out.write(chunk)
    return out.getvalue()
