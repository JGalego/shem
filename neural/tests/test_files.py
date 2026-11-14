"""Images of checkpoints' files, without a Hub: how files are laid in
layers, and that a layer is a tar of them whichever part of it is read."""

import functools
import hashlib
import io
import json
import random
import tarfile

import pytest

from shem_neural import files
from shem_neural.checkpoints import FILEPATH, Artifact, File

DATA = {
    "config.json": b"{}",
    "model-00001-of-00002.safetensors": random.Random(1).randbytes(3000),
    "model-00002-of-00002.safetensors": random.Random(2).randbytes(1024),
    "onnx/" + "long" * 30 + "/model.onnx": random.Random(3).randbytes(700),
    "tokenizer.json": b'{"version": "1.0"}',
}


#: Layers as they would be were these files as large as weights are
plan = functools.partial(files.plan, small=0)


def artifact(data: dict[str, bytes], large=lambda path: path.endswith((".safetensors", ".onnx"))) -> Artifact:
    """An artifact of files with these paths and contents, as `checkpoints` makes one."""
    held, layers = {}, []
    for path, content in data.items():
        hex = hashlib.sha256(content).hexdigest()
        held[hex] = File(path, len(content), large(path))
        layers.append({"digest": f"sha256:{hex}", "size": len(content), "annotations": {FILEPATH: path}})
    descriptor = {"name": "clay", "vendor": "org", "sourceURL": "https://huggingface.co/org/clay", "licenses": ["mit"]}
    config = json.dumps({"descriptor": descriptor}).encode()
    digest = "sha256:" + hashlib.sha256(config).hexdigest()
    manifest = json.dumps({"config": {"digest": digest}, "layers": layers}).encode()
    return Artifact("org/clay", "c0ffee", manifest, {digest.removeprefix("sha256:"): config}, held)


def read(file: File, start: int, end: int):
    """A file's bytes, a few at a time."""
    data = DATA[file.path][start:end]
    yield from (data[at : at + 100] for at in range(0, len(data), 100))


def test_small_files_share_a_layer_and_weights_have_their_own():
    small, weights = plan(artifact(DATA))
    assert [file.path for _, file in small.files] == ["config.json", "tokenizer.json"]
    assert [[file.path for _, file in layer.files] for layer in weights] == [
        ["model-00001-of-00002.safetensors"],
        ["model-00002-of-00002.safetensors"],
        ["onnx/" + "long" * 30 + "/model.onnx"],
    ]
    # More files of weights than a filesystem stacks layers share them, in order,
    # as do files with the same bytes, which the artifact holds as one blob
    many = {f"shard-{i:04}.safetensors": bytes([i % 256]) for i in range(400)}
    small, weights = plan(artifact(many))
    assert small is None and len(weights) <= files.MAX_LAYERS
    assert [file.path for layer in weights for _, file in layer.files] == sorted(many)


def test_a_formats_image_leaves_out_the_other_formats_files():
    data = DATA | {"onnx/model.onnx_data": b"external", "model.safetensors.index.json": b"{}", "tokenizer.model": b"sp"}
    made = artifact(data, large=lambda path: files.formats(path) or path == "tokenizer.model")

    def paths(view: str) -> list[str]:
        small, weights = plan(made, view)
        return [file.path for layer in [small, *weights] for _, file in layer.files]

    assert sorted(paths("files")) == sorted(data)
    assert sorted(paths("safetensors")) == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    assert sorted(paths("onnx")) == [
        "config.json",
        "onnx/" + "long" * 30 + "/model.onnx",
        "onnx/model.onnx_data",
        "tokenizer.json",
        "tokenizer.model",
    ]
    with pytest.raises(ValueError, match="org/clay has no gguf weights"):
        plan(made, "gguf")
    # A file in large-file storage that is small all the same goes with the small files
    small, weights = plan(made, "safetensors", small=2000)
    assert [file.path for layer in weights for _, file in layer.files] == ["model-00001-of-00002.safetensors"]
    assert "model-00002-of-00002.safetensors" in [file.path for _, file in small.files]


def test_layers_are_tars_of_their_files_whichever_part_is_read():
    small, weights = plan(artifact(DATA))
    for layer in [small, *weights, files.Layer(small.files + weights[0].files + weights[2].files)]:
        whole = files.tar(layer, read)
        assert len(whole) == layer.size and layer.digest(read) == hashlib.sha256(whole).hexdigest()
        with tarfile.open(fileobj=io.BytesIO(whole)) as archive:
            held = {member.name: archive.extractfile(member).read() for member in archive}
        assert held == {file.path: DATA[file.path] for _, file in layer.files}
        spans = random.Random(len(whole))
        for _ in range(50):
            start = spans.randrange(len(whole))
            end = spans.randrange(start, len(whole) + 1)
            assert b"".join(layer.read(read, start, end)) == whole[start:end]
    # What a layer's digest depends on: its files' paths and contents
    assert weights[0].key == (
        ("model-00001-of-00002.safetensors", hashlib.sha256(DATA[weights[0].files[0][1].path]).hexdigest()),
    )


def test_images_list_the_layers_under_each_platform():
    made = artifact(DATA)
    small, weights = plan(made)
    image = files.build(made, (small, files.tar(small, read)), [(layer, layer.digest(read)) for layer in weights])
    index = json.loads(image.index)
    assert image.descriptor["digest"] == "sha256:" + hashlib.sha256(image.index).hexdigest()
    assert [entry["platform"]["architecture"] for entry in index["manifests"]] == list(files.ARCHITECTURES)
    labels = {
        "org.opencontainers.image.title": "clay",
        "org.opencontainers.image.vendor": "org",
        "org.opencontainers.image.source": "https://huggingface.co/org/clay",
        "org.opencontainers.image.licenses": "mit",
    }
    for entry in index["manifests"]:
        manifest = json.loads(image.blobs[entry["digest"].removeprefix("sha256:")])
        config = json.loads(image.blobs[manifest["config"]["digest"].removeprefix("sha256:")])
        assert config["config"]["Labels"] == manifest["annotations"] == labels
        digests = [layer["digest"].removeprefix("sha256:") for layer in manifest["layers"]]
        # The registry holds the layer of small files, and sends the others as it reads them
        assert digests[0] in image.blobs and digests[1:] == list(image.layers)
        assert [layer["size"] for layer in manifest["layers"][1:]] == [layer.size for layer in weights]
