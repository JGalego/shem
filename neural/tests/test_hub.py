"""Images in Hugging Face repos: fetched, pushed and unpacked as OCI layouts,
and served as a registry. A fake Hub keeps each repo's revisions in a
directory, so these need no network; with ORAS installed, a real client
pulls through the registry, following a large blob to where it is kept,
and logs in to a public one."""

import base64
import functools
import hashlib
import html
import http.server
import io
import json
import random
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from shem_neural import hub, registry

MANIFEST = "application/vnd.oci.image.manifest.v1+json"
#: Bytes that compress no smaller: a layer larger than the registry serves itself
WEIGHTS = random.Random(0).randbytes(16384)


def put(layout: Path, data: bytes) -> dict:
    digest = hashlib.sha256(data).hexdigest()
    (layout / "blobs" / "sha256").mkdir(parents=True, exist_ok=True)
    (layout / "blobs" / "sha256" / digest).write_bytes(data)
    return {"digest": f"sha256:{digest}", "size": len(data)}


def tar(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


def layout(dir: Path, files: list[dict[str, bytes]], process: dict | None = None) -> str:
    """An image layout in `dir` with a layer of each of `files`, the first
    compressed with zstd, and `process` as its config's. Returns the
    manifest's digest."""
    layers = []
    for i, layer in enumerate(files):
        data = tar(layer)
        if i == 0:
            compressed = zstandard.ZstdCompressor().compress(data)
            layers.append(put(dir, compressed) | {"mediaType": "application/vnd.oci.image.layer.v1.tar+zstd"})
        else:
            layers.append(put(dir, data) | {"mediaType": "application/vnd.oci.image.layer.v1.tar"})
    settings = {"architecture": "riscv32", "os": "linux", "config": process or {}, "rootfs": {"type": "layers"}}
    config = put(dir, json.dumps(settings).encode())
    config["mediaType"] = "application/vnd.oci.image.config.v1+json"
    manifest = {"schemaVersion": 2, "mediaType": MANIFEST, "config": config, "layers": layers}
    entry = put(dir, json.dumps(manifest).encode()) | {"mediaType": MANIFEST}
    (dir / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [entry]}))
    (dir / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    return entry["digest"]


#: A response as the Hub's errors read one, whichever HTTP library
#: huggingface_hub is built on
NOT_FOUND = SimpleNamespace(headers={}, request=None, status_code=404)
UNAUTHORIZED = SimpleNamespace(headers={}, request=None, status_code=401)
#: The one token the fake Hub knows, which reads its private and gated repos
TOKEN = "hf_clay"


class FakeApi:
    """A Hub whose repos are directories: ROOT/ORG/NAME/REVISION holds the
    files of each revision, `main` the default branch's."""

    endpoint = "https://hub.example"
    token = None

    def __init__(self, root: Path):
        self.root = root
        #: The branch each commit model_info gave is on
        self.commits: dict[str, str] = {}
        #: Repos only TOKEN may see, and repos anyone may see but only TOKEN read
        self.private: set[str] = set()
        self.gated: set[str] = set()

    def auth_check(self, repo):
        self.repo(repo)
        if repo in self.gated and self.token != TOKEN:
            raise GatedRepoError(f"{repo} is gated", response=UNAUTHORIZED)

    def whoami(self):
        if self.token != TOKEN:
            raise HfHubHTTPError("no such token", response=UNAUTHORIZED)
        return {"name": "clay"}

    def dir(self, repo: str, rev: str | None) -> Path:
        self.repo(repo)
        path = self.root / repo / self.commits.get(rev, rev or "main")
        if not path.is_dir():
            raise RevisionNotFoundError(f"{repo} has no revision {rev}", response=NOT_FOUND)
        return path

    def repo(self, repo: str) -> Path:
        if not (self.root / repo).is_dir() or (repo in self.private and self.token != TOKEN):
            raise RepositoryNotFoundError(f"no repo {repo}", response=NOT_FOUND)
        return self.root / repo

    def list_repo_files(self, repo, revision=None):
        base = self.dir(repo, revision)
        return [p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()]

    def hf_hub_download(self, repo, path, revision=None):
        # Whoever asks, as huggingface_hub's cache answers for a commit
        # it has kept
        token, self.token = self.token, TOKEN
        try:
            file = self.dir(repo, revision) / path
        finally:
            self.token = token
        if not file.is_file():
            raise EntryNotFoundError(f"{repo} has no {path}")
        return str(file)

    def snapshot_download(self, repo, revision=None, local_dir=None, allow_patterns=None):
        shutil.copytree(self.dir(repo, revision), local_dir, dirs_exist_ok=True)
        (Path(local_dir) / ".cache" / "huggingface").mkdir(parents=True)

    def model_info(self, repo, revision=None, files_metadata=False):
        """A revision's files, and a commit that changes with them; a file
        above 1 KB is in large-file storage, with its sha256."""
        base = self.dir(repo, revision)
        files = sorted(p for p in base.rglob("*") if p.is_file())
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.relative_to(base).as_posix().encode() + file.read_bytes())
        commit = digest.hexdigest()[:40]
        self.commits[commit] = base.name
        siblings = []
        for file in files:
            data = file.read_bytes()
            lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data)) if len(data) > 1024 else None
            siblings.append(SimpleNamespace(rfilename=file.relative_to(base).as_posix(), size=len(data), lfs=lfs))
        return SimpleNamespace(id=repo, sha=commit, siblings=siblings, card_data=None, safetensors=None)

    def list_repo_refs(self, repo):
        names = [p.name for p in self.repo(repo).iterdir()]
        return SimpleNamespace(branches=[SimpleNamespace(name=n) for n in names], tags=[])

    # What a push does
    def create_repo(self, repo, private=False, exist_ok=False):
        (self.root / repo / "main").mkdir(parents=True, exist_ok=True)

    def create_branch(self, repo, branch, exist_ok=False):
        if not (self.root / repo / branch).exists():
            shutil.copytree(self.root / repo / "main", self.root / repo / branch)

    def upload_folder(self, repo_id, folder_path, revision=None, allow_patterns=(), delete_patterns=(), **_):
        base = self.dir(repo_id, revision)
        for pattern in delete_patterns or ():
            for file in base.glob(pattern):
                file.unlink()
        for pattern in allow_patterns:
            for file in Path(folder_path).glob(pattern):
                target = base / file.relative_to(folder_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(file, target)
        return SimpleNamespace(commit_url=f"{self.endpoint}/{repo_id}/commit/1")


@pytest.fixture
def api(tmp_path):
    return FakeApi(tmp_path / "hub")


@pytest.fixture
def pushed(api, tmp_path):
    """An image pushed to org/clay, with a large layer and a small one."""
    image = tmp_path / "image"
    digest = layout(image, [{"model/weights": WEIGHTS}, {"model/shem.json": b"{}"}])
    hub.push(image, "org/clay", api=api)
    return digest


def test_latest_is_the_default_branch():
    assert hub.revision(None) is None
    assert hub.revision("latest") is None
    assert hub.revision("v1") == "v1"


def test_pushed_layouts_are_fetched_back(api, pushed, tmp_path):
    out = tmp_path / "fetched"
    assert hub.fetch("org/clay", out, api=api)
    assert json.loads((out / "index.json").read_text())["manifests"][0]["digest"] == pushed
    assert not (out / ".cache").exists()


def test_repos_without_a_layout_are_left_to_the_importer(api, tmp_path):
    (api.root / "org/checkpoint/main").mkdir(parents=True)
    (api.root / "org/checkpoint/main/config.json").write_text("{}")
    assert not hub.fetch("org/checkpoint", tmp_path / "a", api=api)
    # A repo of GGUF files has no revision named for a quantization
    assert not hub.fetch("org/checkpoint", tmp_path / "b", "Q8_0", api=api)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_pushes_to_a_branch_leave_main_alone(api, pushed, tmp_path):
    other = tmp_path / "other"
    digest = layout(other, [{"model/shem.json": b'{"v": 2}'}])
    hub.push(other, "org/clay", "v2", api=api)
    main = json.loads((api.root / "org/clay/main/index.json").read_text())
    v2 = json.loads((api.root / "org/clay/v2/index.json").read_text())
    assert main["manifests"][0]["digest"] == pushed
    assert v2["manifests"][0]["digest"] == digest


def test_pushes_remove_only_the_blobs_earlier_pushes_had(api, pushed, tmp_path):
    other = tmp_path / "other"
    layout(other, [{"model/shem.json": b'{"v": 2}'}])
    hub.push(other, "org/clay", api=api)
    remote = {p.name for p in (api.root / "org/clay/main/blobs/sha256").iterdir()}
    assert remote == {p.name for p in (other / "blobs/sha256").iterdir()}


def test_layouts_unpack_into_their_filesystem(tmp_path):
    image = tmp_path / "image"
    layout(image, [{"model/a": b"zstd"}, {"model/b": b"plain"}])
    hub.unpack(image, tmp_path / "root")
    assert (tmp_path / "root/model/a").read_bytes() == b"zstd"
    assert (tmp_path / "root/model/b").read_bytes() == b"plain"


@pytest.fixture
def serving(api, monkeypatch):
    """Start registries over the fake Hub, with large blobs where a CDN
    keeps them: a server of their own, as Hugging Face's are elsewhere."""
    cdn = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=api.root)
    )

    def metadata(url, token=None):
        # https://hub.example/ORG/NAME/resolve/REVISION/PATH
        repo, _, rest = url.removeprefix(f"{api.endpoint}/").partition("/resolve/")
        rev, _, path = rest.partition("/")
        file = api.dir(repo, None if rev == "main" else rev) / path
        if not file.is_file():
            raise EntryNotFoundError(url)
        where = file.relative_to(api.root).as_posix()
        return SimpleNamespace(size=file.stat().st_size, location=f"http://127.0.0.1:{cdn.server_port}/{where}")

    monkeypatch.setattr(registry, "get_hf_file_metadata", metadata)
    # Between a tar of a small file, a 10 KiB record, and the weights
    monkeypatch.setattr(registry, "SMALL", 12 * 1024)
    servers = [cdn]
    threading.Thread(target=cdn.serve_forever, daemon=True).start()

    def start(public=False, limit=None):
        server = registry.serve(("127.0.0.1", 0), registry.Hub(api), public, limit)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for s in servers:
        s.shutdown()


@pytest.fixture
def served(serving, pushed):
    return serving()


def get(server, path: str, headers: dict | None = None, method: str = "GET"):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def test_tags_are_manifests(served, pushed):
    status, headers, body = get(served, "/v2/org/clay/manifests/latest")
    assert status == 200
    assert headers["Docker-Content-Digest"] == pushed
    assert headers["Content-Type"] == MANIFEST
    assert "sha256:" + hashlib.sha256(body).hexdigest() == pushed
    assert get(served, f"/v2/org/clay/manifests/{pushed}")[2] == body


def test_large_blobs_redirect_and_small_ones_are_served(served, api):
    manifest = json.loads(get(served, "/v2/org/clay/manifests/latest")[2])
    large, small = manifest["layers"]
    status, headers, _ = get(served, f"/v2/org/clay/blobs/{large['digest']}")
    assert status == 307
    assert headers["Location"].endswith(large["digest"].removeprefix("sha256:"))
    status, _, body = get(served, f"/v2/org/clay/blobs/{small['digest']}")
    assert status == 200
    assert "sha256:" + hashlib.sha256(body).hexdigest() == small["digest"]


def test_registry_answers_in_the_distribution_specs_errors(served):
    assert get(served, "/v2/")[0] == 200
    status, _, body = get(served, "/v2/org/clay/manifests/v9")
    assert (status, json.loads(body)["errors"][0]["code"]) == (404, "MANIFEST_UNKNOWN")
    status, _, body = get(served, "/v2/org/clay/blobs/sha256:" + "0" * 64)
    assert (status, json.loads(body)["errors"][0]["code"]) == (404, "BLOB_UNKNOWN")
    assert json.loads(get(served, "/v2/org/clay/tags/list")[2]) == {"name": "org/clay", "tags": ["latest"]}


def test_repos_without_files_are_no_artifacts(served, api):
    (api.root / "org/empty/main").mkdir(parents=True)
    (api.root / "org/empty/main/.gitattributes").write_text("")
    status, _, body = get(served, "/v2/org/empty/manifests/latest")
    error = json.loads(body)["errors"][0]
    assert (status, error["code"]) == (404, "MANIFEST_UNKNOWN")
    assert error["message"] == "org/empty holds no files"


def test_registry_is_read_only(served):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", served.server_port)
    connection.request("PUT", "/v2/org/clay/manifests/latest", body=b"{}")
    response = connection.getresponse()
    assert response.status == 405
    assert json.loads(response.read())["errors"][0]["code"] == "UNSUPPORTED"


@pytest.mark.skipif(not shutil.which("oras"), reason="needs ORAS")
def test_oras_pulls_through_the_registry(served, pushed, tmp_path):
    out = tmp_path / "pulled"
    reference = f"127.0.0.1:{served.server_port}/org/clay:latest"
    subprocess.run(
        ["oras", "cp", "--from-plain-http", reference, "--to-oci-layout", f"{out}:latest"],
        check=True,
        capture_output=True,
    )
    hub.unpack(out, tmp_path / "root")
    assert (tmp_path / "root/model/weights").read_bytes() == WEIGHTS


SHEM_JSON = {
    "format": "shem.model/v2",
    "source": {
        "repo": "Org/Clay-1B",
        "revision": "0123456789abcdef",
        "architecture": "LlamaForCausalLM",
        "spdx": "MIT",
    },
    "parameters": 1_230_000_000,
    "precision": ["bfloat16"],
    "blocks": [{"type": "attention"}, {"type": "mlp"}, {"type": "attention"}, {"type": "mlp"}],
}


@pytest.fixture
def model(api, tmp_path):
    """A model image pushed to org/golem: a system prompt primed into a
    session, as its last layer, and a comparison with transformers."""
    image = tmp_path / "model"
    process = {
        "Entrypoint": ["shem:generate", "--system", "You are a <golem>."],
        "Labels": {"dev.shem.machine": "rv32im", "dev.shem.model": "/model", "org.example": "kept"},
    }
    files = [
        {"model/shem.json": json.dumps(SHEM_JSON).encode()},
        {"model/weights": WEIGHTS},
        {".shem/session.json": b"{}"},
    ]
    layout(image, files, process)
    hub.push(image, "org/golem", api=api)
    (api.root / "org/golem/main/comparison.json").write_text(
        json.dumps({"difference": 8.1e-5, "tokens": True, "length": 24, "passed": True})
    )
    hub.push(image, "org/golem", "modelpack", api=api)


def test_home_page_lists_the_repos_served(served, model):
    status, headers, body = get(served, "/")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert b"org/golem" not in body
    get(served, "/v2/org/golem/manifests/latest")
    assert b'href="/org/golem"' in get(served, "/")[2]
    # What a page of the repo shows of latest comes to its card
    get(served, "/org/golem")
    card = get(served, "/")[2].decode()
    assert "✓ verified" in card and "1.23B params" in card and "1 pull" in card


def test_repo_pages_show_the_image_and_its_model(served, model):
    status, _, body = get(served, "/org/golem")
    page = html.unescape(body.decode())
    assert status == 200
    # How to run and pull it, from this registry
    assert 'shem run hf.co/org/golem "What is a golem?"' in page
    assert f"oras pull 127.0.0.1:{served.server_port}/org/golem:latest" in page
    # The image, its model and its comparison
    assert "You are a &lt;golem&gt;." in body.decode()
    assert "Primed session" in page
    assert "Org/Clay-1B" in page and "0123456789ab" in page and "1.23B" in page
    assert "2 attention, 2 mlp" in page
    assert "8.10e-05" in page and "✓ Verified" in page
    assert '<a href="/org/golem?tag=modelpack">modelpack</a>' in page
    assert "org.example" in page and "dev.shem.model" not in page.split("<summary>Labels</summary>")[1]


def test_repo_pages_take_a_tag_and_explain_failures(served, model):
    page = get(served, "/org/golem?tag=modelpack")[2].decode()
    assert 'class="chip on" href="/org/golem?tag=modelpack" aria-current="page">modelpack<' in page
    status, headers, _ = get(served, "/go?repo=hf.co/org/golem/")
    assert (status, headers["Location"]) == (302, "/org/golem")
    status, headers, body = get(served, "/org/nothing-here")
    assert status == 404 and headers["Content-Type"].startswith("text/html")
    assert b"<h1>404 Not Found</h1>" in body
    assert get(served, "/org/golem?tag=no/such")[0] == 404


@pytest.fixture
def checkpoint(api):
    """A repo that holds a checkpoint: small configs, a document, large
    weights, and what Git keeps of its own."""
    base = api.root / "org/ckpt/main"
    (base / "sub").mkdir(parents=True)
    files = {
        "config.json": b'{"architectures": ["LlamaForCausalLM"]}',
        "README.md": b"# A checkpoint",
        "model.safetensors": WEIGHTS,
        "sub/tokenizer.json": b"{}",
        ".gitattributes": b"*.safetensors filter=lfs",
    }
    for path, data in files.items():
        (base / path).write_bytes(data)
    return files


def test_checkpoint_repos_are_modelpack_artifacts_of_their_files(served, checkpoint):
    status, headers, body = get(served, "/v2/org/ckpt/manifests/latest")
    manifest = json.loads(body)
    assert status == 200 and manifest["artifactType"] == "application/vnd.cncf.model.manifest.v1+json"
    assert headers["Docker-Content-Digest"] == "sha256:" + hashlib.sha256(body).hexdigest()
    layers = {layer["annotations"]["org.cncf.model.filepath"]: layer for layer in manifest["layers"]}
    assert sorted(layers) == ["README.md", "config.json", "model.safetensors", "sub/tokenizer.json"]
    for path, layer in layers.items():
        assert layer["digest"] == "sha256:" + hashlib.sha256(checkpoint[path]).hexdigest()
        assert layer["annotations"]["org.opencontainers.image.title"] == path
    kinds = {path: layer["mediaType"].split(".model.")[1] for path, layer in layers.items()}
    assert kinds == {
        "README.md": "doc.v1.raw",
        "config.json": "weight.config.v1.raw",
        "model.safetensors": "weight.v1.raw",
        "sub/tokenizer.json": "weight.config.v1.raw",
    }
    config = json.loads(get(served, f"/v2/org/ckpt/blobs/{manifest['config']['digest']}")[2])
    assert config["config"]["format"] == "safetensors"
    assert config["modelfs"]["diffIds"] == [layer["digest"] for layer in manifest["layers"]]
    # The same commit makes the same artifact
    assert get(served, "/v2/org/ckpt/manifests/latest")[2] == body
    # Small files are served, and large ones redirected to where the Hub keeps them
    status, _, data = get(served, f"/v2/org/ckpt/blobs/{layers['config.json']['digest']}")
    assert (status, data) == (200, checkpoint["config.json"])
    status, headers, _ = get(served, f"/v2/org/ckpt/blobs/{layers['model.safetensors']['digest']}")
    assert status == 307 and headers["Location"].endswith("org/ckpt/main/model.safetensors")


def test_checkpoint_pages_list_their_files(served, checkpoint):
    page = html.unescape(get(served, "/org/ckpt")[2].decode())
    assert "ModelPack" in page and "model.safetensors" in page and "sub/tokenizer.json" in page
    assert 'shem run hf.co/org/ckpt "What is a golem?"' in page


@pytest.mark.skipif(not shutil.which("oras"), reason="needs ORAS")
def test_oras_pulls_a_checkpoints_files(served, checkpoint, tmp_path):
    reference = f"127.0.0.1:{served.server_port}/org/ckpt:latest"
    command = ["oras", "pull", "--plain-http", "-o", str(tmp_path / "out"), reference]
    subprocess.run(command, check=True, capture_output=True)
    for path, data in checkpoint.items():
        if path != ".gitattributes":
            assert (tmp_path / "out" / path).read_bytes() == data


def unpacked(server, repo: str, manifest: dict) -> dict[str, bytes]:
    """The files an image's layers hold, each layer checked against its descriptor."""
    found = {}
    for layer in manifest["layers"]:
        status, headers, data = get(server, f"/v2/{repo}/blobs/{layer['digest']}")
        assert status == 200 and int(headers["Content-Length"]) == layer["size"] == len(data)
        assert "sha256:" + hashlib.sha256(data).hexdigest() == layer["digest"]
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive:
                assert (member.mode, member.uid, member.mtime) == (0o644, 0, 0)
                found[member.name] = archive.extractfile(member).read()
    return found


def test_checkpoint_repos_are_images_of_their_files_too(served, checkpoint):
    status, headers, body = get(served, "/v2/files/org/ckpt/manifests/latest")
    index = json.loads(body)
    assert status == 200 and headers["Content-Type"] == index["mediaType"] == "application/vnd.oci.image.index.v1+json"
    assert headers["Docker-Content-Digest"] == "sha256:" + hashlib.sha256(body).hexdigest()
    platforms = [entry["platform"] for entry in index["manifests"]]
    assert platforms == [{"architecture": a, "os": "linux"} for a in ("amd64", "arm64")]
    files = {path: data for path, data in checkpoint.items() if path != ".gitattributes"}
    layers = None
    for entry in index["manifests"]:
        status, headers, body = get(served, f"/v2/files/org/ckpt/manifests/{entry['digest']}")
        manifest = json.loads(body)
        assert status == 200 and headers["Content-Type"] == MANIFEST == entry["mediaType"]
        config = json.loads(get(served, f"/v2/files/org/ckpt/blobs/{manifest['config']['digest']}")[2])
        assert (config["architecture"], config["os"]) == (entry["platform"]["architecture"], "linux")
        # Uncompressed layers, so what they unpack to is what they are
        assert config["rootfs"]["diff_ids"] == [layer["digest"] for layer in manifest["layers"]]
        assert {layer["mediaType"] for layer in manifest["layers"]} == {"application/vnd.oci.image.layer.v1.tar"}
        # Every platform's image is of the same layers: the small files, then the weights
        assert layers in (None, manifest["layers"])
        layers = manifest["layers"]
        assert unpacked(served, "files/org/ckpt", manifest) == files
    assert len(layers) == 2 and layers[1]["size"] == 512 + len(WEIGHTS) + 1024
    # The same commit makes the same image, and its tags are the repo's
    assert json.loads(get(served, "/v2/files/org/ckpt/manifests/latest")[2]) == index
    assert json.loads(get(served, "/v2/files/org/ckpt/tags/list")[2]) == {"name": "files/org/ckpt", "tags": ["latest"]}
    # A layer of weights is as long as it says before it is sent, and sends the part asked for
    blob = f"/v2/files/org/ckpt/blobs/{layers[1]['digest']}"
    whole = get(served, blob)[2]
    status, headers, body = get(served, blob, method="HEAD")
    assert (status, int(headers["Content-Length"]), body) == (200, len(whole), b"")
    for asked, start, end in [
        ("bytes=500-5000", 500, 5001),
        ("bytes=16000-", 16000, len(whole)),
        ("bytes=-700", -700, None),
    ]:
        status, headers, body = get(served, blob, {"Range": asked})
        first, last = range(len(whole))[start:end][0], range(len(whole))[start:end][-1]
        assert (status, body) == (206, whole[start:end])
        assert headers["Content-Range"] == f"bytes {first}-{last}/{len(whole)}"
    # The artifact is still what the repo's own name gives
    assert "artifactType" in json.loads(get(served, "/v2/org/ckpt/manifests/latest")[2])
    page = html.unescape(get(served, "/org/ckpt")[2].decode())
    assert f"docker pull 127.0.0.1:{served.server_port}/files/org/ckpt:latest" in page
    assert "from the Hub, as tars" in get(served, "/")[2].decode()


def test_weights_are_read_through_once_whatever_else_changes(serving, api, checkpoint, monkeypatch):
    server = serving()
    asked = []
    metadata = registry.get_hf_file_metadata
    monkeypatch.setattr(registry, "get_hf_file_metadata", lambda url, token=None: asked.append(url) or metadata(url))
    first = json.loads(get(server, "/v2/files/org/ckpt/manifests/latest")[2])
    assert len(asked) == 1 and asked[0].endswith("/model.safetensors")
    (api.root / "org/ckpt/main/README.md").write_text("# The same checkpoint, said otherwise")
    second = json.loads(get(server, "/v2/files/org/ckpt/manifests/latest")[2])
    assert second != first and len(asked) == 1

    def layers(index: dict) -> list[str]:
        manifest = get(server, f"/v2/files/org/ckpt/manifests/{index['manifests'][0]['digest']}")[2]
        return [layer["digest"] for layer in json.loads(manifest)["layers"]]

    # Another layer of small files, over the same layer of weights
    assert layers(first)[0] != layers(second)[0] and layers(first)[1] == layers(second)[1]


def test_a_formats_image_is_of_no_other_formats_files(served, api, checkpoint):
    (api.root / "org/ckpt/main/model.onnx").write_bytes(WEIGHTS[::-1])

    def files(view: str) -> list[str]:
        index = json.loads(get(served, f"/v2/{view}/org/ckpt/manifests/latest")[2])
        manifest = get(served, f"/v2/{view}/org/ckpt/manifests/{index['manifests'][0]['digest']}")[2]
        return sorted(unpacked(served, f"{view}/org/ckpt", json.loads(manifest)))

    shared = ["README.md", "config.json", "sub/tokenizer.json"]
    assert files("files") == sorted([*shared, "model.onnx", "model.safetensors"])
    assert files("safetensors") == sorted([*shared, "model.safetensors"])
    assert files("onnx") == sorted([*shared, "model.onnx"])
    # A checkpoint has images of the formats it has
    status, _, body = get(served, "/v2/gguf/org/ckpt/manifests/latest")
    assert (status, json.loads(body)["errors"][0]["message"]) == (404, "org/ckpt has no gguf weights")
    assert json.loads(get(served, "/v2/onnx/org/ckpt/tags/list")[2])["name"] == "onnx/org/ckpt"


def test_images_of_files_are_of_checkpoints_within_the_limit(serving, pushed, checkpoint):
    status, _, body = get(serving(), "/v2/files/org/clay/manifests/latest")
    error = json.loads(body)["errors"][0]
    assert (status, error["code"]) == (404, "MANIFEST_UNKNOWN")
    assert error["message"] == "org/clay holds an image, not a checkpoint: pull it without files/"
    limited = serving(limit=len(WEIGHTS) - 1)
    status, _, body = get(limited, "/v2/files/org/ckpt/manifests/latest")
    message = json.loads(body)["errors"][0]["message"]
    assert status == 404 and message.startswith("org/ckpt has 16.4 KB of weights, and this registry makes images")
    # Its artifact sends nothing through the registry, so it is served whatever its size
    assert get(limited, "/v2/org/ckpt/manifests/latest")[0] == 200
    assert get(serving(limit=len(WEIGHTS)), "/v2/files/org/ckpt/manifests/latest")[0] == 200


def test_open_pages_learn_when_they_change(served, api, tmp_path):
    def version(path):
        return json.loads(get(served, path)[2])["version"]

    page = get(served, "/")[2].decode()
    home = version("/live")
    assert f'<meta name="live" content="{home}"' in page
    # A pull changes what the home page shows
    get(served, "/v2/org/clay/manifests/latest")
    assert version("/live") != home
    # A push changes the commit a repo's page shows
    before = version("/live?repo=org/clay&tag=latest")
    assert f'content="{before}"' in get(served, "/org/clay")[2].decode()
    other = tmp_path / "other"
    layout(other, [{"model/shem.json": b'{"v": 2}'}])
    hub.push(other, "org/clay", api=api)
    assert version("/live?repo=org/clay&tag=latest") != before


@pytest.fixture
def public(serving, api, pushed, tmp_path):
    """A public registry, with a private image and a gated one beside org/clay."""
    for repo in ("org/secret", "org/gated"):
        hub.push(tmp_path / "image", repo, api=api)
    api.private.add("org/secret")
    api.gated.add("org/gated")
    return serving(public=True)


def basic(password: str) -> dict:
    return {"Authorization": "Basic " + base64.b64encode(f"user:{password}".encode()).decode()}


def test_public_registries_give_tokens_for_hugging_faces(public):
    port = public.server_port
    status, headers, _ = get(public, "/v2/")
    assert status == 401
    assert headers["WWW-Authenticate"] == f'Bearer realm="http://127.0.0.1:{port}/token",service="127.0.0.1:{port}"'
    # Behind a proxy that says it was asked over HTTPS
    status, headers, _ = get(public, "/v2/", {"X-Forwarded-Proto": "https"})
    assert headers["WWW-Authenticate"].startswith(f'Bearer realm="https://127.0.0.1:{port}/token"')
    assert json.loads(get(public, "/token")[2])["token"] == "anonymous"
    assert json.loads(get(public, "/token", basic(TOKEN))[2])["token"] == TOKEN
    status, _, body = get(public, "/token", basic("hf_mud"))
    assert (status, json.loads(body)["errors"][0]["code"]) == (401, "UNAUTHORIZED")
    assert get(public, "/v2/", {"Authorization": "Bearer anonymous"})[0] == 200


def test_public_registries_serve_repos_as_the_hub_would_to_the_client(public, pushed):
    anonymous = {"Authorization": "Bearer anonymous"}
    logged_in = {"Authorization": f"Bearer {TOKEN}"}
    assert get(public, "/v2/org/clay/manifests/latest", anonymous)[0] == 200
    for repo in ("org/secret", "org/gated"):
        status, headers, _ = get(public, f"/v2/{repo}/manifests/latest", anonymous)
        assert status == 401 and headers["WWW-Authenticate"].endswith(f',scope="repository:{repo}:pull"')
        status, _, body = get(public, f"/v2/{repo}/manifests/latest", logged_in)
        assert status == 200
    # What the registry keeps of a repo, it serves only to whoever may read it
    small = json.loads(body)["layers"][1]["digest"]
    assert get(public, f"/v2/org/gated/blobs/{small}", logged_in)[0] == 200
    assert get(public, f"/v2/org/gated/blobs/{small}", anonymous)[0] == 401
    assert get(public, f"/v2/org/gated/blobs/{small}", {"Authorization": "Bearer hf_mud"})[0] == 403
    # Nor do its pages show private repos, or name them
    assert get(public, "/org/secret")[0] == 404
    home = get(public, "/")[2]
    assert b"org/gated" in home and b"org/clay" in home and b"org/secret" not in home


@pytest.mark.skipif(not shutil.which("oras"), reason="needs ORAS")
def test_oras_logs_in_to_a_public_registry(public, tmp_path):
    reference = f"127.0.0.1:{public.server_port}/org/gated:latest"
    command = ["oras", "cp", "--from-plain-http", reference, "--to-oci-layout", f"{tmp_path / 'out'}:latest"]
    assert subprocess.run(command, capture_output=True).returncode != 0
    subprocess.run([*command, "--from-username", "clay", "--from-password", TOKEN], check=True, capture_output=True)
    hub.unpack(tmp_path / "out", tmp_path / "root")
    assert (tmp_path / "root/model/weights").read_bytes() == WEIGHTS
