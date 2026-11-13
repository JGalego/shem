"""Images in Hugging Face repos: fetched, pushed and unpacked as OCI layouts,
and served as a registry. A fake Hub keeps each repo's revisions in a
directory, so these need no network; with ORAS installed, a real client
pulls through the registry, following a large blob to where it is kept."""

import functools
import hashlib
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
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

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


class FakeApi:
    """A Hub whose repos are directories: ROOT/ORG/NAME/REVISION holds the
    files of each revision, `main` the default branch's."""

    endpoint = "https://hub.example"
    token = None

    def __init__(self, root: Path):
        self.root = root

    def dir(self, repo: str, rev: str | None) -> Path:
        self.repo(repo)
        path = self.root / repo / (rev or "main")
        if not path.is_dir():
            raise RevisionNotFoundError(f"{repo} has no revision {rev}", response=NOT_FOUND)
        return path

    def repo(self, repo: str) -> Path:
        if not (self.root / repo).is_dir():
            raise RepositoryNotFoundError(f"no repo {repo}", response=NOT_FOUND)
        return self.root / repo

    def list_repo_files(self, repo, revision=None):
        base = self.dir(repo, revision)
        return [p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()]

    def hf_hub_download(self, repo, path, revision=None):
        file = self.dir(repo, revision) / path
        if not file.is_file():
            raise EntryNotFoundError(f"{repo} has no {path}")
        return str(file)

    def snapshot_download(self, repo, revision=None, local_dir=None, allow_patterns=None):
        shutil.copytree(self.dir(repo, revision), local_dir, dirs_exist_ok=True)
        (Path(local_dir) / ".cache" / "huggingface").mkdir(parents=True)

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
def served(api, pushed, monkeypatch):
    """The registry over the fake Hub, with large blobs where a CDN keeps
    them: a server of their own, as Hugging Face's are elsewhere."""
    blobs = api.root / "org/clay/main/blobs/sha256"
    cdn = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=blobs)
    )

    def metadata(url, token=None):
        path = Path(url.split("/resolve/")[1].split("/", 1)[1])
        file = blobs / path.name
        if "/resolve/main/" not in url or not file.is_file():
            raise EntryNotFoundError(url)
        return SimpleNamespace(size=file.stat().st_size, location=f"http://127.0.0.1:{cdn.server_port}/{path.name}")

    monkeypatch.setattr(registry, "get_hf_file_metadata", metadata)
    # Between a tar of a small file, a 10 KiB record, and the weights
    monkeypatch.setattr(registry, "SMALL", 12 * 1024)
    server = registry.serve(("127.0.0.1", 0), registry.Hub(api))
    for s in (cdn, server):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield server
    for s in (cdn, server):
        s.shutdown()


def get(server, path: str):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    connection.request("GET", path)
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


def test_repo_pages_show_the_image_and_its_model(served, model):
    status, _, body = get(served, "/org/golem")
    page = body.decode()
    assert status == 200
    # How to run and pull it, from this registry
    assert 'shem run hf.co/org/golem "What is a golem?"' in page
    assert f"oras pull 127.0.0.1:{served.server_port}/org/golem:latest" in page
    # The image, its model and its comparison
    assert "You are a &lt;golem&gt;." in page
    assert "Yes: runs start after the system prompt" in page
    assert "Org/Clay-1B" in page and "0123456789ab" in page and "1.23B" in page
    assert "2 attention, 2 mlp" in page
    assert "8.10e-05" in page and "Verified" in page
    assert '<a href="/org/golem?tag=modelpack">modelpack</a>' in page
    assert "org.example" in page and "dev.shem.model" not in page.split("<h2>Labels</h2>")[1]


def test_repo_pages_take_a_tag_and_explain_failures(served, model):
    assert b"<b>modelpack</b>" in get(served, "/org/golem?tag=modelpack")[2]
    status, headers, _ = get(served, "/go?repo=hf.co/org/golem/")
    assert (status, headers["Location"]) == (302, "/org/golem")
    status, headers, body = get(served, "/org/nothing-here")
    assert status == 404 and headers["Content-Type"].startswith("text/html")
    assert b"<h1>404 Not Found</h1>" in body
    assert get(served, "/org/golem?tag=no/such")[0] == 404
