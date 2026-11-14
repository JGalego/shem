"""Serve images kept in Hugging Face repos as a read-only OCI registry.

    python -m shem_neural.registry [--host HOST] [--port PORT] [--public]

A repo that holds an OCI image layout, as `shem push IMAGE hf.co/ORG/NAME`
writes one, is the registry's repository ORG/NAME, and its revisions are the
repository's tags: `latest` is `main`, and a branch, tag or commit is the
image it holds. So `oras pull`, containerd or Docker can pull
`localhost:5000/ORG/NAME:TAG` as `shem run hf.co/ORG/NAME:TAG` runs it. A
repo that holds no layout, but a checkpoint, is a ModelPack artifact of its
files (see `checkpoints`), and at `files/ORG/NAME` an image of them, for
Docker and Kubernetes, which take no artifacts, or at `safetensors/ORG/NAME`
of those in one format of weights (see `files`).

Manifests come from the repo's `index.json` and blobs, and small blobs are
served as they are. A large blob, a layer of weights, redirects to where
Hugging Face keeps it, so its bytes never pass through the registry. The
layers of weights of an image of a checkpoint's files are the exception:
tars, which the registry sends as it reads their files from the Hub, keeping
nothing. `--files-limit` says how large a checkpoint may be to be served so,
which a registry anyone may reach should say. Requests
to Hugging Face use this host's token (`HF_TOKEN`, or `hf auth login`), so
private and gated repos it can read are served too, to whoever can reach the
registry: it listens on this host only unless told otherwise.

A public registry, one anyone may reach, uses each client's token instead,
and never this host's. It answers the distribution spec's token flow:
`docker login` with a Hugging Face token as the password, or none, to pull
public repos anonymously. Its home page lists only repos that aren't private.

Outside `/v2/`, the registry serves pages for people: what it is and how
to pull from it at `/`, and each repo's image at `/ORG/NAME`. `/live` tells
an open page whether what it shows has changed, so it can show it again.
"""

import argparse
import base64
import copy
import hashlib
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from shem_neural import checkpoints, files
from shem_neural import registry_pages as pages
from shem_neural.checkpoints import Artifact, File
from shem_neural.hub import revision

#: Blobs up to this size are served, and larger ones redirected
SMALL = 1 << 20
#: Repository names: ORG/NAME, as Hugging Face names repos
NAME = r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)"
DIGEST = r"sha256:(?P<hex>[0-9a-f]{64})"
#: The token a public registry gives clients without a Hugging Face token
ANONYMOUS = "anonymous"
#: How long, in seconds, the Hub's answer to who may read a repo is kept
ACCESS = 300
TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")
#: A repo's page, for people
PAGE = re.compile(rf"^/{NAME}/?$")
#: What a repository's name starts with to mean an image of a checkpoint's
#: files rather than the artifact of them: `files`, or a format of weights
VIEW = rf"(?:(?P<view>{'|'.join(files.VIEWS)})/)?"
ROUTES = [
    (re.compile(r"^/v2/?$"), "base"),
    (re.compile(rf"^/v2/{VIEW}{NAME}/manifests/(?:{DIGEST}|(?P<tag>{TAG.pattern}))$"), "manifest"),
    (re.compile(rf"^/v2/{VIEW}{NAME}/blobs/{DIGEST}$"), "blob"),
    (re.compile(rf"^/v2/{VIEW}{NAME}/tags/list$"), "tags"),
]
#: How much of a file is read from the Hub, and sent, at a time
PIECE = 1 << 20
#: How many files of weights are read at once to learn their tars' digests
READERS = 4
#: How many times in a row reading a file from the Hub may get nowhere,
#: and how long, in seconds, it waits for any of it
RETRIES, TIMEOUT = 3, 60


class Unknown(Exception):
    """An OCI error: a code from the distribution spec, and its status."""

    def __init__(self, code: str, message: str, status: HTTPStatus = HTTPStatus.NOT_FOUND):
        super().__init__(message)
        self.code = code
        self.status = status


class Stream:
    """A body too large to hold: its size, and `read`, which gives its bytes
    from one offset to another, in pieces, as they come."""

    def __init__(self, size: int, read):
        self.size, self.read = size, read

    def __len__(self) -> int:
        return self.size


class Hub:
    """Layouts in Hugging Face repos, read with this host's token, or the one
    `using` gives it."""

    def __init__(self, api: HfApi | None = None):
        self.api = api or HfApi()
        #: When the Hub's answers, by token and question, stop being trusted:
        #: whether a token is one, a repo readable with it, or a repo private
        self.answers: dict[tuple[str, str], tuple[float, bool]] = {}
        #: The revisions served by tag, by repo: where a digest is looked for
        #: when the default branch hasn't it
        self.seen: dict[str, set[str]] = {}
        #: The artifacts made of repos that hold a checkpoint, by repo and commit
        self.artifacts: dict[tuple[str, str], Artifact] = {}
        #: The images made of checkpoints' files, by repo, commit and view
        self.images: dict[tuple[str, str, str], files.Image] = {}
        #: The digests of layers of weights, by the files they hold, which
        #: outlast a commit that changes other files
        self.tars: dict[tuple, str] = {}
        #: Held to make an image, one at a time: it reads its weights through
        self.making = threading.Lock()

    def using(self, token: str | None) -> "Hub":
        """This Hub, sharing what it has learnt, reading as `token`: a
        client's token, or no token at all, never this host's."""
        view = copy.copy(self)
        view.api = copy.copy(self.api)
        view.api.token = token or False
        return view

    def ask(self, question: str, call, *exceptions) -> bool:
        """Whether `call` succeeds rather than raising one of `exceptions`,
        as the Hub answered it for this token in the last few minutes."""
        token = self.api.token or ""
        key = (hashlib.sha256(token.encode()).hexdigest() if token else "", question)
        until, answer = self.answers.get(key, (0, False))
        if until < time.time():
            try:
                call()
                answer = True
            except exceptions:
                answer = False
            self.answers[key] = (time.time() + ACCESS, answer)
        return answer

    def admit(self, repo: str) -> None:
        """Raise unless this token may read `repo`'s files: what's kept of
        repos is served only to those the Hub would serve it to."""
        # A gated repo is also one not found
        if not self.ask(f"read {repo.lower()}", lambda: self.api.auth_check(repo), RepositoryNotFoundError):
            # Ask again, for the error that says why
            self.api.auth_check(repo)

    def valid(self) -> bool:
        """Whether this Hub's token is one Hugging Face knows."""
        return self.ask("whoami", self.api.whoami, HfHubHTTPError)

    def listed(self, repo: str) -> bool:
        """Whether anyone may see that `repo` exists: it isn't private."""
        anyone = self.using(None)
        return anyone.ask(f"see {repo.lower()}", lambda: anyone.api.model_info(repo), RepositoryNotFoundError)

    def read(self, repo: str, path: str, rev: str | None) -> bytes:
        return Path(self.api.hf_hub_download(repo, path, revision=rev)).read_bytes()

    def revisions(self, repo: str) -> list[str | None]:
        return [None, *sorted(self.seen.get(repo, ()))]

    def index(self, repo: str, tag: str) -> dict | None:
        """The layout's index at `tag`, or None if the repo holds no layout there."""
        rev = revision(tag)
        try:
            data = json.loads(self.read(repo, "index.json", rev))
        except EntryNotFoundError:
            return None
        if rev:
            self.seen.setdefault(repo, set()).add(rev)
        return data

    def commit(self, repo: str, tag: str) -> str:
        """The commit `tag` names, which changes with whatever the repo holds."""
        return self.api.model_info(repo, revision=revision(tag)).sha

    def checkpoint(self, repo: str, tag: str) -> Artifact:
        """The artifact of the checkpoint `repo` holds at `tag`, made once a commit."""
        key = (repo.lower(), self.commit(repo, tag))
        if key not in self.artifacts:
            try:
                self.artifacts[key] = checkpoints.build(self.api, repo, key[1])
            except ValueError as error:
                raise Unknown("MANIFEST_UNKNOWN", str(error)) from None
        return self.artifacts[key]

    def entry(self, repo: str, tag: str) -> dict:
        """The descriptor of what `repo` holds at `tag`: the first manifest its
        layout lists, as `shem run` takes it, or else its checkpoint's artifact."""
        index = self.index(repo, tag)
        if index is None:
            return self.checkpoint(repo, tag).descriptor
        entries = index.get("manifests") or []
        if not entries:
            raise Unknown("MANIFEST_UNKNOWN", f"{repo}:{tag} lists no manifest")
        return entries[0]

    def stream(self, repo: str, file: File, rev: str, start: int, end: int):
        """The bytes of `file` of `repo` at `rev`, from `start` to `end`, in
        pieces: a small file's from the copy kept of it, and a large one's
        from where the Hub keeps it, going on from where it breaks off."""
        if not file.large:
            with open(self.api.hf_hub_download(repo, file.path, revision=rev), "rb") as kept:
                kept.seek(start)
                while start < end and (piece := kept.read(min(PIECE, end - start))):
                    start += len(piece)
                    yield piece
            return
        url = hf_hub_url(repo, file.path, revision=rev, endpoint=self.api.endpoint)
        stuck = 0
        while start < end:
            before = start
            try:
                # Asked anew each time, as the Hub's addresses for a file expire
                location = get_hf_file_metadata(url, token=self.api.token).location
                request = urllib.request.Request(location, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                    # A server that knows no ranges sends the file from its start
                    skip = start if response.status == HTTPStatus.OK else 0
                    while start < end and (piece := response.read(PIECE)):
                        piece, skip = piece[skip:][: end - start], max(skip - len(piece), 0)
                        start += len(piece)
                        if piece:
                            yield piece
            except (OSError, HTTPException) as error:
                if stuck == RETRIES:
                    raise OSError(f"reading {repo}'s {file.path} from the Hub: {error}") from error
            stuck = 0 if start > before else stuck + 1
            if stuck > RETRIES:
                raise OSError(f"reading {repo}'s {file.path} from the Hub: it ends at byte {start} of {end}")
            time.sleep(stuck)

    def image(self, repo: str, tag: str, view: str = files.ALL, limit: int | None = None) -> files.Image:
        """The image of the files of the checkpoint `repo` holds at `tag`,
        or of those `view` keeps, made once a commit. Making it reads each
        file of weights through, unless another image had it, which `limit`,
        a size in bytes, refuses to do for more weights than that."""
        if self.index(repo, tag) is not None:
            raise Unknown("MANIFEST_UNKNOWN", f"{repo} holds an image, not a checkpoint: pull it without {view}/")
        artifact = self.checkpoint(repo, tag)
        key = (artifact.repo.lower(), artifact.commit, view)
        if key not in self.images:
            try:
                small, weights = files.plan(artifact, view, SMALL)
            except ValueError as error:
                raise Unknown("MANIFEST_UNKNOWN", str(error)) from None
            unread = sum(file.size for layer in weights if layer.key not in self.tars for _, file in layer.files)
            if limit is not None and unread > limit:
                raise Unknown(
                    "MANIFEST_UNKNOWN",
                    f"{repo} has {pages.size(unread)} of weights, and this registry makes images of "
                    f"checkpoints' files up to {pages.size(limit)}: pull it without {view}/, as an artifact",
                )

            def read(file: File, start: int, end: int):
                return self.stream(artifact.repo, file, artifact.commit, start, end)

            def digest(layer: files.Layer) -> str:
                if layer.key not in self.tars:
                    self.tars[layer.key] = layer.digest(read)
                return self.tars[layer.key]

            with self.making:
                if key not in self.images:
                    with ThreadPoolExecutor(READERS) as readers:
                        digests = list(readers.map(digest, weights))
                    tar = (small, files.tar(small, read)) if small else None
                    self.images[key] = files.build(artifact, tar, list(zip(weights, digests, strict=True)))
        return self.images[key]

    def layer(self, repo: str, hex: str) -> tuple[files.Image, files.Layer] | None:
        """A layer of weights of an image made of `repo`'s files."""
        for (name, *_), image in list(self.images.items()):
            if name == repo.lower() and hex in image.layers:
                return image, image.layers[hex]
        return None

    def made(self, repo: str, hex: str) -> tuple[Artifact | files.Image, bytes | File] | None:
        """A blob of an artifact or an image made of `repo`: one made for it,
        or one of the repo's files."""
        for (name, _), artifact in list(self.artifacts.items()):
            if name == repo.lower():
                found = artifact.blobs.get(hex) or artifact.files.get(hex)
                if found:
                    return artifact, found
        for (name, *_), image in list(self.images.items()):
            if name == repo.lower() and hex in image.blobs:
                return image, image.blobs[hex]
        return None

    def blob(self, repo: str, hex: str) -> bytes:
        """A blob, from the default branch or a revision already served."""
        if found := self.made(repo, hex):
            artifact, item = found
            if isinstance(item, bytes):
                return item
            return Path(self.api.hf_hub_download(artifact.repo, item.path, revision=artifact.commit)).read_bytes()
        for rev in self.revisions(repo):
            try:
                return self.read(repo, f"blobs/sha256/{hex}", rev)
            except EntryNotFoundError:
                continue
        raise EntryNotFoundError(f"{repo} has no blob sha256:{hex}")

    def locate(self, repo: str, hex: str) -> tuple[int, str | None]:
        """A blob's size, and the URL its bytes are at, if they're best
        fetched there: None for a blob the registry serves itself."""
        if found := self.made(repo, hex):
            artifact, item = found
            if isinstance(item, bytes) or not item.large:
                return (len(item), None) if isinstance(item, bytes) else (item.size, None)
            url = hf_hub_url(artifact.repo, item.path, revision=artifact.commit, endpoint=self.api.endpoint)
            return item.size, get_hf_file_metadata(url, token=self.api.token).location
        for rev in self.revisions(repo):
            url = hf_hub_url(repo, f"blobs/sha256/{hex}", revision=rev, endpoint=self.api.endpoint)
            try:
                found = get_hf_file_metadata(url, token=self.api.token)
            except EntryNotFoundError:
                continue
            return found.size, found.location
        raise EntryNotFoundError(f"{repo} has no blob sha256:{hex}")

    def comparison(self, repo: str, tag: str) -> dict | None:
        """The comparison with transformers the Space recorded beside the
        image, if it did."""
        try:
            return json.loads(self.read(repo, "comparison.json", revision(tag)))
        except (EntryNotFoundError, ValueError):
            return None

    def tags(self, repo: str) -> list[str]:
        refs = self.api.list_repo_refs(repo)
        names = {"latest"} | {r.name for r in refs.branches if r.name != "main"} | {r.name for r in refs.tags}
        return sorted(names)


def media_type(data: bytes, fallback: str) -> str:
    try:
        return json.loads(data).get("mediaType") or fallback
    except (ValueError, AttributeError):
        return fallback


class Registry:
    """The distribution API's read side over `hub`: each call returns a
    status, headers and a body."""

    def __init__(self, hub, public: bool = False, limit: int | None = None):
        self.hub = hub
        #: Whether each client reads as itself, with its own Hugging Face
        #: token, rather than as this host
        self.public = public
        #: The most weights, in bytes, a checkpoint may have to be served as
        #: an image of its files, which takes reading them through
        self.limit = limit
        #: What the home page shows of the repos served, by repo
        self.served: dict[str, dict] = {}
        #: Pulls, and the bytes of blobs redirected to the Hub, sent from
        #: here, and read from the Hub to send as layers of images of files
        self.stats = {"pulls": 0, "redirected": 0, "sent": 0, "streamed": 0}

    def base(self, hub, match) -> tuple[HTTPStatus, dict, bytes]:
        return HTTPStatus.OK, {"Content-Type": "application/json"}, b"{}"

    def token(self, authorization: str) -> tuple[HTTPStatus, dict, bytes]:
        """A public registry's token for a client: the Hugging Face token it
        logged in with, once the Hub knows it, or else the anonymous one."""
        secret = credential(authorization)
        if secret and not self.hub.using(secret).valid():
            raise Unknown("UNAUTHORIZED", "Hugging Face knows no such token", HTTPStatus.UNAUTHORIZED)
        token = secret or ANONYMOUS
        body = json.dumps({"token": token, "access_token": token, "expires_in": 3600}).encode()
        return HTTPStatus.OK, {"Content-Type": "application/json"}, body

    def admit(self, hub, route: str, match, authorization: str) -> None:
        """Let a public registry's client in, or ask it to log in: for a
        token, at first, anonymous or not, and then for a token of its own
        to read a repo the anonymous may not."""
        login = Unknown("UNAUTHORIZED", "log in with a Hugging Face token, or none", HTTPStatus.UNAUTHORIZED)
        anonymous = not credential(authorization)
        if route == "base":
            if not authorization:
                raise login
            return
        try:
            hub.admit(match["name"])
        except RepositoryNotFoundError:
            # Gated repos too: a client that logs in may read them
            if not anonymous:
                raise
            raise login from None

    def manifest(self, hub, match) -> tuple[HTTPStatus, dict, bytes]:
        name = match["name"]
        if match["hex"]:
            data = hub.blob(name, match["hex"])
            kind = media_type(data, "application/vnd.oci.image.manifest.v1+json")
        else:
            if match["view"]:
                entry = hub.image(name, match["tag"], match["view"], self.limit).descriptor
            else:
                entry = hub.entry(name, match["tag"])
            data = hub.blob(name, entry["digest"].removeprefix("sha256:"))
            if not self.public or self.hub.listed(name):
                # An image of files is of an artifact, which is what the home page shows
                self.pulled(name, match["tag"], hub.checkpoint(name, match["tag"]).manifest if match["view"] else data)
            kind = entry.get("mediaType") or media_type(data, "application/vnd.oci.image.manifest.v1+json")
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if match["hex"] and digest != f"sha256:{match['hex']}":
            raise Unknown("MANIFEST_INVALID", f"{name}'s blob {match['hex']} does not match its digest")
        return HTTPStatus.OK, {"Content-Type": kind, "Docker-Content-Digest": digest}, data

    def pulled(self, name: str, tag: str, manifest: bytes) -> None:
        """Note a pull of `name` at `tag`, for the home page."""
        summary = self.served.setdefault(name, {})
        summary.setdefault("tags", set()).add(tag)
        summary["pulls"] = summary.get("pulls", 0) + 1
        summary["last"] = time.time()
        self.stats["pulls"] += 1
        try:
            data = json.loads(manifest)
        except ValueError:
            return
        kind = data.get("artifactType") or (data.get("config") or {}).get("mediaType", "")
        summary["kind"] = "artifact" if ".model." in kind else "image"
        if "layers" in data:
            summary["size"] = sum(layer.get("size", 0) for layer in data["layers"])

    def blob(self, hub, match) -> tuple[HTTPStatus, dict, bytes | Stream]:
        name, hex = match["name"], match["hex"]
        headers = {"Docker-Content-Digest": f"sha256:{hex}"}
        if found := hub.layer(name, hex):
            image, layer = found

            def of(file: File, start: int, end: int):
                return hub.stream(image.repo, file, image.commit, start, end)

            def read(start: int, end: int):
                for piece in layer.read(of, start, end):
                    self.stats["streamed"] += len(piece)
                    yield piece

            return HTTPStatus.OK, headers | {"Content-Type": "application/octet-stream"}, Stream(layer.size, read)
        size, location = hub.locate(name, hex)
        if location and size > SMALL:
            self.stats["redirected"] += size
            return HTTPStatus.TEMPORARY_REDIRECT, headers | {"Location": location}, b""
        data = hub.blob(name, hex)
        self.stats["sent"] += len(data)
        return HTTPStatus.OK, headers | {"Content-Type": "application/octet-stream"}, data

    def tags(self, hub, match) -> tuple[HTTPStatus, dict, bytes]:
        name = f"{match['view']}/{match['name']}" if match["view"] else match["name"]
        body = json.dumps({"name": name, "tags": hub.tags(match["name"])}).encode()
        return HTTPStatus.OK, {"Content-Type": "application/json"}, body

    def page(self, hub, path: str, query: dict[str, list[str]], host: str) -> tuple[HTTPStatus, dict, bytes]:
        """A page for people: the home page, or a repo's."""
        html = {"Content-Type": "text/html; charset=utf-8"}
        if path == "/":
            return HTTPStatus.OK, html, pages.home(host, self.served, self.stats, self.version())
        if path == "/live":
            repo, tag = (query.get("repo") or [""])[0], (query.get("tag") or ["latest"])[0]
            if repo and not (PAGE.match(f"/{repo}") and TAG.fullmatch(tag)):
                raise Unknown("NAME_INVALID", f"{repo}:{tag} is no repo and tag")
            version = hub.commit(repo, tag) if repo else self.version()
            return HTTPStatus.OK, {"Content-Type": "application/json"}, json.dumps({"version": version}).encode()
        if path == "/go":
            repo = (query.get("repo") or [""])[0].strip().strip("/").removeprefix("hf.co/")
            if PAGE.match(f"/{repo}"):
                return HTTPStatus.FOUND, {"Location": f"/{repo}"}, b""
            raise Unknown("NAME_INVALID", f"{repo!r} is no repo: ORG/NAME names one")
        if match := PAGE.match(path):
            tag = (query.get("tag") or ["latest"])[0]
            if not TAG.fullmatch(tag):
                raise Unknown("TAG_INVALID", f"{tag!r} is no tag")
            if self.public:
                hub.admit(match["name"])
            # Listed once it shows, so a repo that fails to isn't
            summary = dict(self.served.get(match["name"], {}))
            body = pages.repo_page(hub, host, match["name"], tag, summary)
            self.served[match["name"]] = summary
            return HTTPStatus.OK, html, body
        raise Unknown("NAME_INVALID", f"no such page {path}")

    def version(self) -> str:
        """What the home page shows, as a digest that changes with it."""
        text = json.dumps([self.served, self.stats], sort_keys=True, default=sorted)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def handle(
        self, target: str, host: str = "localhost:5000", authorization: str = "", scheme: str = "http"
    ) -> tuple[HTTPStatus, dict, bytes | Stream]:
        """Answer a GET of `target`, sent to `host` over `scheme`, with
        `authorization`, the request's header of that name."""
        path, _, query = target.partition("?")
        api = path.startswith("/v2") or (self.public and path == "/token")
        token = credential(authorization) if api else None
        hub = self.hub.using(token) if self.public else self.hub
        try:
            if self.public and path == "/token":
                return self.token(authorization)
            if not api:
                return self.page(hub, path, urllib.parse.parse_qs(query), host)
            for pattern, route in ROUTES:
                if match := pattern.match(path):
                    if self.public:
                        self.admit(hub, route, match, authorization)
                    return getattr(self, route)(hub, match)
            raise Unknown("NAME_INVALID", f"no such endpoint {path}")
        except Exception as error:
            status, code, message = failure(error, path)
            if api:
                headers, body = errors(code, message)
                if status == HTTPStatus.UNAUTHORIZED and path.startswith("/v2"):
                    headers["WWW-Authenticate"] = challenge(scheme, host, path)
                return status, headers, body
            body = pages.page("shem registry", f"<h1>{status.value} {status.phrase}</h1><p>{pages.e(message)}</p>")
            return status, {"Content-Type": "text/html; charset=utf-8"}, body


def failure(error: Exception, path: str) -> tuple[HTTPStatus, str, str]:
    """What went wrong, as the distribution spec's errors say it."""
    match error:
        case Unknown():
            return error.status, error.code, str(error)
        # A gated repo is also one not found
        case GatedRepoError():
            return HTTPStatus.FORBIDDEN, "DENIED", "a gated repo, and no access to it with this token"
        case RepositoryNotFoundError():
            return HTTPStatus.NOT_FOUND, "NAME_UNKNOWN", "no such repo, or no access to it"
        case RevisionNotFoundError():
            return HTTPStatus.NOT_FOUND, "MANIFEST_UNKNOWN", "no such revision"
        case EntryNotFoundError():
            code = "BLOB_UNKNOWN" if "/blobs/" in path else "MANIFEST_UNKNOWN"
            return HTTPStatus.NOT_FOUND, code, f"not an image layout, or no such blob: {error}"
        case HfHubHTTPError():
            return HTTPStatus.BAD_GATEWAY, "UNKNOWN", f"Hugging Face: {error}"
    print(f"shem registry: {path}: {error!r}", file=sys.stderr, flush=True)
    return HTTPStatus.INTERNAL_SERVER_ERROR, "UNKNOWN", f"the registry failed: {error}"


def credential(authorization: str) -> str | None:
    """The Hugging Face token an Authorization header carries: a bearer
    token this registry gave, or the password of a basic one."""
    scheme, _, value = authorization.strip().partition(" ")
    if scheme.lower() == "bearer":
        token = value.strip()
    elif scheme.lower() == "basic":
        try:
            token = base64.b64decode(value.strip()).decode().partition(":")[2]
        except ValueError:
            return None
    else:
        return None
    return None if token in ("", ANONYMOUS) else token


def challenge(scheme: str, host: str, path: str) -> str:
    """Where a client gets a token to pull what `path` names, and for what."""
    found = f'Bearer realm="{scheme}://{host}/token",service="{host}"'
    for pattern, route in ROUTES:
        if route != "base" and (match := pattern.match(path)):
            name = f"{match['view']}/{match['name']}" if match["view"] else match["name"]
            found += f',scope="repository:{name}:pull"'
    return found


def byte_range(header: str, size: int) -> tuple[int, int] | None:
    """The bytes a Range header asks for of a body of `size`, as where they
    start and end, if it asks for one run of them, and there is one."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or not (match[1] or match[2]):
        return None
    if match[1]:
        start, end = int(match[1]), min(int(match[2]) + 1, size) if match[2] else size
    else:
        start, end = max(size - int(match[2]), 0), size
    return (start, end) if start < end else None


def byte_size(text: str) -> int:
    """A size in bytes, as `16G` or `500M` or `1048576` says it."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?)i?B?", text.strip(), re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"{text!r} is no size: 16G is one")
    return int(float(match[1]) * 1024 ** " KMGT".index(match[2].upper() or " "))


def errors(code: str, message: str) -> tuple[dict, bytes]:
    body = json.dumps({"errors": [{"code": code, "message": message}]}).encode()
    return {"Content-Type": "application/json"}, body


def handler(registry: Registry) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "shem-registry"
        protocol_version = "HTTP/1.1"

        def respond(self, body: bool) -> None:
            status, headers, data = registry.handle(
                self.path,
                self.headers.get("Host") or "localhost:5000",
                self.headers.get("Authorization") or "",
                # As a proxy in front of it, such as a Space's, says it was asked
                self.headers.get("X-Forwarded-Proto") or "http",
            )
            start, end = 0, len(data)
            if isinstance(data, Stream) and status == HTTPStatus.OK:
                if span := byte_range(self.headers.get("Range") or "", data.size):
                    start, end = span
                    status = HTTPStatus.PARTIAL_CONTENT
                    headers = headers | {"Content-Range": f"bytes {start}-{end - 1}/{data.size}"}
            self.send_response(status)
            headers = {"Docker-Distribution-API-Version": "registry/2.0", "Content-Length": str(end - start)} | headers
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            if not body:
                return
            if not isinstance(data, Stream):
                self.wfile.write(data)
                return
            try:
                for piece in data.read(start, end):
                    self.wfile.write(piece)
                    start += len(piece)
            except OSError as error:
                print(f"shem registry: {self.path}: {error}", file=sys.stderr, flush=True)
            finally:
                # A body cut short can't be told from the response after it
                self.close_connection = self.close_connection or start < end

        def do_GET(self) -> None:
            self.respond(body=True)

        def do_HEAD(self) -> None:
            self.respond(body=False)

        def refuse(self) -> None:
            headers, data = errors("UNSUPPORTED", "a read-only registry: push with `shem push IMAGE hf.co/ORG/NAME`")
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            for key, value in (headers | {"Content-Length": str(len(data))}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        do_PUT = do_POST = do_PATCH = do_DELETE = refuse

        def log_message(self, format: str, *args) -> None:
            print(f"shem registry: {format % args}", file=sys.stderr, flush=True)

    return Handler


def serve(address: tuple[str, int], hub=None, public: bool = False, limit: int | None = None) -> ThreadingHTTPServer:
    """A registry over `hub` (Hugging Face's, by default) at `address`, not
    yet serving: a public one, if `public`, reading with clients' tokens,
    which makes images of the files of checkpoints with weights up to `limit`."""
    return ThreadingHTTPServer(address, handler(Registry(hub or Hub(), public, limit)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen at (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="the port to listen at (default 5000)")
    parser.add_argument(
        "--public", action="store_true", help="read repos with each client's Hugging Face token, never this host's"
    )
    parser.add_argument(
        "--files-limit",
        type=byte_size,
        metavar="SIZE",
        help="the most weights a checkpoint may have, such as 16G, to be served as an image of its files, "
        "which sends them through this registry (default: any)",
    )
    args = parser.parse_args()
    server = serve((args.host, args.port), public=args.public, limit=args.files_limit)
    whose = "each client's token" if args.public else "this host's token"
    print(
        f"shem registry: serving Hugging Face repos at {args.host}:{args.port}, read-only, with {whose}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
