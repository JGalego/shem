"""Serve images kept in Hugging Face repos as a read-only OCI registry.

    python -m shem_neural.registry [--host HOST] [--port PORT]

A repo that holds an OCI image layout, as `shem push IMAGE hf.co/ORG/NAME`
writes one, is the registry's repository ORG/NAME, and its revisions are the
repository's tags: `latest` is `main`, and a branch, tag or commit is the
image it holds. So `oras pull`, containerd or Docker can pull
`localhost:5000/ORG/NAME:TAG` as `shem run hf.co/ORG/NAME:TAG` runs it.

Manifests come from the repo's `index.json` and blobs, and small blobs are
served as they are. A large blob, a layer of weights, redirects to where
Hugging Face keeps it, so its bytes never pass through the registry. Requests
to Hugging Face use this host's token (`HF_TOKEN`, or `hf auth login`), so
private and gated repos it can read are served too, to whoever can reach the
registry: it listens on this host only unless told otherwise.

Outside `/v2/`, the registry serves pages for people: what it is and how
to pull from it at `/`, and each repo's image at `/ORG/NAME`.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from http import HTTPStatus
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

from shem_neural import registry_pages as pages
from shem_neural.hub import revision

#: Blobs up to this size are served, and larger ones redirected
SMALL = 1 << 20
#: Repository names: ORG/NAME, as Hugging Face names repos
NAME = r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)"
DIGEST = r"sha256:(?P<hex>[0-9a-f]{64})"
TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")
#: A repo's page, for people
PAGE = re.compile(rf"^/{NAME}/?$")
ROUTES = [
    (re.compile(r"^/v2/?$"), "base"),
    (re.compile(rf"^/v2/{NAME}/manifests/(?:{DIGEST}|(?P<tag>{TAG.pattern}))$"), "manifest"),
    (re.compile(rf"^/v2/{NAME}/blobs/{DIGEST}$"), "blob"),
    (re.compile(rf"^/v2/{NAME}/tags/list$"), "tags"),
]


class Unknown(Exception):
    """An OCI error: a code from the distribution spec, and its status."""

    def __init__(self, code: str, message: str, status: HTTPStatus = HTTPStatus.NOT_FOUND):
        super().__init__(message)
        self.code = code
        self.status = status


class Hub:
    """Layouts in Hugging Face repos, read with this host's token."""

    def __init__(self, api: HfApi | None = None):
        self.api = api or HfApi()
        #: The revisions served by tag, by repo: where a digest is looked for
        #: when the default branch hasn't it
        self.seen: dict[str, set[str]] = {}

    def read(self, repo: str, path: str, rev: str | None) -> bytes:
        return Path(self.api.hf_hub_download(repo, path, revision=rev)).read_bytes()

    def revisions(self, repo: str) -> list[str | None]:
        return [None, *sorted(self.seen.get(repo, ()))]

    def index(self, repo: str, tag: str) -> dict:
        rev = revision(tag)
        data = json.loads(self.read(repo, "index.json", rev))
        if rev:
            self.seen.setdefault(repo, set()).add(rev)
        return data

    def blob(self, repo: str, hex: str) -> bytes:
        """A blob, from the default branch or a revision already served."""
        for rev in self.revisions(repo):
            try:
                return self.read(repo, f"blobs/sha256/{hex}", rev)
            except EntryNotFoundError:
                continue
        raise EntryNotFoundError(f"{repo} has no blob sha256:{hex}")

    def locate(self, repo: str, hex: str) -> tuple[int, str]:
        """A blob's size, and the URL its bytes are at."""
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

    def __init__(self, hub):
        self.hub = hub
        #: The repos served by tag, for the home page
        self.served: set[str] = set()

    def base(self, match) -> tuple[HTTPStatus, dict, bytes]:
        return HTTPStatus.OK, {"Content-Type": "application/json"}, b"{}"

    def manifest(self, match) -> tuple[HTTPStatus, dict, bytes]:
        name = match["name"]
        if match["hex"]:
            data = self.hub.blob(name, match["hex"])
            kind = media_type(data, "application/vnd.oci.image.manifest.v1+json")
        else:
            entries = self.hub.index(name, match["tag"]).get("manifests") or []
            if not entries:
                raise Unknown("MANIFEST_UNKNOWN", f"{name}:{match['tag']} lists no manifest")
            # The first, as `shem run` takes it
            entry = entries[0]
            data = self.hub.blob(name, entry["digest"].removeprefix("sha256:"))
            self.served.add(name)
            kind = entry.get("mediaType") or media_type(data, "application/vnd.oci.image.manifest.v1+json")
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if match["hex"] and digest != f"sha256:{match['hex']}":
            raise Unknown("MANIFEST_INVALID", f"{name}'s blob {match['hex']} does not match its digest")
        return HTTPStatus.OK, {"Content-Type": kind, "Docker-Content-Digest": digest}, data

    def blob(self, match) -> tuple[HTTPStatus, dict, bytes]:
        name, hex = match["name"], match["hex"]
        size, location = self.hub.locate(name, hex)
        headers = {"Docker-Content-Digest": f"sha256:{hex}"}
        if size > SMALL:
            return HTTPStatus.TEMPORARY_REDIRECT, headers | {"Location": location}, b""
        data = self.hub.blob(name, hex)
        return HTTPStatus.OK, headers | {"Content-Type": "application/octet-stream"}, data

    def tags(self, match) -> tuple[HTTPStatus, dict, bytes]:
        body = json.dumps({"name": match["name"], "tags": self.hub.tags(match["name"])}).encode()
        return HTTPStatus.OK, {"Content-Type": "application/json"}, body

    def page(self, path: str, query: dict[str, list[str]], host: str) -> tuple[HTTPStatus, dict, bytes]:
        """A page for people: the home page, or a repo's."""
        html = {"Content-Type": "text/html; charset=utf-8"}
        if path == "/":
            return HTTPStatus.OK, html, pages.home(host, sorted(self.served))
        if path == "/go":
            repo = (query.get("repo") or [""])[0].strip().strip("/").removeprefix("hf.co/")
            if PAGE.match(f"/{repo}"):
                return HTTPStatus.FOUND, {"Location": f"/{repo}"}, b""
            raise Unknown("NAME_INVALID", f"{repo!r} is no repo: ORG/NAME names one")
        if match := PAGE.match(path):
            tag = (query.get("tag") or ["latest"])[0]
            if not TAG.fullmatch(tag):
                raise Unknown("TAG_INVALID", f"{tag!r} is no tag")
            return HTTPStatus.OK, html, pages.repo_page(self.hub, host, match["name"], tag)
        raise Unknown("NAME_INVALID", f"no such page {path}")

    def handle(self, target: str, host: str = "localhost:5000") -> tuple[HTTPStatus, dict, bytes]:
        path, _, query = target.partition("?")
        api = path.startswith("/v2")
        try:
            if not api:
                return self.page(path, urllib.parse.parse_qs(query), host)
            for pattern, route in ROUTES:
                if match := pattern.match(path):
                    return getattr(self, route)(match)
            raise Unknown("NAME_INVALID", f"no such endpoint {path}")
        except Exception as error:
            status, code, message = failure(error, path)
            if api:
                return status, *errors(code, message)
            body = pages.page("shem registry", f"<h1>{status.value} {status.phrase}</h1><p>{pages.e(message)}</p>")
            return status, {"Content-Type": "text/html; charset=utf-8"}, body


def failure(error: Exception, path: str) -> tuple[HTTPStatus, str, str]:
    """What went wrong, as the distribution spec's errors say it."""
    match error:
        case Unknown():
            return error.status, error.code, str(error)
        # A gated repo is also one not found
        case GatedRepoError():
            return HTTPStatus.FORBIDDEN, "DENIED", "a gated repo: this host's token has no access to it"
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


def errors(code: str, message: str) -> tuple[dict, bytes]:
    body = json.dumps({"errors": [{"code": code, "message": message}]}).encode()
    return {"Content-Type": "application/json"}, body


def handler(registry: Registry) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "shem-registry"
        protocol_version = "HTTP/1.1"

        def respond(self, body: bool) -> None:
            status, headers, data = registry.handle(self.path, self.headers.get("Host") or "localhost:5000")
            self.send_response(status)
            headers = {"Docker-Distribution-API-Version": "registry/2.0", "Content-Length": str(len(data))} | headers
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            if body:
                self.wfile.write(data)

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


def serve(address: tuple[str, int], hub=None) -> ThreadingHTTPServer:
    """A registry over `hub` (Hugging Face's, by default) at `address`, not yet serving."""
    return ThreadingHTTPServer(address, handler(Registry(hub or Hub())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen at (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="the port to listen at (default 5000)")
    args = parser.parse_args()
    server = serve((args.host, args.port))
    print(
        f"shem registry: serving Hugging Face repos at {args.host}:{args.port}, read-only", file=sys.stderr, flush=True
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
