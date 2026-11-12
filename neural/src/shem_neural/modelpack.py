"""Fetch a ModelPack artifact from a registry and lay it out as a model.

    shem-neural-modelpack REGISTRY/REPO:TAG OUT [--plain-http]

The artifact's files land in OUT as its layers hold them, and the model's
manifest is imported in place beside them, as `shem pull` makes an
artifact's image twin. The BuildKit frontend runs this for `FROM` an
artifact, which BuildKit can't build on itself.

A registry that asks for credentials gets them from a Docker config file:
`REGISTRY_AUTH_FILE`, or `config.json` in `DOCKER_CONFIG` or `~/.docker`.
Credential helpers are not read.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

NAMESPACES = ("cncf", "cnai")
ACCEPT = "application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json"


def is_modelpack(media_type: str) -> bool:
    return any(
        media_type in (f"application/vnd.{ns}.model.manifest.v1+json", f"application/vnd.{ns}.model.config.v1+json")
        for ns in NAMESPACES
    )


def parse_reference(reference: str) -> tuple[str, str, str]:
    """A reference's registry, repository, and tag or digest, with Docker
    Hub's defaults for a reference that names no registry."""
    name, at, digest = reference.partition("@")
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        host, path = first, rest
    else:
        host, path = "registry-1.docker.io", name if "/" in name else f"library/{name}"
    if at:
        return host, path, digest
    path, colon, tag = path.rpartition(":") if ":" in path.rsplit("/", 1)[-1] else (path, "", "")
    return host, path, tag if colon else "latest"


def credentials(host: str) -> str | None:
    """`user:password` for `host`, from a Docker config file, if it has them."""
    path = os.environ.get("REGISTRY_AUTH_FILE")
    config = Path(path) if path else Path(os.environ.get("DOCKER_CONFIG", Path.home() / ".docker")) / "config.json"
    if not config.is_file():
        return None
    auths = json.loads(config.read_text()).get("auths", {})
    keys = (
        ["https://index.docker.io/v1/"]
        if host == "registry-1.docker.io"
        else [host, f"https://{host}", f"http://{host}"]
    )
    for key in keys:
        entry = auths.get(key, {})
        if entry.get("auth"):
            return base64.b64decode(entry["auth"]).decode()
        if entry.get("username"):
            return f"{entry['username']}:{entry.get('password', '')}"
    return None


class SameHostRedirects(urllib.request.HTTPRedirectHandler):
    """Redirects that drop the registry's credentials on the way to another
    host, such as the storage a registry sends blobs from."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


class Registry:
    """A repository in a registry, through the OCI distribution API."""

    def __init__(self, host: str, repository: str, plain_http: bool = False):
        self.base = f"{'http' if plain_http else 'https'}://{host}/v2/{repository}"
        self.host, self.repository = host, repository
        self.authorization: str | None = None
        self.opener = urllib.request.build_opener(SameHostRedirects)

    def open(self, path: str, accept: str | None = None):
        def request():
            headers = {"Accept": accept} if accept else {}
            if self.authorization:
                headers["Authorization"] = self.authorization
            return self.opener.open(urllib.request.Request(self.base + path, headers=headers))

        try:
            return request()
        except urllib.error.HTTPError as error:
            if error.code != 401 or self.authorization:
                raise
            self.authorize(error.headers.get("WWW-Authenticate", ""))
            return request()

    def authorize(self, challenge: str) -> None:
        """Answer a registry's challenge: a bearer token from its token
        service, or basic credentials."""
        login = credentials(self.host)
        if challenge.lower().startswith("basic"):
            if not login:
                raise SystemExit(f"{self.host} needs credentials: log in with docker login")
            self.authorization = "Basic " + base64.b64encode(login.encode()).decode()
            return
        fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        query = {"service": fields.get("service", ""), "scope": f"repository:{self.repository}:pull"}
        url = fields["realm"] + "?" + urllib.parse.urlencode(query)
        headers = {"Authorization": "Basic " + base64.b64encode(login.encode()).decode()} if login else {}
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            answer = json.load(response)
        self.authorization = "Bearer " + (answer.get("token") or answer["access_token"])

    def manifest(self, reference: str) -> dict:
        with self.open(f"/manifests/{reference}", ACCEPT) as response:
            return json.load(response)

    def blob(self, digest: str, into: Path) -> None:
        """Download the blob `digest` into the file `into`, checking it."""
        hashed = hashlib.sha256()
        with self.open(f"/blobs/{digest}") as response, open(into, "wb") as out:
            while chunk := response.read(1 << 20):
                hashed.update(chunk)
                out.write(chunk)
        if "sha256:" + hashed.hexdigest() != digest:
            raise SystemExit(f"blob {digest} does not match its digest")


def annotation(layer: dict, name: str) -> str | None:
    annotations = layer.get("annotations", {})
    return next(
        (annotations[f"org.{ns}.model.{name}"] for ns in NAMESPACES if f"org.{ns}.model.{name}" in annotations), None
    )


def unpack(layer: dict, blob: Path, out: Path) -> None:
    """Lay a layer's files out in `out`: a tar's entries, or a raw layer's
    file at its file path."""
    media_type = layer["mediaType"]
    form = media_type.rsplit(".v1.", 1)[-1]
    if form == "raw":
        path = (annotation(layer, "filepath") or "").lstrip("/")
        if not path or ".." in Path(path).parts:
            raise SystemExit(f"a raw layer's file path is unsafe or missing: {path!r}")
        meta = json.loads(annotation(layer, "file.metadata+json") or "{}")
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(blob, target)
        target.chmod(meta.get("mode", 0o644) & 0o777)
        return
    if form == "tar+zstd":
        import zstandard

        with open(blob, "rb") as compressed:
            data = io.BytesIO(zstandard.ZstdDecompressor().stream_reader(compressed).read())
        archive = tarfile.open(fileobj=data)
    elif form in ("tar", "tar+gzip"):
        archive = tarfile.open(blob, "r:gz" if form == "tar+gzip" else "r:")
    else:
        raise SystemExit(f"a ModelPack layer in {form} format, which shem doesn't read")
    with archive:
        archive.extractall(out, filter="data")


def fetch(reference: str, out: Path, plain_http: bool = False) -> Path:
    """Fetch the ModelPack artifact `reference` into `out`, import it in
    place, and return the model's directory: `out`, or `out/model` for an
    artifact whose files are in one."""
    from .model.importers import import_checkpoint

    host, repository, tag = parse_reference(reference)
    registry = Registry(host, repository, plain_http)
    manifest = registry.manifest(tag)
    if manifest.get("manifests"):
        manifest = registry.manifest(manifest["manifests"][0]["digest"])
    kind = manifest.get("artifactType") or manifest.get("config", {}).get("mediaType", "")
    if not is_modelpack(kind):
        raise SystemExit(f"{reference} is no ModelPack artifact: its type is {kind}")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as blobs:
        for layer in manifest["layers"]:
            blob = Path(blobs) / layer["digest"].split(":")[-1]
            registry.blob(layer["digest"], blob)
            unpack(layer, blob, out)
    model = next((d for d in (out, out / "model") if (d / "config.json").is_file()), None)
    if model is None:
        raise SystemExit(f"{reference} has no config.json at its root or in model/: shem runs Hugging Face checkpoints")
    if not (model / "shem.json").is_file():
        import_checkpoint(model, model, {"artifact": reference}, in_place=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", help="the artifact, such as registry.example.com/team/model:tag")
    parser.add_argument("out", type=Path, help="the directory to lay its files out in")
    parser.add_argument("--plain-http", action="store_true", help="talk to the registry without TLS")
    args = parser.parse_args()
    model = fetch(args.reference, args.out, args.plain_http)
    print(f"{args.reference}: a model in {model}", file=sys.stderr)


if __name__ == "__main__":
    main()
