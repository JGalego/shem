"""OCI image layouts kept in Hugging Face repos.

    python -m shem_neural.hub fetch REPO OUT [--revision REVISION]
    python -m shem_neural.hub push LAYOUT REPO [--revision BRANCH] [--private]

A repo holds an image when it has an OCI image layout at its root, as
`shem push IMAGE hf.co/ORG/NAME` writes one: `oci-layout`, `index.json`, and
the blobs under `blobs/sha256`, each named by its digest. A revision of the
repo, a branch, tag or commit, holds one image, the first that `index.json`
lists, as a tag does in a registry; `latest` is `main`.

`fetch` downloads the layout at a revision, if the repo holds one there, and
writes nothing otherwise, so that `shem run hf.co/ORG/NAME` imports a repo
that holds a checkpoint instead. `push` uploads a layout to a branch,
creating the repo and the branch if need be, and removes the blobs only an
earlier push had.
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

#: What a layout holds: `blobs/sha256/*` covers every blob
LAYOUT = ["oci-layout", "index.json", "blobs/sha256/*"]


def revision(tag: str | None) -> str | None:
    """The revision a tag names: `latest`, like no tag, is the default branch."""
    return None if tag in (None, "latest") else tag


def is_layout(files: list[str]) -> bool:
    return "oci-layout" in files and "index.json" in files


def fetch(repo: str, out: Path, tag: str | None = None, api: HfApi | None = None) -> bool:
    """Download the layout `repo` holds at `tag` into `out`. Returns False,
    writing nothing, if it holds none there, or has no such revision, as a
    repo of GGUF files hasn't for a quantization's name."""
    api = api or HfApi()
    try:
        files = api.list_repo_files(repo, revision=revision(tag))
    except RevisionNotFoundError:
        return False
    if not is_layout(files):
        return False
    api.snapshot_download(repo, revision=revision(tag), local_dir=out, allow_patterns=LAYOUT)
    # Where huggingface_hub keeps track of a local directory's downloads
    shutil.rmtree(out / ".cache", ignore_errors=True)
    return True


def push(layout: Path, repo: str, tag: str | None = None, private: bool = False, api: HfApi | None = None) -> str:
    """Upload `layout` to `repo`, at the branch `tag` names. Returns the
    commit's URL."""
    if not (layout / "oci-layout").is_file():
        raise SystemExit(f"{layout}: not an OCI image layout")
    api = api or HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    branch = revision(tag)
    if branch:
        api.create_branch(repo, branch=branch, exist_ok=True)
    # The blobs first, which a large upload may commit a batch at a time,
    # then the index, in a commit that also removes the blobs only earlier
    # pushes had: whichever commit a reader sees, its index finds its blobs
    api.upload_folder(
        repo_id=repo,
        folder_path=layout,
        revision=branch,
        allow_patterns=["blobs/sha256/*"],
        commit_message="Push an image's blobs with shem",
    )
    commit = api.upload_folder(
        repo_id=repo,
        folder_path=layout,
        revision=branch,
        allow_patterns=["oci-layout", "index.json"],
        delete_patterns=[f"blobs/sha256/{blob.name}" for blob in stale(api, repo, branch, layout)],
        commit_message="Push an image with shem",
    )
    return commit.commit_url


def stale(api: HfApi, repo: str, branch: str | None, layout: Path) -> list[Path]:
    """The blobs at `branch` of `repo` that `layout` doesn't have."""
    ours = {blob.name for blob in (layout / "blobs" / "sha256").iterdir()}
    files = api.list_repo_files(repo, revision=branch)
    return [Path(f) for f in files if f.startswith("blobs/sha256/") and Path(f).name not in ours]


def unpack(layout: Path, out: Path) -> None:
    """Unpack the filesystem of the image in `layout` into `out`: each layer
    of its first manifest, in order, compressed with zstd or gzip or not."""
    import json

    import zstandard

    def blob(digest: str) -> Path:
        return layout / "blobs" / "sha256" / digest.removeprefix("sha256:")

    entry = json.loads((layout / "index.json").read_text())["manifests"][0]
    manifest = json.loads(blob(entry["digest"]).read_text())
    # An index lists the image under platforms: they are all the same image
    while "manifests" in manifest:
        manifest = json.loads(blob(manifest["manifests"][0]["digest"]).read_text())
    out.mkdir(parents=True, exist_ok=True)
    for layer in manifest["layers"]:
        with open(blob(layer["digest"]), "rb") as file:
            if layer["mediaType"].endswith("+zstd"):
                stream = zstandard.ZstdDecompressor().stream_reader(file)
                archive = tarfile.open(fileobj=stream, mode="r|")
            else:
                archive = tarfile.open(fileobj=file, mode="r|*")
            with archive:
                archive.extractall(out, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("fetch", help="download the layout a repo holds, if it holds one")
    get.add_argument("repo", help="a Hugging Face repo, such as ORG/NAME")
    get.add_argument("out", type=Path, help="the directory to download the layout to")
    get.add_argument("--revision", help="a branch, tag or commit; latest is main")
    put = commands.add_parser("push", help="upload a layout to a repo")
    put.add_argument("layout", type=Path, help="the OCI image layout's directory")
    put.add_argument("repo", help="a Hugging Face repo, such as ORG/NAME")
    put.add_argument("--revision", help="the branch to push to; latest is main")
    put.add_argument("--private", action="store_true", help="create the repo private")
    args = parser.parse_args()
    if args.command == "fetch":
        found = fetch(args.repo, args.out, args.revision)
        print(f"{args.repo}: {'an image' if found else 'no image layout'}", file=sys.stderr)
    else:
        print(push(args.layout, args.repo, args.revision, args.private), file=sys.stderr)


if __name__ == "__main__":
    main()
