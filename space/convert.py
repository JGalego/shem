"""Turn a Hugging Face checkpoint into a shem image in a Hugging Face repo.

    python convert.py SOURCE [--system TEXT] [--no-verify] [--push ORG/NAME] [--modelpack] [--private]

The steps the Space runs, which work on their own too:

1. Check the checkpoint before downloading it: an architecture shem imports,
   safetensors weights, and a size this machine can take.
2. Build the image from a Shemfile of `FROM hf.co/SOURCE:COMMIT`, with the
   system prompt primed by `WARM --until=prompt` if there is one.
3. Compare the imported model with transformers, if it is small enough to
   hold in memory twice over, one model at a time.
4. Push the image to a repo, `shem push IMAGE hf.co/ORG/NAME`, and a ModelPack
   artifact of the same blobs to its `modelpack` branch, with a model card.

Each step's output is yielded as it comes, a line at a time.
"""

import argparse
import codecs
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

from shem_neural.hub import unpack
from shem_neural.model.importers import FAMILIES

#: What fits in a free CPU Space's 16 GB. A build holds the model's layers
#: in memory, about 1.4 times its weights' size; priming a system prompt runs
#: a float32 copy beside them; and a comparison holds float32 copies of
#: transformers' model and shem's, one at a time. Raise them on larger hardware.
MAX_BYTES = int(float(os.environ.get("SHEM_MAX_BYTES", 8e9)))
MAX_PRIMED = int(float(os.environ.get("SHEM_MAX_PRIMED", 2e9)))
MAX_VERIFIED = int(float(os.environ.get("SHEM_MAX_VERIFIED", 1.5e9)))
#: The bytes a parameter takes in each of safetensors' dtypes
DTYPE_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2}
SHEM = os.environ.get("SHEM_BIN", "shem")


class Refused(Exception):
    """A checkpoint the Space can't convert, and why."""


@dataclass
class Checkpoint:
    #: The repo, as the Hub names it, and the commit converted
    repo: str
    commit: str
    architecture: str
    parameters: int
    #: The weights' size, as they are stored
    size: int
    license: str | None = None
    gated: bool = False

    @property
    def verifiable(self) -> bool:
        return self.parameters <= MAX_VERIFIED


@dataclass
class Result:
    checkpoint: Checkpoint
    layout: Path
    digest: str = ""
    comparison: dict | None = None
    #: Why there is no comparison, if there isn't
    unverified: str | None = None
    url: str | None = None


def check(source: str, api: HfApi) -> Checkpoint:
    """What the Space needs to know of `source` before downloading it."""
    try:
        info = api.model_info(source, files_metadata=False)
        config = json.loads(Path(api.hf_hub_download(info.id, "config.json", revision=info.sha)).read_text())
    except GatedRepoError:
        raise Refused(f"{source} is gated: accept its licence on its page first, then log in again") from None
    except RepositoryNotFoundError:
        raise Refused(f"{source}: no such model, or it is private to someone else") from None
    except EntryNotFoundError:
        raise Refused(f"{source} has no config.json: the Space converts checkpoints, not GGUF repos") from None
    architectures = config.get("architectures") or []
    known = [a for a in architectures if a in FAMILIES]
    if not known:
        families = ", ".join(sorted(FAMILIES))
        raise Refused(f"{source} is {', '.join(architectures) or 'of no architecture'}; shem imports {families}")
    try:
        counts = api.get_safetensors_metadata(info.id, revision=info.sha).parameter_count
    except Exception:
        raise Refused(f"{source} has no safetensors weights, which the Space converts") from None
    parameters = sum(counts.values())
    size = sum(count * DTYPE_BYTES.get(dtype, 1) for dtype, count in counts.items())
    if size > MAX_BYTES:
        raise Refused(f"{source}'s weights take {size / 1e9:.1f} GB; this Space takes up to {MAX_BYTES / 1e9:g} GB")
    card = info.card_data.to_dict() if info.card_data else {}
    return Checkpoint(info.id, info.sha, known[0], parameters, size, card.get("license"), bool(info.gated))


def refuse(checkpoint: Checkpoint, system: str | None) -> None:
    """Refuse what this machine can't build of `checkpoint`."""
    if system and system.strip() and checkpoint.parameters > MAX_PRIMED:
        limit = f"{MAX_PRIMED / 1e9:g}B"
        raise Refused(f"Priming a system prompt runs the model: this Space does so up to {limit} parameters")


def shemfile(checkpoint: Checkpoint, system: str | None) -> str:
    """The Shemfile of the image: the checkpoint at its commit, and the
    system prompt, primed into a session, if there is one."""
    lines = [f"FROM hf.co/{checkpoint.repo}:{checkpoint.commit}"]
    if system and system.strip():
        lines.append("ENTRYPOINT " + json.dumps(["shem:generate", "--system", system.strip()]))
        lines.append("WARM --until=prompt")
    return "\n".join(lines) + "\n"


def target(user: str, source: str) -> str:
    """The repo an image of `source` goes to: the user's, named for it, in
    lowercase, as OCI names repositories."""
    name = source.split("/")[-1].lower()
    return f"{user}/{re.sub(r'[^a-z0-9._-]', '-', name)}-shem"


#: How often a progress bar's redraws are passed on, in seconds
REDRAW = 0.5


def run(command: list[str], env: dict[str, str]) -> Iterator[str]:
    """Run `command`, yielding its output as it comes, a line at a time. A
    progress bar, which redraws its line after a carriage return, comes as
    that line with a carriage return in front, at most every REDRAW seconds,
    and once more as a plain line when it ends."""
    # Python, which shem runs, buffers what it writes to a pipe unless told not to
    env = os.environ | {"PYTHONUNBUFFERED": "1"} | env
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending, bar, shown = "", None, 0.0
    while chunk := process.stdout.read1(1 << 16):
        pending += decoder.decode(chunk)
        while end := re.search(r"[\r\n]", pending):
            text, ending, pending = pending[: end.start()], end.group(), pending[end.end() :]
            if ending == "\r":
                bar = text or bar
                if text and time.monotonic() - shown > REDRAW:
                    shown = time.monotonic()
                    yield "\r" + text
            elif text or bar:
                yield text or bar
                bar = None
    if pending:
        yield pending
    if process.wait():
        raise RuntimeError(f"{command[0]} {command[1]} failed ({process.returncode})")


def convert(
    checkpoint: Checkpoint,
    work: Path,
    token: str | None,
    system: str | None = None,
    verify: bool = True,
    push: str | None = None,
    modelpack: bool = True,
    private: bool = False,
    cache: Path | None = None,
) -> Iterator[str | Result]:
    """Build, compare and push, yielding each step's output a line at a time,
    and last the result. Downloads go to `cache`, or Hugging Face's own."""
    refuse(checkpoint, system)
    env = {"SHEM_HOME": str(work / "shem")}
    # Given, rather than found where HF_HOME says, so a cache elsewhere doesn't lose it
    token = token or get_token()
    if token:
        env["HF_TOKEN"] = token
    if cache:
        env["HF_HUB_CACHE"] = str(cache)
    context = work / "context"
    context.mkdir(parents=True)
    (context / "Shemfile").write_text(shemfile(checkpoint, system))
    result = Result(checkpoint, work / "image")

    yield f"## Building {checkpoint.repo} at {checkpoint.commit[:12]}"
    yield from run([SHEM, "build", "-o", str(result.layout), str(context)], env)
    result.digest = json.loads((result.layout / "index.json").read_text())["manifests"][0]["digest"]

    if verify and checkpoint.verifiable:
        yield "## Comparing the import with transformers"
        unpack(result.layout, work / "root")
        report = work / "comparison.json"
        command = [sys.executable, "-m", "shem_neural.model.verify", checkpoint.repo, str(work / "root" / "model")]
        command += ["--revision", checkpoint.commit, "--report", str(report)]
        try:
            yield from run(command, env)
        except RuntimeError:
            # A comparison that disagrees exits 1 too, having written its report
            pass
        if report.is_file():
            result.comparison = json.loads(report.read_text())
        else:
            result.unverified = "The comparison with transformers failed to run."
            yield f"## {result.unverified}"
    elif verify:
        result.unverified = f"At {checkpoint.parameters / 1e9:.1f}B parameters, the model is larger than the Space compares with transformers."
        yield f"## {result.unverified}"
    else:
        result.unverified = "The comparison with transformers was skipped."

    if push:
        api = HfApi(token=token)
        api.create_repo(push, private=private, exist_ok=True)
        yield f"## Pushing to hf.co/{push}"
        yield from run([SHEM, "push", str(result.layout), f"hf.co/{push}"], env)
        if modelpack:
            yield f"## Pushing a ModelPack artifact to hf.co/{push}:modelpack"
            yield from run([SHEM, "push", "--modelpack", str(result.layout), f"hf.co/{push}:modelpack"], env)
        api.upload_file(
            path_or_fileobj=card(result, push, system, modelpack).encode(),
            path_in_repo="README.md",
            repo_id=push,
            commit_message="Describe the image",
        )
        # Where `shem registry`'s page of the image finds the comparison
        if result.comparison:
            api.upload_file(
                path_or_fileobj=json.dumps(result.comparison, indent=2).encode(),
                path_in_repo="comparison.json",
                repo_id=push,
                commit_message="Record the comparison with transformers",
            )
        result.url = f"https://huggingface.co/{push}"
        yield f"## Done: {result.url}"
    yield result


def card(result: Result, repo: str, system: str | None, modelpack: bool) -> str:
    """The model card of the image's repo."""
    checkpoint = result.checkpoint
    front = ["---", f"base_model: {checkpoint.repo}", "tags:", "- shem", "- oci"]
    if checkpoint.license:
        front.append(f"license: {checkpoint.license}")
    front.append("---")
    name = repo.split("/")[-1]
    lines = [
        *front,
        "",
        f"# {name}",
        "",
        f"[{checkpoint.repo}](https://huggingface.co/{checkpoint.repo}) at commit `{checkpoint.commit[:12]}`, as a [shem](https://github.com/JGalego/shem) model image: an OCI image layout, whose manifest is `{result.digest}`.",
        "",
    ]
    if system and system.strip():
        lines += [
            "Its system prompt is primed into a saved session, so a run starts after it:",
            "",
            "```text",
            system.strip(),
            "```",
            "",
        ]
    lines += [
        "## Run it",
        "",
        "```sh",
        f'shem run hf.co/{repo} "What is a golem?"',
        "```",
        "",
        "Through a registry, for other OCI clients:",
        "",
        "```sh",
        "shem registry &",
        f"oras pull localhost:5000/{repo}:latest",
        f'docker run --rm --runtime io.containerd.shem.v1 --platform linux/riscv32 localhost:5000/{repo} "What is a golem?"',
        "```",
        "",
    ]
    if modelpack:
        lines += [
            f"The `modelpack` branch holds the model as a [ModelPack](https://github.com/modelpack/model-spec) artifact over the same blobs: `localhost:5000/{repo}:modelpack`.",
            "",
        ]
    comparison = result.comparison
    if comparison:
        verdict = "✅ Verified" if comparison.get("passed") else "❌ Differs"
        lines += [
            f"## {verdict} against transformers",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| Largest difference between the prompt's logits, in float32 | {comparison['difference']:.2e} (tolerance 1e-3) |",
            f"| Greedy reply, {comparison['length']} tokens | {'the same tokens' if comparison['tokens'] else 'different tokens'} |",
            "",
        ]
    else:
        lines += ["## Not verified", "", result.unverified or "", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="a Hugging Face model, such as Qwen/Qwen3-0.6B")
    parser.add_argument("--system", help="a system prompt to prime into the image")
    parser.add_argument("--no-verify", dest="verify", action="store_false", help="don't compare with transformers")
    parser.add_argument("--push", metavar="ORG/NAME", help="the repo to push the image to")
    parser.add_argument("--modelpack", action="store_true", help="also push a ModelPack artifact")
    parser.add_argument("--private", action="store_true", help="create the repo private")
    parser.add_argument("--work", type=Path, help="where to build (default: a temporary directory, removed after)")
    args = parser.parse_args()
    api = HfApi()
    try:
        checkpoint = check(args.source, api)
        refuse(checkpoint, args.system)
    except Refused as error:
        raise SystemExit(str(error)) from None
    with tempfile.TemporaryDirectory() as temporary:
        work = args.work or Path(temporary)
        steps = convert(checkpoint, work, None, args.system, args.verify, args.push, args.modelpack, args.private)
        for item in steps:
            if isinstance(item, Result):
                print(json.dumps({"digest": item.digest, "comparison": item.comparison, "url": item.url}, indent=2))
            elif item.startswith("\r"):
                # A progress bar, redrawn in place
                print(f"\r\033[K{item[1:]}", end="", flush=True)
            else:
                print(f"\r\033[K{item}", flush=True)


if __name__ == "__main__":
    main()
