"""Check a receipt that `shem run --receipt` wrote, on this machine.

    python replay.py RECEIPT [--prompt TEXT]

The steps the shem replay Space runs, which work on their own too:

1. Read the receipt, and refuse what this machine won't replay: an image
   that is not in a public Hugging Face repo, a session from elsewhere, or
   more than it can hold.
2. Pull the image, `shem pull hf.co/ORG/NAME`, which must be the one the
   receipt names, by its digest, and run nothing but a model.
3. Reply to the receipt's prompt again, `shem replay RECEIPT`, and compare
   each token the model chooses with the receipt's.
4. With `--prompt`, reply to that prompt on the same image, and show how the
   reply differs: nothing is sampled, so the prompt is all that changed.

Each step is yielded as it comes: a note, the reply as far as it has got,
and the verdict.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

from shem_neural.hub import revision
from shem_neural.model.manifest import Manifest, locate
from shem_neural.model.tokenizer import Detokenizer, Tokenizer

FORMAT = "shem.receipt/v1"
#: What fits in a free CPU Space, and what a visitor waits for: the bytes of
#: an image, the tokens of a reply, and the seconds a replay may take
MAX_BYTES = int(float(os.environ.get("SHEM_REPLAY_MAX_BYTES", 2.5e9)))
MAX_TOKENS = int(os.environ.get("SHEM_REPLAY_MAX_TOKENS", 512))
TIMEOUT = int(os.environ.get("SHEM_REPLAY_TIMEOUT", 900))
#: The bytes of images kept from one replay to the next, before they are all deleted
MAX_KEPT = int(float(os.environ.get("SHEM_REPLAY_MAX_KEPT", 20e9)))
SHEM = os.environ.get("SHEM_BIN", "shem")
#: Where shem keeps images, as it finds it itself
HOME = Path(
    os.environ.get("SHEM_HOME") or Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "shem"
)
#: An image in a Hugging Face repo, as a receipt names one
IMAGE = re.compile(
    r"(?P<reference>hf\.co/(?P<repo>[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)(?::(?P<tag>\w[\w.-]{0,127}))?)"
    r"@(?P<digest>sha256:[0-9a-f]{64})"
)
#: The driver's options an image may give: a model that replies, and no more.
#: Not its tools, which are programs, nor a server
OPTIONS = {"--max-tokens", "--system", "--system-file", "--stop", "--dtype"}
#: What `shem replay` exits with, in a word
VERDICTS = {0: "verified", 1: "mismatch", 3: "undecided"}
#: The margin under which another device may choose another token, as shem has it
CLOSE = {"float32": 2e-3, "bfloat16": 0.5}


class Refused(Exception):
    """A receipt this machine won't replay, and why."""


@dataclass
class Note:
    """A step begins."""

    text: str


@dataclass
class Reply:
    """The model's reply, as far as it has got."""

    text: str


@dataclass
class Verdict:
    """How the replay compares with the receipt: `shem replay`'s status, the
    word for it, and the sentence shem says why in."""

    status: int
    word: str
    why: str
    reply: str = ""
    #: The receipt's tokens as text, each with its margin, for a replay that ran
    pieces: list[tuple[str, float]] = field(default_factory=list)


def read(text: str) -> dict:
    """The receipt in `text`, if it is one this machine replays."""
    try:
        receipt = json.loads(text)
        whole = receipt["format"] == FORMAT and len(receipt["tokens"]) == len(receipt["margins"])
        whole = whole and all(isinstance(receipt[key], str) for key in ("prompt", "reply"))
    except (ValueError, KeyError, TypeError):
        whole = False
    if not whole:
        raise Refused(f"This is not a {FORMAT} receipt, as `shem run --receipt FILE` writes one.")
    image = receipt.get("image")
    if not isinstance(image, str) or not IMAGE.fullmatch(image):
        raise Refused(
            "The receipt's image is not one in a Hugging Face repo, named as `hf.co/ORG/NAME@sha256:…`: "
            "this replays images anyone can pull from the Hub. Replay it yourself with `shem replay`."
        )
    if "state" in receipt:
        raise Refused("The receipt's run continued a published session, which this does not fetch.")
    if len(receipt["tokens"]) > MAX_TOKENS:
        raise Refused(f"The receipt's reply has {len(receipt['tokens'])} tokens, and this replays up to {MAX_TOKENS}.")
    if not receipt["prompt"].strip():
        raise Refused("The receipt has no prompt.")
    return receipt


def admit(receipt: dict, api: HfApi | None = None) -> None:
    """Refuse a receipt whose image the Hub doesn't hold, for anyone, as an
    image of a size this machine can take."""
    named = IMAGE.fullmatch(receipt["image"])
    repo, tag = named["repo"], named["tag"]
    try:
        info = (api or HfApi(token=False)).model_info(repo, revision=revision(tag), files_metadata=True)
    except GatedRepoError:
        raise Refused(f"{repo} is gated: this replays images anyone can pull.") from None
    except (RepositoryNotFoundError, RevisionNotFoundError):
        raise Refused(f"Hugging Face has no public repo {repo}{f' at {tag}' if tag else ''}.") from None
    files = {sibling.rfilename: sibling.size or 0 for sibling in info.siblings}
    if "oci-layout" not in files:
        raise Refused(
            f"{repo} holds a checkpoint, not an image: this replays receipts of images, "
            "as `shem push IMAGE hf.co/ORG/NAME` or the shem my repo Space writes them."
        )
    size = sum(files.values())
    if size > MAX_BYTES:
        raise Refused(
            f"{repo}'s image takes {size / 1e9:.1f} GB, and this replays images up to {MAX_BYTES / 1e9:g} GB."
        )


def shem(*args: str, timeout: int = TIMEOUT) -> subprocess.CompletedProcess:
    env = os.environ | {"PYTHONUNBUFFERED": "1", "SHEM_HOME": str(HOME)}
    return subprocess.run([SHEM, *args], env=env, capture_output=True, text=True, timeout=timeout)


def inspect(reference: str) -> dict | None:
    """What shem's store holds as `reference`, if it holds it."""
    done = shem("inspect", reference)
    return json.loads(done.stdout)[0] if done.returncode == 0 else None


def tidy() -> None:
    """Delete the images kept, once they take more than this machine keeps."""
    kept = sum(path.lstat().st_size for path in HOME.rglob("*") if path.is_file()) if HOME.is_dir() else 0
    if kept > MAX_KEPT:
        shutil.rmtree(HOME, ignore_errors=True)


def pull(receipt: dict) -> str:
    """Have the receipt's image in the store, and return its reference: the
    image the repo holds now, which must be the receipt's, and one that
    runs a model and nothing else."""
    named = IMAGE.fullmatch(receipt["image"])
    reference, digest = named["reference"], named["digest"]
    held = inspect(reference)
    if not held or held["Id"] != digest:
        # Not pulled yet, or pulled before the repo was pushed to again
        if held:
            shem("rmi", "-f", reference)
        tidy()
        done = shem("pull", reference)
        if done.returncode:
            raise Refused(f"Pulling {reference} failed: {last(done.stderr)}")
        held = inspect(reference)
    if not held or held["Id"] != digest:
        found = held["Id"] if held else "nothing"
        raise Refused(
            f"{reference} now holds {found}, not the receipt's {digest}: has the repo been pushed to since? "
            "With the image the receipt was made with, `shem replay RECEIPT IMAGE` still checks it."
        )
    entrypoint = held["Config"]["Entrypoint"]
    if held.get("Kind") != "model" or entrypoint[:1] != ["shem:generate"]:
        raise Refused(f"{reference} is not a model image.")
    for option in (word.split("=", 1)[0] for word in entrypoint[1:] if word.startswith("--")):
        if option not in OPTIONS:
            raise Refused(f"{reference} runs its model with {option}, which this does not run.")
    return reference


def last(text: str) -> str:
    """The last line of what shem said, without its name."""
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1].removeprefix("shem: ") if lines else ""


def run(command: list[str]) -> Iterator[str | tuple[int, str]]:
    """Run shem with `command`, yielding what it has written to its output
    so far, each time there is more, and once it has ended, its status and
    what it said besides: its errors, and its verdict."""
    env = os.environ | {"PYTHONUNBUFFERED": "1", "SHEM_HOME": str(HOME)}
    with tempfile.TemporaryFile("w+") as said:
        process = subprocess.Popen([SHEM, *command], env=env, stdout=subprocess.PIPE, stderr=said, text=True)
        timer = threading.Timer(TIMEOUT, process.kill)
        timer.start()
        try:
            text = ""
            while piece := process.stdout.read(1):
                text += piece
                # A word at a time is often enough
                if piece.isspace():
                    yield text
            yield text
            process.wait()
        finally:
            timer.cancel()
            process.kill()
        said.seek(0)
        yield process.returncode, said.read()


def pieces(receipt: dict) -> list[tuple[str, float]]:
    """The receipt's tokens as the text each added to the reply, with its
    margin, read with the tokenizer of its image, which the store has
    unpacked. A token that ends a reply adds none."""
    digest = IMAGE.fullmatch(receipt["image"])["digest"]
    try:
        directory, _ = locate(HOME / "rootfs" / digest.removeprefix("sha256:"))
        tokenizer = Tokenizer(directory, Manifest.load(directory).tokenizer)
    except (OSError, ValueError, KeyError):
        return []
    detokenizer, found = Detokenizer(tokenizer), []
    for token, margin in zip(receipt["tokens"], receipt["margins"], strict=True):
        if token in tokenizer.eos:
            found.append(("", margin))
            detokenizer = Detokenizer(tokenizer)
        else:
            found.append((detokenizer.push(token), margin))
    return found


def replay(text: str) -> Iterator[Note | Reply | Verdict]:
    """Check the receipt in `text`: each step as it begins, the reply as it
    comes, and the verdict. Raises `Refused` for a receipt not replayed."""
    receipt = read(text)
    named = IMAGE.fullmatch(receipt["image"])
    yield Note(f"Looking for {named['reference']} on Hugging Face")
    admit(receipt)
    yield Note(f"Pulling {named['reference']}, unless it is here already")
    reference = pull(receipt)
    yield Note(f"Replying to the receipt's prompt on {reference}")
    with tempfile.TemporaryDirectory() as work:
        file = Path(work) / "receipt.json"
        file.write_text(json.dumps(receipt))
        reply = ""
        for item in run(["replay", str(file)]):
            if isinstance(item, str):
                yield Reply(item)
                reply = item
            else:
                status, said = item
    if status not in VERDICTS:
        raise Refused(f"The replay did not run: {last(said) or f'shem exited with {status}'}")
    word, _, why = last(said).partition(": ")
    yield Verdict(status, VERDICTS[status], why or word, reply.strip(), pieces(receipt))


def ask(text: str, prompt: str) -> Iterator[Note | Reply | dict]:
    """Reply to `prompt` on the image of the receipt in `text`, which a
    replay has pulled: the reply as it comes, and then its own receipt."""
    receipt = read(text)
    if not prompt.strip():
        raise Refused("Give a prompt.")
    reference = pull(receipt)
    yield Note(f"Replying to the changed prompt on {reference}")
    with tempfile.TemporaryDirectory() as work:
        file = Path(work) / "receipt.json"
        for item in run(["run", "--receipt", str(file), reference, prompt]):
            if isinstance(item, str):
                yield Reply(item)
            else:
                status, said = item
        if status or not file.is_file():
            raise Refused(f"The run failed: {last(said) or f'shem exited with {status}'}")
        yield json.loads(file.read_text())


def difference(before: str, after: str) -> list[tuple[str, str]]:
    """How `after` differs from `before`, a word at a time: each run of words
    with whether it is the `same`, was `removed`, or was `added`."""
    old, new = re.findall(r"\s+|\S+", before), re.findall(r"\s+|\S+", after)
    runs = []
    for kind, a, b, c, d in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if kind == "equal":
            runs.append(("same", "".join(old[a:b])))
        else:
            runs += [(name, "".join(words)) for name, words in (("removed", old[a:b]), ("added", new[c:d])) if words]
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("receipt", type=Path, help="the receipt, as shem run --receipt wrote it")
    parser.add_argument("--prompt", help="also reply to this prompt on the receipt's image, and show the difference")
    args = parser.parse_args()
    text = args.receipt.read_text()
    status = 125
    try:
        for item in replay(text):
            if isinstance(item, Note):
                print(f"… {item.text}", file=sys.stderr)
            elif isinstance(item, Verdict):
                status = item.status
                print(item.reply)
                print(f"{item.word}: {item.why}", file=sys.stderr)
        if args.prompt:
            for item in ask(text, args.prompt):
                if isinstance(item, Note):
                    print(f"… {item.text}", file=sys.stderr)
                elif isinstance(item, dict):
                    marks = {"same": "{}", "removed": "[-{}-]", "added": "{{+{}+}}"}
                    print(
                        "".join(
                            marks[kind].format(words) for kind, words in difference(read(text)["reply"], item["reply"])
                        )
                    )
    except Refused as error:
        sys.exit(str(error))
    sys.exit(status)


if __name__ == "__main__":
    main()
