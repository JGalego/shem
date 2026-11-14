"""Receipts: a run written down, and checked by running it again, on the
tiny model `test_sessions` keeps."""

import io
import json
from pathlib import Path

import pytest

from shem_neural.model import receipt
from shem_neural.model.blocks import Model
from shem_neural.model.generate import options, resolve, run
from shem_neural.model.tokenizer import Tokenizer

MODEL = Path(__file__).parent / "sessions" / "model"
PROMPT = "t3 t17 t42"


@pytest.fixture(scope="module")
def driver():
    model = Model(MODEL)
    tokenizer = Tokenizer(MODEL, model.manifest.tokenizer)

    def drive(*argv: str, prompt: str = PROMPT) -> tuple[int, str, str]:
        """Run the driver with these options: its status, reply and messages."""
        args = resolve(options(["--max-tokens", "12", *argv], model=str(MODEL)))
        out, err = io.StringIO(), io.StringIO()
        return run(model, tokenizer, args, prompt, out, err), out.getvalue(), err.getvalue()

    return drive


@pytest.fixture
def written(driver, tmp_path):
    """A receipt of a run, as its file and what it holds."""
    path = tmp_path / "receipt.json"
    status, reply, log = driver("--receipt", str(path), "--subject", "image=tiny@sha256:abc")
    assert status == 0 and "wrote a receipt of 12 tokens, the closest call by" in log
    return path, json.loads(path.read_text()), reply


def test_receipts_say_what_was_asked_and_what_was_chosen(written):
    _, held, reply = written
    assert list(held) == ["format", "image", "prompt", "reply", "tokens", "margins", "runtime"]
    assert held["image"] == "tiny@sha256:abc" and held["prompt"] == PROMPT
    assert held["reply"] == reply.strip() == " ".join(f"t{token}" for token in held["tokens"])
    assert len(held["margins"]) == 12 and all(margin >= 0 for margin in held["margins"])
    assert held["runtime"]["device"] == "cpu" and held["runtime"]["precision"] == "float32"
    assert receipt.closest(held) == min((margin, at + 1) for at, margin in enumerate(held["margins"]))


def test_replays_choose_a_receipts_tokens_again(driver, written):
    path, _, reply = written
    # The prompt is the receipt's, whatever the command line says
    assert driver("--replay", str(path), prompt=PROMPT) == (
        0,
        reply,
        "shem: verified: the model chose the receipt's 12 tokens again\n",
    )


def test_replays_part_from_receipts_that_are_not_the_models(driver, written, tmp_path):
    path, held, _ = written

    def replay(**changes) -> tuple[int, str]:
        receipt.save(held | changes, tmp_path / "changed.json")
        status, _, log = driver("--replay", str(tmp_path / "changed.json"), prompt=(held | changes)["prompt"])
        return status, log

    # Another prompt gets another reply, from its first token
    status, log = replay(prompt="t5 t40 t18")
    assert status == receipt.DIFFERENT and "mismatch: the replay parts from the receipt at token 1 of 12" in log
    # A token the model did not choose, where its choice was firm
    firm = max(range(12), key=lambda at: held["margins"][at])
    tokens = held["tokens"].copy()
    tokens[firm] = (tokens[firm] + 1) % 64
    status, log = replay(tokens=tokens)
    assert status == receipt.DIFFERENT and f"at token {firm + 1} of 12" in log
    assert "this image does not give that reply to this prompt" in log
    # The same where the receipt's choice was a close call settles nothing
    margins = held["margins"].copy()
    margins[firm] = 1e-5
    status, log = replay(tokens=tokens, margins=margins)
    assert status == receipt.UNDECIDED and "a close call, by 1e-05" in log
    # The model's tokens, and a reply they do not spell
    status, log = replay(reply="Certainly.")
    assert status == receipt.DIFFERENT and "they are not the reply it gives" in log
    # A receipt cut short, and one made by another runtime
    status, log = replay(tokens=held["tokens"][:8], margins=held["margins"][:8])
    assert status == receipt.DIFFERENT and "the replay has 12 tokens and the receipt 8" in log
    status, log = replay(tokens=tokens, runtime=held["runtime"] | {"torch": "1.0", "device": "cuda"})
    assert "the receipt's device, torch differ from this runtime's" in log


def test_only_receipts_replay(driver, tmp_path):
    for junk in ["", "{}", '{"format": "shem.receipt/v1", "tokens": [1], "margins": [], "prompt": "", "reply": ""}']:
        (tmp_path / "junk.json").write_text(junk)
        with pytest.raises(ValueError, match="not a shem.receipt/v1 receipt"):
            driver("--replay", str(tmp_path / "junk.json"))
