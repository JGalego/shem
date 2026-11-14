"""Receipts: what a model was asked and what it replied, written down so
that anyone with the image can run it again and check.

    {
      "format": "shem.receipt/v1",
      "image": "hf.co/jgalego/smollm2-135m-instruct-shem@sha256:...",
      "prompt": "Who are you?",
      "reply": "I am Shem, a golem made of clay.",
      "tokens": [57, 744, 2],
      "margins": [4.3125, 0.27, 9.5],
      "runtime": {"shem": "0.3.0", "torch": "2.9.0", "device": "cpu", "machine": "x86_64",
                  "precision": "float32", "blocks": {"attention": 1}}
    }

`shem run --receipt FILE` writes one and `shem replay FILE` checks one. The
image's digest stands for everything the reply depends on besides the
prompt: the weights, the tokenizer, the system prompt, the session the image
was primed with, its tools and the CPU they run on. `state` names a
published session the run continued, the same way. Decoding is greedy, so
the same image and prompt choose the same tokens.

`tokens` are the tokens the model chose, in order: each reply's and the one
that ended it, those of tool calls included. `margins` say how firmly: how
far ahead of the runner-up's each one's logit was. Devices round sums
differently, so a token chosen by less than `CLOSE` may be another on
another device, and a replay that parts from the receipt there settles
nothing. One that parts where the choice was firm does: this image did not
make that reply from that prompt.
"""

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

FORMAT = "shem.receipt/v1"
#: The margin under which another device may choose another token, by the
#: type the model computes in: twice how far apart two devices' logits get.
#: float32's is twice what imports are held to against transformers;
#: bfloat16's is a few of its steps at a logit's size
CLOSE = {"float32": 2e-3, "bfloat16": 0.5}

#: What a replay finds, as the driver's exit status, and in a word
SAME, DIFFERENT, UNDECIDED = 0, 1, 3
VERDICTS = {SAME: "verified", DIFFERENT: "mismatch", UNDECIDED: "undecided"}


def runtime(model) -> dict:
    """What computed a reply, besides the image: this package and torch, the
    device, and the model's type and block versions on it."""
    try:
        package = version("shem-neural")
    except PackageNotFoundError:
        package = "unknown"
    cuda = model.device.type == "cuda"
    return {
        "shem": package,
        "torch": torch.__version__,
        "device": model.device.type,
        "machine": torch.cuda.get_device_name(model.device) if cuda else platform.machine(),
        "precision": model.precision,
        "blocks": model.versions,
    }


def replied(session) -> str:
    """The text of the session's last reply."""
    last = session.messages[-1] if session.messages else {}
    return last.get("content", "") if last.get("role") == "assistant" else ""


def record(session, prompt: str, subjects: dict[str, str] | None = None) -> dict:
    """The receipt of the turn `session` just took in reply to `prompt`,
    having kept its choices. `subjects` are what the host knows and the
    driver doesn't: the image's name and digest, and a session's."""
    choices = session.choices or []
    return {
        "format": FORMAT,
        **(subjects or {}),
        "prompt": prompt,
        "reply": replied(session),
        "tokens": [token for token, _ in choices],
        "margins": [round(margin, 6) for _, margin in choices],
        "runtime": runtime(session.model),
    }


def save(receipt: dict, path: Path) -> None:
    """Write `receipt` to the file at `path`, a field to a line."""
    fields = [f" {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}" for key, value in receipt.items()]
    Path(path).write_text("{\n" + ",\n".join(fields) + "\n}\n")


def load(path: Path) -> dict:
    """The receipt in the file at `path`."""
    try:
        receipt = json.loads(Path(path).read_text())
        whole = receipt["format"] == FORMAT and len(receipt["tokens"]) == len(receipt["margins"])
        whole = whole and isinstance(receipt["prompt"], str) and isinstance(receipt["reply"], str)
    except (ValueError, KeyError, TypeError):
        whole = False
    if not whole:
        raise ValueError(f"{path}: not a {FORMAT} receipt")
    return receipt


def closest(receipt: dict) -> tuple[float, int] | None:
    """The receipt's closest call: its margin, and which token it was, from 1."""
    margins = receipt["margins"]
    return min((margin, at + 1) for at, margin in enumerate(margins)) if margins else None


def summary(receipt: dict) -> str:
    """A receipt in a sentence, for whoever just made it."""
    count = len(receipt["tokens"])
    text = f"{count} token{'' if count == 1 else 's'}"
    if call := closest(receipt):
        margin, at = call
        text += f", the closest call by {margin:.3g} at token {at}"
        if margin < CLOSE.get(receipt["runtime"]["precision"], 0):
            text += ", which another device may make otherwise"
    return text


def check(receipt: dict, session) -> tuple[int, str]:
    """How `session`, which has just replied to the receipt's prompt and
    kept its choices, compares with `receipt`: `SAME`, `DIFFERENT` or
    `UNDECIDED`, and a sentence that says why."""
    want, margins = receipt["tokens"], receipt["margins"]
    got = session.choices or []
    here = runtime(session.model)
    shared = min(len(want), len(got))
    at = next((i for i in range(shared) if want[i] != got[i][0]), shared)
    if at == len(want) == len(got):
        if replied(session) != receipt["reply"]:
            return DIFFERENT, f"the model chose the receipt's {at} tokens again, but they are not the reply it gives"
        return SAME, f"the model chose the receipt's {at} tokens again"
    there = receipt.get("runtime", {})
    other = sorted(key for key in here if there.get(key) != here[key])
    made = f"; the receipt's {', '.join(other)} differ from this runtime's" if other else ""
    if at == shared:
        return DIFFERENT, f"the replay has {len(got)} tokens and the receipt {len(want)}{made}"
    where = f"token {at + 1} of {len(want)}"
    margin = min(margins[at], got[at][1])
    close = max(CLOSE.get(here["precision"], 0), CLOSE.get(there.get("precision"), 0))
    if margin < close:
        return UNDECIDED, (
            f"the replay parts from the receipt at {where}, a close call, by {margin:.3g}, "
            f"which devices may make differently: this settles nothing{made}"
        )
    return DIFFERENT, (
        f"the replay parts from the receipt at {where}, which the model chose by {margins[at]:.3g} "
        f"there and another by {got[at][1]:.3g} here: this image does not give that reply to this prompt{made}"
    )
