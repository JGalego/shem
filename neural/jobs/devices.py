# /// script
# requires-python = ">=3.12"
# dependencies = ["torch>=2.9", "numpy>=2", "safetensors>=0.4", "huggingface_hub>=1", "tokenizers>=0.22", "jinja2>=3"]
# ///
"""How far apart two devices' logits are, measured, for the receipts' rule of thumb.

    hf jobs uv run --flavor l4x1 --timeout 1h neural/jobs/devices.py --rev <commit>
    hf jobs uv run --flavor a100-large --timeout 1h neural/jobs/devices.py --rev <commit>

A receipt says a choice is firm unless its margin (how far its logit is ahead
of the runner-up's) is within a tolerance: 0.002 in float32 and 0.5 in
bfloat16. This runs the same conversations on each device the job has and in
each type, along the path that the job's CPU in float32 takes: at each step
the logits of the likeliest tokens, against that CPU's. It prints how far
they are apart (the largest and the 99th percentile of the difference), how
far the margin moves, and whether any choice flipped and with what margin, so
that the tolerance rests on measurement.
"""

import argparse
import io
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import torch

REPOS = ["HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen3-0.6B"]
PROMPTS = [
    "Name three uses of clay, in one sentence.",
    "Write a haiku about the sea.",
    "What is 17 times 23? Answer with the number only.",
    "Explain what a hash table is to a child.",
    "List the planets of the solar system, in order.",
    "Translate 'good morning' into French, Spanish and German.",
    "Why is the sky blue? Be brief.",
    "Give me a one-line summary of the plot of Hamlet.",
    "What is the capital of Australia, and why isn't it Sydney?",
    "Write a Python function that reverses a string.",
    "Describe the taste of a lemon.",
    "What should I pack for a day hike?",
]
STEPS = 64
TOP = 16


def fetch(rev: str) -> Path:
    url = f"https://github.com/JGalego/shem/archive/{rev}.tar.gz"
    root = Path(tempfile.mkdtemp())
    with urllib.request.urlopen(url) as response:
        tarfile.open(fileobj=io.BytesIO(response.read())).extractall(root, filter="data")
    return next(root.iterdir()) / "neural"


def reference_path(model, prompt: list[int]) -> tuple[list[int], list[torch.Tensor]]:
    """The greedy reply on `model`, with the logits at each step."""
    state = model.init_state()
    logits = model.forward(torch.tensor([prompt]), state, 0)[0, -1].float()
    reply, all_logits = [], [logits]
    for position in range(len(prompt), len(prompt) + STEPS - 1):
        token = int(logits.argmax())
        reply.append(token)
        logits = model.forward(torch.tensor([[token]]), state, position)[0, -1].float()
        all_logits.append(logits)
    reply.append(int(logits.argmax()))
    return reply, all_logits


def teacher_forced(model, prompt: list[int], reply: list[int], device) -> list[torch.Tensor]:
    """The logits at each step along the path `reply`, whatever the model would have chosen."""
    state = model.init_state()
    logits = model.forward(torch.tensor([prompt], device=device), state, 0)[0, -1].float().cpu()
    out = [logits]
    for position, token in zip(
        range(len(prompt), len(prompt) + STEPS - 1), reply, strict=False
    ):  # the path's last token is not fed
        logits = model.forward(torch.tensor([[token]], device=device), state, position)[0, -1].float().cpu()
        out.append(logits)
    return out


def compare(reference: list[torch.Tensor], other: list[torch.Tensor]) -> dict:
    """Differences between two devices' logits along one path."""
    deltas, margin_moves, margins, flips = [], [], [], []
    for ref, got in zip(reference, other, strict=True):
        top = ref.topk(TOP)
        deltas.append((got[top.indices] - top.values).abs())
        ref_margin = float(top.values[0] - top.values[1])
        got_top = got.topk(2)
        margin_moves.append(abs(float(got_top.values[0] - got_top.values[1]) - ref_margin))
        margins.append(ref_margin)
        if int(got.argmax()) != int(ref.argmax()):
            flips.append(ref_margin)
    delta = torch.cat(deltas)
    return {
        "positions": len(reference),
        "max logit difference": float(delta.max()),
        "p99 logit difference": float(delta.quantile(0.99)),
        "max margin movement": max(margin_moves),
        "smallest margin": min(margins),
        "flips": len(flips),
        "largest margin that flipped": max(flips) if flips else None,
        # A choice is firm when its margin exceeds twice the logits' difference
        "firm if margin >": 2 * float(delta.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rev", default="main")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    neural = Path(__file__).resolve().parent.parent if args.local else fetch(args.rev)
    sys.path.insert(0, str(neural / "src"))
    from shem_neural.model.blocks import Model
    from shem_neural.model.importers import import_model
    from shem_neural.model.tokenizer import Tokenizer

    cuda = torch.cuda.is_available()
    print(
        json.dumps(
            {
                "rev": args.rev,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name() if cuda else None,
                "cpu threads": torch.get_num_threads(),
            }
        ),
        flush=True,
    )
    report = {}
    for repo in REPOS:
        out = Path(tempfile.mkdtemp())
        import_model(repo, out)
        reference = Model(out, device="cpu")
        tokenizer = Tokenizer(out, reference.manifest.tokenizer)
        configs = {"cpu bfloat16": ("cpu", torch.bfloat16)}
        if cuda:
            configs |= {"cuda float32": ("cuda", torch.float32), "cuda bfloat16": ("cuda", torch.bfloat16)}
        models = {name: Model(out, dtype=dtype, device=device) for name, (device, dtype) in configs.items()}
        found = {name: {"reference": [], "other": []} for name in configs}
        with torch.no_grad():
            for text in PROMPTS:
                prompt = tokenizer.chat([{"role": "user", "content": text}])
                reply, ref_logits = reference_path(reference, prompt)
                for name, model in models.items():
                    other = teacher_forced(model, prompt, reply, configs[name][0])
                    found[name]["reference"] += ref_logits
                    found[name]["other"] += other
        report[repo] = {name: compare(f["reference"], f["other"]) for name, f in found.items()}
        for name, row in report[repo].items():
            print(repo, name, json.dumps(row), flush=True)
    print("RESULTS " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
