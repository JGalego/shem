# /// script
# requires-python = ">=3.12"
# dependencies = ["torch>=2.9", "numpy>=2", "safetensors>=0.4", "huggingface_hub>=1", "tokenizers>=0.22", "jinja2>=3"]
# ///
"""Model sessions on a GPU: check them against the CPU, and time them.

    hf jobs uv run --flavor l4x1 --timeout 30m neural/jobs/models_gpu.py --rev <commit>

For Qwen3-0.6B and Qwen3.5-0.8B: the prompt's logits on the GPU against the
CPU, the same greedy reply from both, a session saved on the GPU that
continues as if it had never stopped, and decoding speed on each.
"""

import argparse
import io
import json
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import torch

REPOS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"]
MESSAGES = [{"role": "user", "content": "Name three uses of clay, in one sentence."}]


def fetch(rev: str) -> Path:
    url = f"https://github.com/JGalego/shem/archive/{rev}.tar.gz"
    root = Path(tempfile.mkdtemp())
    with urllib.request.urlopen(url) as response:
        tarfile.open(fileobj=io.BytesIO(response.read())).extractall(root, filter="data")
    return next(root.iterdir()) / "neural"


def greedy(model, tokens: list[int], steps: int) -> tuple[list[int], float]:
    state = model.init_state()
    logits = model.forward(torch.tensor([tokens]), state, 0)
    reply, started = [], time.perf_counter()
    for position in range(len(tokens), len(tokens) + steps):
        token = int(logits[0, -1].argmax())
        reply.append(token)
        logits = model.forward(torch.tensor([[token]]), state, position)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    return reply, steps / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rev", default="main")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--steps", type=int, default=48)
    args = parser.parse_args()
    neural = Path(__file__).resolve().parent.parent if args.local else fetch(args.rev)
    sys.path.insert(0, str(neural / "src"))
    from shem_neural.model.blocks import Model
    from shem_neural.model.generate import Session, converse
    from shem_neural.model.importers import import_model
    from shem_neural.model.tokenizer import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        json.dumps(
            {
                "rev": args.rev,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            }
        ),
        flush=True,
    )
    report, ok = {}, True
    for repo in REPOS:
        out = Path(tempfile.mkdtemp())
        import_model(repo, out)
        cpu, gpu = Model(out, device="cpu"), Model(out, device=device)
        tokenizer = Tokenizer(out, cpu.manifest.tokenizer)
        prompt = tokenizer.chat(MESSAGES)

        with torch.no_grad():
            difference = (
                (
                    gpu.forward(torch.tensor([prompt]), gpu.init_state(), 0).cpu()
                    - cpu.forward(torch.tensor([prompt]), cpu.init_state(), 0)
                )
                .abs()
                .max()
                .item()
            )
        cpu_reply, cpu_rate = greedy(cpu, prompt, args.steps)
        gpu_reply, gpu_rate = greedy(gpu, prompt, args.steps)

        # A session saved on the GPU continues as if it had never stopped
        def turn(session, text, tokenizer=tokenizer):
            sink = io.StringIO()
            converse(session, tokenizer, text, 16, sink)
            return sink.getvalue()

        live = Session(gpu)
        turn(live, "My name is Joana.")
        copy = Session.from_files(gpu, live.to_files())
        resumed = turn(live, "What did I just tell you my name is?") == turn(
            copy, "What did I just tell you my name is?"
        )

        same = cpu_reply == gpu_reply
        ok &= same and resumed and difference < 1e-3
        report[repo] = {
            "max logit difference": difference,
            "same greedy reply": same,
            "resumed session matches": resumed,
            "cpu tokens/s": round(cpu_rate, 1),
            "gpu tokens/s": round(gpu_rate, 1),
            "reply": tokenizer.decode(gpu_reply),
        }
        print(repo, json.dumps(report[repo]), flush=True)
    print("RESULTS " + json.dumps(report), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
