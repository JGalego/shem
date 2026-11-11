"""`shem:generate`: run a session of a model from its files.

    python -m shem_neural.model.generate MODEL_DIR [--max-tokens N] [--system TEXT] [PROMPT...]

The prompt is the arguments, or stdin when there are none. The reply streams
to stdout. Decoding is greedy, so a prompt always gets the same reply.
"""

import argparse
import sys
import time

import torch

from .blocks import Model
from .tokenizer import Detokenizer, Tokenizer


def generate(model: Model, tokenizer: Tokenizer, messages: list[dict], max_tokens: int, out=sys.stdout) -> list[int]:
    """Reply to `messages`, writing the text to `out` as it comes. Returns the reply's tokens."""
    prompt = tokenizer.chat(messages)
    state = model.init_state()
    logits = model.forward(torch.tensor([prompt]), state, 0)
    position = len(prompt)
    detokenizer = Detokenizer(tokenizer)
    reply = []
    for _ in range(max_tokens):
        token = int(logits[0, -1].argmax())
        if token in tokenizer.eos:
            break
        reply.append(token)
        out.write(detokenizer.push(token))
        out.flush()
        logits = model.forward(torch.tensor([[token]]), state, position)
        position += 1
    out.write("\n")
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="the model's directory, holding shem.json")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--system", help="a system prompt")
    parser.add_argument("--stats", action="store_true", help="print the token rate to stderr")
    parser.add_argument("prompt", nargs="*")
    args = parser.parse_args()

    prompt = " ".join(args.prompt) or sys.stdin.read()
    model = Model(args.model)
    tokenizer = Tokenizer(args.model, model.manifest.tokenizer)
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
        {"role": "user", "content": prompt}
    ]
    started = time.perf_counter()
    reply = generate(model, tokenizer, messages, args.max_tokens)
    if args.stats:
        elapsed = time.perf_counter() - started
        print(f"shem: {len(reply)} tokens in {elapsed:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
