"""`shem:generate`: run a session of a model from its files.

    python -m shem_neural.model.generate [--max-tokens N] [--system TEXT] MODEL_DIR [PROMPT...]
    python -m shem_neural.model.generate --system TEXT --warm OUT MODEL_DIR
    python -m shem_neural.model.generate --session DIR MODEL_DIR [PROMPT...]

The prompt is the arguments, or stdin when there are none. The reply streams
to stdout. Decoding is greedy, so a prompt always gets the same reply.

`--warm` primes a session with the system prompt and saves it: every block's
state after reading it, which is what `WARM --until=prompt` puts in a state
layer. `--session` continues a saved session instead of starting afresh.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .blocks import Model
from .tokenizer import Detokenizer, Tokenizer

SESSION_FORMAT = "shem.session/v1"


class Session:
    """A conversation in progress: its tokens so far, and every block's state
    after reading them, which the host can save and restore."""

    def __init__(self, model: Model):
        self.model = model
        self.state = model.init_state()
        self.tokens: list[int] = []
        self.logits: torch.Tensor | None = None

    def feed(self, tokens: list[int]) -> None:
        if tokens:
            logits = self.model.forward(torch.tensor([tokens]), self.state, len(self.tokens))
            self.tokens += tokens
            self.logits = logits[0, -1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tensors = {
            f"{i}.{name}": tensor.contiguous()
            for i, block in enumerate(self.state)
            if block
            for name, tensor in block.items()
        }
        save_file(tensors, str(directory / "session.safetensors"))
        meta = {"format": SESSION_FORMAT, "tokens": self.tokens}
        (directory / "session.json").write_text(json.dumps(meta) + "\n")

    @classmethod
    def load(cls, model: Model, directory: Path) -> "Session":
        meta = json.loads((directory / "session.json").read_text())
        if meta.get("format") != SESSION_FORMAT:
            raise ValueError(f"{directory}: not a {SESSION_FORMAT} session")
        session = cls(model)
        session.tokens = meta["tokens"]
        for key, tensor in load_file(str(directory / "session.safetensors")).items():
            block, name = key.split(".", 1)
            session.state[int(block)][name] = tensor.to(model.dtype)
        return session


def reply(session: Session, tokenizer: Tokenizer, max_tokens: int, out=sys.stdout) -> list[int]:
    """Continue `session` with the model's reply, writing the text to `out`
    as it comes. Returns the reply's tokens."""
    detokenizer = Detokenizer(tokenizer)
    tokens = []
    for _ in range(max_tokens):
        token = int(session.logits.argmax())
        if token in tokenizer.eos:
            break
        tokens.append(token)
        out.write(detokenizer.push(token))
        out.flush()
        session.feed([token])
    out.write("\n")
    out.flush()
    return tokens


def options(argv: list[str], model: str | None = None) -> argparse.Namespace:
    """The driver's arguments: its options, the model, and the prompt. With
    `model`, `argv` holds only options and the prompt, in any order, as a
    container's command line does."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--system", help="a system prompt")
    parser.add_argument("--warm", type=Path, help="prime a session with the system prompt and save it here")
    parser.add_argument("--session", type=Path, help="continue the session saved here")
    parser.add_argument("--stats", action="store_true", help="print the token rate to stderr")
    if model is not None:
        parser.add_argument("prompt", nargs="*")
        args = parser.parse_intermixed_args(argv)
        args.model = model
        return args
    parser.add_argument("model", help="the model's directory, holding shem.json")
    parser.add_argument("prompt", nargs="*")
    return parser.parse_args(argv)


def run(model: Model, tokenizer: Tokenizer, args: argparse.Namespace, prompt: str, out=sys.stdout, err=sys.stderr):
    """One run of the driver: prime and save a session with `--warm`, or
    reply to `prompt`, continuing the `--session` if there is one."""
    system = [{"role": "system", "content": args.system}] if args.system else []
    if args.warm:
        if not system:
            raise ValueError("priming a session needs a system prompt: give --system")
        session = Session(model)
        session.feed(tokenizer.prefix(system))
        session.save(args.warm)
        print(f"shem: primed a session with {len(session.tokens)} tokens", file=err)
        return
    session = Session.load(model, args.session) if args.session else Session(model)
    tokens = tokenizer.chat(system + [{"role": "user", "content": prompt}])
    if tokens[: len(session.tokens)] != session.tokens:
        raise ValueError("the saved session is not the start of this conversation: was it saved with another --system?")
    started = time.perf_counter()
    session.feed(tokens[len(session.tokens) :])
    replied = reply(session, tokenizer, args.max_tokens, out)
    if args.stats:
        print(f"shem: {len(replied)} tokens in {time.perf_counter() - started:.1f} s", file=err)


def main() -> None:
    args = options(sys.argv[1:])
    model = Model(args.model)
    tokenizer = Tokenizer(args.model, model.manifest.tokenizer)
    prompt = " ".join(args.prompt) or ("" if args.warm else sys.stdin.read())
    try:
        run(model, tokenizer, args, prompt)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
