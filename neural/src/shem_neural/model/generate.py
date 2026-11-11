"""`shem:generate`: run a session of a model from its files.

    python -m shem_neural.model.generate [--max-tokens N] [--system TEXT] MODEL_DIR [PROMPT...]
    python -m shem_neural.model.generate --system TEXT --warm OUT MODEL_DIR
    python -m shem_neural.model.generate --session DIR MODEL_DIR [PROMPT...]

The prompt comes from arguments or stdin; greedy decoding streams to stdout.
`--warm` saves a system-prompt session; `--session` resumes one. The model
server keeps sessions across turns and saves them for `docker commit`.
`--tools DIR` offers programs run on the image's compiled CPU (see `tools`).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load, load_file, save, save_file

from .blocks import Model
from .tokenizer import Detokenizer, Tokenizer
from .tools import Tools, compiled_layers, tool_calls

SESSION_FORMAT = "shem.session/v1"
#: Tool calls a turn may make before the model must answer
MAX_TOOL_ROUNDS = 4
TOOL_CALL = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
#: Tokens a tool call may run past `--max-tokens` to finish: a number is a token per digit
TOOL_CALL_ALLOWANCE = 1024


class Session:
    """A conversation in progress: its tokens so far, and every block's state
    after reading them, which the host can save and restore."""

    def __init__(self, model: Model):
        self.model = model
        self.state = model.init_state()
        self.tokens: list[int] = []
        self.logits: torch.Tensor | None = None
        #: The conversation so far, as chat messages
        self.messages: list[dict] = []
        #: The tools the conversation offers, as the chat template takes them
        self.tools: list[dict] = []
        #: Programs the model can call, run on its compiled layer
        self.runner: Tools | None = None

    def feed(self, tokens: list[int]) -> None:
        if tokens:
            logits = self.model.forward(torch.tensor([tokens]), self.state, len(self.tokens))
            self.tokens += tokens
            self.logits = logits[0, -1]

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            f"{i}.{name}": tensor.contiguous().cpu()
            for i, block in enumerate(self.state)
            if block
            for name, tensor in block.items()
        }

    def meta(self) -> dict:
        return {"format": SESSION_FORMAT, "tokens": self.tokens, "messages": self.messages, "tools": self.tools}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        save_file(self.tensors(), str(directory / "session.safetensors"))
        (directory / "session.json").write_text(json.dumps(self.meta()) + "\n")

    def to_files(self) -> dict[str, bytes]:
        """The session as the files `save` writes, in memory."""
        return {
            "session.json": (json.dumps(self.meta()) + "\n").encode(),
            "session.safetensors": save(self.tensors()),
        }

    @classmethod
    def load(cls, model: Model, directory: Path) -> "Session":
        meta = json.loads((directory / "session.json").read_text())
        return cls._from(model, meta, load_file(str(directory / "session.safetensors")))

    @classmethod
    def from_files(cls, model: Model, files: dict[str, bytes]) -> "Session":
        return cls._from(model, json.loads(files["session.json"]), load(files["session.safetensors"]))

    @classmethod
    def _from(cls, model: Model, meta: dict, tensors: dict[str, torch.Tensor]) -> "Session":
        if meta.get("format") != SESSION_FORMAT:
            raise ValueError(f"not a {SESSION_FORMAT} session")
        session = cls(model)
        session.tokens = meta["tokens"]
        session.messages = meta.get("messages", [])
        session.tools = meta.get("tools", [])
        for key, tensor in tensors.items():
            block, name = key.split(".", 1)
            session.state[int(block)][name] = tensor.to(device=model.device, dtype=model.dtype)
        return session


def next_turn(session: Session, tokenizer: Tokenizer, new: list[dict]) -> list[int]:
    """The tokens that take `session` from where it is to the start of the
    model's reply to the `new` messages: a user's turn, or tools' replies.

    A conversation that has not replied yet must be the start of the new one,
    token for token. After a reply, the session holds the model's own tokens
    up to the token that ended it, which it did not read: the turn is that
    token, then whatever the template puts after the reply. When the template
    renders the old conversation the same way with the new messages added,
    that is the difference of the two; otherwise it is what follows the
    reply's text, found before the new messages. If neither works, the
    session starts over and reads the whole conversation again."""
    messages = session.messages + new
    tools = session.tools
    if not session.messages or session.messages[-1]["role"] != "assistant":
        tokens = tokenizer.chat(messages, tools)
        if tokens[: len(session.tokens)] != session.tokens:
            raise ValueError(
                "the saved session is not the start of this conversation: was it saved with another --system?"
            )
        return tokens[len(session.tokens) :]
    after = tokenizer.render(messages, tools)
    try:
        before = tokenizer.render(session.messages, tools, reply=False)
    except Exception:  # templates may refuse a conversation that ends with a reply
        before = ""
    stops = [(before.rfind(stop), stop) for stop in tokenizer.stop_texts()]
    at, _ = max(stops, default=(-1, ""))
    if before and after.startswith(before) and at >= 0:
        return tokenizer.encode(before[at:] + after[len(before) :])
    last = session.messages[-1]["content"].strip()
    end = after.rfind(new[-1]["content"]) if new[-1]["content"] else len(after)
    at = after.rfind(last, 0, end if end >= 0 else len(after)) if last else -1
    if at < 0:
        session.state, session.tokens = session.model.init_state(), []
        return tokenizer.chat(messages, tools)
    return tokenizer.encode(after[at + len(last) :])


def converse(session: Session, tokenizer: Tokenizer, text: str, max_tokens: int, out=sys.stdout, err=sys.stderr):
    """One turn: the user's `text`, then the model's reply, written to `out`
    as it comes. When the reply calls tools, the host runs them on the
    compiled layer and the model replies again, up to `MAX_TOOL_ROUNDS`
    times. The session remembers it all. Returns the last reply's tokens."""
    new = [{"role": "user", "content": text}]
    for _ in range(MAX_TOOL_ROUNDS + 1):
        session.feed(next_turn(session, tokenizer, new))
        replied = reply(session, tokenizer, max_tokens, out, hold=session.runner is not None)
        content = tokenizer.decode(replied).strip()
        session.messages += new
        calls = tool_calls(content) if session.runner is not None else []
        if not calls:
            session.messages.append({"role": "assistant", "content": content})
            return replied
        visible = content[: content.find(TOOL_CALL)].strip()
        session.messages.append(
            {
                "role": "assistant",
                "content": visible,
                "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls],
            }
        )
        new = [{"role": "tool", "content": session.runner.run(name, arguments, err)} for name, arguments in calls]
    return replied


def reply(session: Session, tokenizer: Tokenizer, max_tokens: int, out=sys.stdout, hold: bool = False) -> list[int]:
    """Continue `session` with the model's reply, writing the text to `out`
    as it comes. With `hold`, a tool call is not written: text stops at the
    first `<tool_call>`, holding back what may be its start, and a tool call
    may run past `max_tokens` to finish. Held text that turns out not to be a
    complete tool call is written at the end. Returns the reply's tokens."""
    detokenizer = Detokenizer(tokenizer)
    tokens = []
    text, written = "", 0

    def calling() -> bool:
        call = text.find(TOOL_CALL)
        return call >= 0 and TOOL_CALL_END not in text[call:]

    while len(tokens) < max_tokens or (hold and calling() and len(tokens) < max_tokens + TOOL_CALL_ALLOWANCE):
        token = int(session.logits.argmax())
        if token in tokenizer.eos:
            break
        tokens.append(token)
        text += detokenizer.push(token)
        shown = len(text)
        if hold:
            call = text.find(TOOL_CALL)
            shown = call if call >= 0 else len(text) - _partial(text, TOOL_CALL)
        if shown > written:
            out.write(text[written:shown])
            out.flush()
            written = shown
        session.feed([token])
    if hold and TOOL_CALL_END not in text[written:] and written < len(text):
        out.write(text[written:])
        written = len(text)
    if written or not hold:
        out.write("\n")
        out.flush()
    return tokens


def _partial(text: str, tag: str) -> int:
    """How many characters at the end of `text` could start `tag`."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-size:]):
            return size
    return 0


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
    parser.add_argument("--keep", action="store_true", help="in a container, keep the session open for docker exec")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto: a GPU if there is one")
    parser.add_argument("--tools", help="offer the programs in this directory of the image to the model as tools")
    if model is not None:
        parser.add_argument("prompt", nargs="*")
        args = parser.parse_intermixed_args(argv)
        args.model = model
        return args
    parser.add_argument("model", help="the model's directory, holding shem.json")
    parser.add_argument("prompt", nargs="*")
    return parser.parse_args(argv)


def tools_for(model_dir: Path, directory: str, vcpu_socket: Path | None = None) -> Tools:
    """The programs in `directory` of the image whose model is at
    `model_dir`, run on its compiled RV32IM layer."""
    layers = compiled_layers(model_dir)
    if "rv32im" not in layers:
        raise ValueError(
            "--tools runs programs on the model's compiled RV32IM layer, and this model has none: "
            "add it with COPY --from=shem:rv32im / /model/compiled/rv32im/"
        )
    return Tools(Path(model_dir).parent, directory, layers["rv32im"] / "model.safetensors", vcpu_socket)


def start(
    model: Model, tokenizer: Tokenizer, args: argparse.Namespace, err=sys.stderr, vcpu_socket: Path | None = None
) -> Session | None:
    """The session a run of the driver works on: primed and saved with
    `--warm` (which returns None), the `--session` it continues, or a new one
    with the system prompt. With `--tools`, the session offers the image's
    programs, run on a vCPU server at `vcpu_socket` or one of its own."""
    system = [{"role": "system", "content": args.system}] if args.system else []
    runner = tools_for(Path(args.model), args.tools, vcpu_socket) if args.tools else None
    tools = runner.schemas() if runner else []
    if args.warm:
        if not system:
            raise ValueError("priming a session needs a system prompt: give --system")
        session = Session(model)
        session.tools = tools
        session.feed(tokenizer.prefix(system, tools))
        session.messages = system
        session.save(args.warm)
        print(f"shem: primed a session with {len(session.tokens)} tokens", file=err)
        return None
    session = Session.load(model, args.session) if args.session else Session(model)
    if not session.messages:
        session.messages, session.tools = system, tools
    session.runner = runner
    return session


def run(model: Model, tokenizer: Tokenizer, args: argparse.Namespace, prompt: str, out=sys.stdout, err=sys.stderr):
    """One run of the driver: prime and save a session with `--warm`, or
    reply to `prompt`, continuing the `--session` if there is one."""
    session = start(model, tokenizer, args, err)
    if session is None:
        return
    started = time.perf_counter()
    replied = converse(session, tokenizer, prompt, args.max_tokens, out, err)
    if args.stats:
        print(f"shem: {len(replied)} tokens in {time.perf_counter() - started:.1f} s", file=err)


def main() -> None:
    args = options(sys.argv[1:])
    model = Model(args.model, device=args.device)
    tokenizer = Tokenizer(args.model, model.manifest.tokenizer)
    prompt = " ".join(args.prompt) or ("" if args.warm else sys.stdin.read())
    try:
        run(model, tokenizer, args, prompt)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
