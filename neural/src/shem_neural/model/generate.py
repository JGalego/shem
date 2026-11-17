"""`shem:generate`: run a session of a model from its files.

    python -m shem_neural.model.generate [--max-tokens N] [--system TEXT] MODEL_DIR [PROMPT...]
    python -m shem_neural.model.generate --system TEXT --warm OUT MODEL_DIR
    python -m shem_neural.model.generate --session DIR MODEL_DIR [PROMPT...]

The prompt comes from arguments or stdin; greedy decoding streams to stdout.
`--warm` saves a system-prompt session; `--session` resumes one. The model
server keeps sessions across turns and saves them for `docker commit`.
`--tools DIR` offers programs run on the image's compiled CPU (see `tools`).
`--system-file PATH` adds a file of the image to the system prompt, such as a
document to answer from. `--stop TEXT` ends a reply where it reaches TEXT,
which a base model, with no end of turn of its own, needs. `--serve PORT`
serves OpenAI's chat completions API instead (see `serve`). `--receipt FILE`
writes a receipt of the run, and `--replay FILE` checks one (see `receipt`).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load, load_file, save, save_file

from . import receipt
from .blocks import Model
from .manifest import locate
from .sampling import Sampler
from .tokenizer import Detokenizer, Tokenizer
from .tools import Tools, compiled_layers, tool_calls

SESSION_FORMAT = "shem.session/v1"
#: Tool calls a turn may make before the model must answer
MAX_TOOL_ROUNDS = 4
TOOL_CALL = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
#: Tokens a tool call may run past `--max-tokens` to finish: a number is a token per digit
TOOL_CALL_ALLOWANCE = 1024


class ContextFull(ValueError):
    """A conversation longer than the model's context window."""


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
        #: Each token the model chose and its margin, when a list is put here
        self.choices: list[tuple[int, float]] | None = None
        #: How the next token is drawn, if not the likeliest
        self.sampler: Sampler | None = None
        #: Each token chosen, its log probability and the likeliest others', when a list is put here
        self.logprobs: list[tuple[int, float, list[tuple[int, float]]]] | None = None
        self.top_logprobs = 0

    def choose(self) -> int:
        token = self._choose()
        if self.logprobs is not None:
            logp = torch.log_softmax(self.logits.float(), dim=-1)
            top = logp.topk(self.top_logprobs) if self.top_logprobs else None
            self.logprobs.append(
                (
                    token,
                    float(logp[token]),
                    list(zip(top.indices.tolist(), top.values.tolist(), strict=True)) if top else [],
                )
            )
        return token

    def _choose(self) -> int:
        """The token the session continues with: the likeliest. A session
        that keeps its choices adds it, with its margin: how far its logit
        is ahead of the runner-up's."""
        if self.sampler is not None and not self.sampler.greedy:
            token, margin = self.sampler.draw(self.logits)
            if self.choices is not None:
                self.choices.append((token, margin))
            return token
        token = int(self.logits.argmax())
        if self.choices is not None:
            first, second = self.logits.float().topk(2).values.tolist()
            self.choices.append((token, first - second))
        return token

    def room(self) -> int | None:
        """How many more tokens the session can read before it fills the
        model's context window, or None if the model has no window."""
        context = self.model.manifest.context
        return None if context is None else context - len(self.tokens)

    def feed(self, tokens: list[int]) -> None:
        room = self.room()
        if room is not None and len(tokens) > room:
            raise ContextFull(
                f"the conversation would be {len(self.tokens) + len(tokens)} tokens, "
                f"past the model's context window of {self.model.manifest.context}"
            )
        if tokens:
            logits = self.model.forward(torch.tensor([tokens]), self.state, len(self.tokens))
            self.tokens += tokens
            self.logits = logits[0, -1]

    @staticmethod
    def feed_batch(sessions: list["Session"], tokens: list[int]) -> None:
        """One more token for each of several sessions of one model, read in
        a single pass that reads the weights once. Each session ends as if
        `feed([token])` had read its token alone, up to rounding."""
        if len(sessions) == 1:
            return sessions[0].feed(tokens)
        model = sessions[0].model
        for session in sessions:
            if (room := session.room()) is not None and room < 1:
                raise ContextFull("the conversation is at its model's context window")
        logits = model.forward_batch(tokens, [s.state for s in sessions], [len(s.tokens) for s in sessions])
        for session, token, row in zip(sessions, tokens, logits, strict=True):
            session.tokens.append(token)
            session.logits = row

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            f"{i}.{name}": tensor.contiguous().cpu()
            for i, block in enumerate(self.state)
            if block
            for name, tensor in block.items()
            if not name.startswith("_")
        }

    def meta(self) -> dict:
        return {
            "format": SESSION_FORMAT,
            "precision": self.model.precision,
            "adapters": self.model.adapter_names,
            "blocks": self.model.versions,
            "tokens": self.tokens,
            "messages": self.messages,
            "tools": self.tools,
        }

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
    def load(cls, model: Model, directory: Path, err=sys.stderr) -> "Session":
        meta = json.loads((directory / "session.json").read_text())
        return cls._from(model, meta, lambda: load_file(str(directory / "session.safetensors")), err)

    @classmethod
    def from_files(cls, model: Model, files: dict[str, bytes], err=sys.stderr) -> "Session":
        return cls._from(model, json.loads(files["session.json"]), lambda: load(files["session.safetensors"]), err)

    @classmethod
    def _from(cls, model: Model, meta: dict, tensors, err) -> "Session":
        """The session `meta` describes, with its blocks' state from
        `tensors()`, or, if this runtime would read that state otherwise than
        the one that saved it, from its tokens read again."""
        if meta.get("format") != SESSION_FORMAT:
            raise ValueError(f"not a {SESSION_FORMAT} session")
        session = cls(model)
        session.messages = meta.get("messages", [])
        session.tools = meta.get("tools", [])
        if stale := staleness(model, meta):
            print(f"shem: the session was saved {stale}: reading its {len(meta['tokens'])} tokens again", file=err)
            session.feed(meta["tokens"])
            return session
        session.tokens = meta["tokens"]
        for key, tensor in tensors().items():
            block, name = key.split(".", 1)
            # In the types it was saved in, which are this model's, as its precision is
            session.state[int(block)][name] = tensor.to(device=model.device)
        return session


def staleness(model: Model, meta: dict) -> str | None:
    """How the runtime that saved the session `meta` describes differs from
    the one that runs `model`, if its state would mean something else here:
    another precision, or another version of a type of block. Sessions saved
    before they recorded these were saved in float32 by each block's first
    version."""
    precision = meta.get("precision", "float32")
    if precision != model.precision:
        return f"in {precision}, and this model runs in {model.precision}"
    if (adapters := meta.get("adapters", [])) != model.adapter_names:
        return f"with adapters {adapters or 'none'}, and this model has {model.adapter_names or 'none'}"
    saved, here = meta.get("blocks", dict.fromkeys(model.versions, 1)), model.versions
    changed = sorted(kind for kind in saved.keys() | here.keys() if saved.get(kind) != here.get(kind))
    if changed:
        return f"by another version of shem's {', '.join(changed)} blocks"
    return None


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


def converse(
    session: Session,
    tokenizer: Tokenizer,
    text: str,
    max_tokens: int,
    out=sys.stdout,
    err=sys.stderr,
    stops: list[str] | None = None,
):
    """One turn: the user's `text`, then the model's reply, written to `out`
    as it comes. When the reply calls tools, the host runs them on the
    compiled layer and the model replies again, up to `MAX_TOOL_ROUNDS`
    times. The session remembers it all. Returns the last reply's tokens."""
    new = [{"role": "user", "content": text}]
    for _ in range(MAX_TOOL_ROUNDS + 1):
        session.feed(next_turn(session, tokenizer, new))
        replied = reply(session, tokenizer, max_tokens, out, hold=session.runner is not None, stops=stops)
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


class Reply:
    """A reply being made, a token at a time: `next` chooses and writes the
    model's next token and returns it for the caller to read into the
    session, which a caller with several sessions of one model can do for all
    of them at once. `reply` is the caller with one."""

    def __init__(
        self,
        session: Session,
        tokenizer: Tokenizer,
        max_tokens: int,
        out=sys.stdout,
        hold: bool = False,
        stops: list[str] | None = None,
    ):
        self.session, self.tokenizer, self.max_tokens = session, tokenizer, max_tokens
        self.out, self.hold = out, hold
        self.stops = [stop for stop in stops or [] if stop]
        self.detokenizer = Detokenizer(tokenizer)
        self.tokens: list[int] = []
        self.text, self.written = "", 0
        self.done = False

    def _calling(self) -> bool:
        call = self.text.find(TOOL_CALL)
        return call >= 0 and TOOL_CALL_END not in self.text[call:]

    def next(self) -> int | None:
        """The model's next token, written, which is to be read into the
        session; or None if the reply is over (it has then written its end)."""
        session, tokens = self.session, self.tokens
        running = len(tokens) < self.max_tokens or (
            self.hold and self._calling() and len(tokens) < self.max_tokens + TOOL_CALL_ALLOWANCE
        )
        # A full window ends a reply where it can read no more
        full = (room := session.room()) is not None and room < 1
        if self.done or not running or full:
            return self._end()
        token = session.choose()
        if token in self.tokenizer.eos:
            return self._end()
        self.text += self.detokenizer.push(token)
        ends = [at for at in (self.text.find(stop) for stop in self.stops) if at >= 0]
        if ends:
            self.text = self.text[: min(ends)]
            return self._end()
        tokens.append(token)
        shown = len(self.text)
        if self.hold:
            call = self.text.find(TOOL_CALL)
            shown = call if call >= 0 else len(self.text) - _partial(self.text, TOOL_CALL)
        if self.stops:
            shown = min(shown, len(self.text) - max(_partial(self.text, stop) for stop in self.stops))
        if shown > self.written:
            self.out.write(self.text[self.written : shown])
            self.out.flush()
            self.written = shown
        return token

    def _end(self) -> None:
        if not self.done:
            self.done = True
            if self.written < len(self.text) and not (self.hold and TOOL_CALL_END in self.text[self.written :]):
                self.out.write(self.text[self.written :])
                self.written = len(self.text)
            if self.written or not self.hold:
                self.out.write("\n")
                self.out.flush()
        return None


def reply(
    session: Session,
    tokenizer: Tokenizer,
    max_tokens: int,
    out=sys.stdout,
    hold: bool = False,
    stops: list[str] | None = None,
) -> list[int]:
    """Continue `session` with the model's reply, writing the text to `out`
    as it comes. With `hold`, a tool call is not written: text stops at the
    first `<tool_call>`, holding back what may be its start, and a tool call
    may run past `max_tokens` to finish. Held text that turns out not to be a
    complete tool call is written at the end. The reply ends before the
    first of `stops` it reaches, whose start is held back until it is clear
    whether it is one; the token that completes it is not read, as an end of
    turn isn't. Returns the reply's tokens."""
    made = Reply(session, tokenizer, max_tokens, out, hold, stops)
    while (token := made.next()) is not None:
        session.feed([token])
    return made.tokens


def _partial(text: str, tag: str) -> int:
    """How many characters at the end of `text` could start `tag`."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-size:]):
            return size
    return 0


#: The types a model can compute in: float32, and bfloat16, which halves the
#: memory its weights and sessions take, and is faster on GPUs that have it
DTYPES = ["float32", "bfloat16"]


def dtype(name: str) -> torch.dtype:
    """The torch type `--dtype` names."""
    return getattr(torch, name)


def options(argv: list[str], model: str | None = None) -> argparse.Namespace:
    """The driver's arguments: its options, the model, and the prompt. With
    `model`, `argv` holds only options and the prompt, in any order, as a
    container's command line does."""
    # No abbreviations: an image's options are checked by their full names
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--stop", action="append", default=[], help="end the reply where it reaches this text; may be given again"
    )
    parser.add_argument("--system", help="a system prompt")
    parser.add_argument(
        "--system-file", help="a file of the image whose text the system prompt ends with, such as a document"
    )
    parser.add_argument("--warm", type=Path, help="prime a session with the system prompt and save it here")
    parser.add_argument("--session", type=Path, help="continue the session saved here")
    parser.add_argument("--stats", action="store_true", help="print the token rate to stderr")
    parser.add_argument("--receipt", type=Path, help="write a receipt of the run here, for --replay to check")
    parser.add_argument(
        "--replay", type=Path, help="reply to this receipt's prompt, and check that the tokens are the receipt's"
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="with --receipt, what the run was of, such as image=NAME@DIGEST; may be given again",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="a conversation: each line of input is a turn, till its end; also what no prompt on a terminal means",
    )
    parser.add_argument("--keep", action="store_true", help="in a container, keep the session open for docker exec")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto: a GPU if there is one")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=DTYPES,
        help="the type the model computes in: bfloat16 takes half the memory",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="with --serve, let pages of this origin call the API, or * for any; may be given again",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME",
        help="merge a LoRA adapter into the weights: its directory in the image, or a name under the model's "
        "adapters/; may be given again, and they merge in order",
    )
    parser.add_argument(
        "--quantize",
        choices=["int8"],
        help="hold the projections' weights as int8, a quarter of float32's memory (embeddings stay as they are)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="draw each token from the model's distribution, sharpened or flattened by this; 0 is greedy",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="draw only from the likeliest tokens whose probabilities add to this"
    )
    parser.add_argument("--top-k", type=int, default=0, help="draw only from this many likeliest tokens; 0 is all")
    parser.add_argument(
        "--seed",
        type=int,
        help="start the draws from this number, so that the reply can be drawn again (default: random, "
        "and recorded in a receipt)",
    )
    parser.add_argument("--tools", help="offer the programs in this directory of the image to the model as tools")
    parser.add_argument(
        "--serve",
        metavar="[HOST:]PORT",
        help="serve OpenAI's chat completions API at this port, on localhost or a container's network, "
        "instead of replying once",
    )
    if model is not None:
        parser.add_argument("prompt", nargs="*")
        args = parser.parse_intermixed_args(argv)
        args.model = model
        return args
    parser.add_argument("model", help="the model's directory, holding shem.json, or an image's root holding one")
    parser.add_argument("prompt", nargs="*")
    return parser.parse_args(argv)


def adapter_paths(model_dir: Path, root: Path, names: list[str]) -> list[Path]:
    """The directories of the adapters `--adapter` names, each as a path in
    the image (under its root, which the model server reads with the host's
    view of the filesystem: a path that leads out is refused) or a name under
    the model's `adapters/`."""
    found = []
    for name in names:
        for base in (Path(root).resolve(), (Path(model_dir) / "adapters").resolve()):
            candidate = (base / name.lstrip("/")).resolve()
            if candidate.is_relative_to(base) and (candidate / "adapter_model.safetensors").is_file():
                found.append(candidate)
                break
        else:
            raise ValueError(f"--adapter {name}: no adapter_model.safetensors there, or in the model's adapters/")
    return found


def tools_for(model_dir: Path, root: Path, directory: str, vcpu_socket: Path | None = None) -> Tools:
    """The programs in `directory` of the image at `root`, whose model is at
    `model_dir`, run on the model's compiled RV32IM layer."""
    layers = compiled_layers(model_dir)
    if "rv32im" not in layers:
        raise ValueError(
            "--tools runs programs on the model's compiled RV32IM layer, and this model has none: "
            "copy it into the model's compiled/rv32im/, as in COPY --from=shem:rv32im / /model/compiled/rv32im/"
        )
    return Tools(Path(root), directory, layers["rv32im"] / "model.safetensors", vcpu_socket)


def resolve(args: argparse.Namespace) -> argparse.Namespace:
    """`args` with the model's directory as `model` and the image's root as
    `root`, for a `model` given as either."""
    args.model, args.root = locate(args.model)
    if args.system_file:
        text = system_file(Path(args.root), args.system_file)
        args.system = f"{args.system}\n\n{text}" if args.system else text
    return args


def system_file(root: Path, path: str) -> str:
    """The text of the file at `path` in the image whose root is `root`. The
    model server reads it with the host's view of the filesystem, so a path
    or a link that leads out of the image is refused."""
    base = root.resolve()
    target = (base / path.lstrip("/")).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"--system-file {path} leads out of the image")
    return target.read_text().strip()


def sampler(args: argparse.Namespace) -> Sampler | None:
    """How the options say to draw tokens, if not greedily."""
    if not args.temperature:
        if args.top_p != 1 or args.top_k or args.seed is not None:
            raise ValueError(
                "--top-p, --top-k and --seed need a --temperature above 0: without one, decoding is greedy"
            )
        return None
    return Sampler(args.temperature, args.top_p, args.top_k, args.seed)


def start(
    model: Model, tokenizer: Tokenizer, args: argparse.Namespace, err=sys.stderr, vcpu_socket: Path | None = None
) -> Session | None:
    """The session a run of the driver works on: primed and saved with
    `--warm` (which returns None), the `--session` it continues, or a new one
    with the system prompt. With `--tools`, the session offers the image's
    programs, run on a vCPU server at `vcpu_socket` or one of its own."""
    system = [{"role": "system", "content": args.system}] if args.system else []
    runner = tools_for(Path(args.model), Path(args.root), args.tools, vcpu_socket) if args.tools else None
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
    session = Session.load(model, args.session, err) if args.session else Session(model)
    if not session.messages:
        session.messages, session.tools = system, tools
    session.runner = runner
    session.sampler = sampler(args)
    return session


def run(
    model: Model, tokenizer: Tokenizer, args: argparse.Namespace, prompt: str, out=sys.stdout, err=sys.stderr
) -> int:
    """One run of the driver: prime and save a session with `--warm`, or
    reply to `prompt`, continuing the `--session` if there is one. With
    `--receipt`, the reply is written down, and with `--replay`, `prompt` is
    a receipt's, which the reply is checked against. Returns the driver's
    exit status: 0, unless a replay is not the receipt's."""
    checked = receipt.load(args.replay) if args.replay else None
    if checked and checked.get("messages"):
        # A conversation the API answered: replayed as it was asked
        status, why = receipt.replay_messages(model, tokenizer, checked, out)
        print(f"shem: {receipt.VERDICTS[status]}: {why}", file=err)
        return status
    session = start(model, tokenizer, args, err)
    if session is None:
        return 0
    if args.receipt or checked:
        session.choices = []
    if checked and checked.get("sampling"):
        # Drawn tokens are checked by drawing them again: from the receipt's seed
        session.sampler = Sampler(**checked["sampling"])
    started = time.perf_counter()
    replied = converse(session, tokenizer, prompt, args.max_tokens, out, err, args.stop)
    if args.stats:
        print(f"shem: {len(replied)} tokens in {time.perf_counter() - started:.1f} s", file=err)
    if args.receipt:
        written = receipt.record(session, prompt, dict(subject.split("=", 1) for subject in args.subject))
        receipt.save(written, args.receipt)
        print(f"shem: wrote a receipt of {receipt.summary(written)}, to {args.receipt}", file=err)
    if checked:
        status, why = receipt.check(checked, session)
        print(f"shem: {receipt.VERDICTS[status]}: {why}", file=err)
        return status
    return 0


def chat(
    session: Session,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    prompt: str = "",
    lines=None,
    out=sys.stdout,
    err=sys.stderr,
    restart=None,
) -> int:
    """A conversation: `prompt` first, if there is one, then a turn for each
    line of `lines` (stdin), until it ends or `/exit`. On a terminal, a `> `
    asks for each, with the line editing and history `input` gives. `/clear`
    starts the conversation over, with a session `restart` makes."""
    terminal = lines is None and sys.stdin.isatty()
    lines = lines if lines is not None else sys.stdin
    if terminal:
        try:
            import readline  # noqa: F401  (input uses it for line editing and history)
        except ImportError:
            pass
        print("shem: a conversation; /clear starts over, /exit or Ctrl-D ends it", file=err)
    text = prompt
    while True:
        if not text:
            try:
                line = input("> ") if terminal else lines.readline()
            except EOFError:
                break
            if not line:
                break
            text = line.strip()
            if text in ("/exit", "/quit"):
                break
            if text == "/clear" and restart is not None:
                session, text = restart(), ""
                continue
            if not text:
                continue
        converse(session, tokenizer, text, args.max_tokens, out, err, args.stop)
        text = ""
    return 0


def main() -> None:
    args = resolve(options(sys.argv[1:]))
    model = Model(
        args.model,
        dtype=dtype(args.dtype),
        device=args.device,
        quantize=args.quantize,
        adapters=adapter_paths(args.model, args.root, args.adapter),
    )
    tokenizer = Tokenizer(args.model, model.manifest.tokenizer)
    if args.serve and not args.warm:
        from .serve import serve_forever

        primed = Session.load(model, args.session) if args.session else None
        serve_forever(model, tokenizer, args, primed)
        return
    try:
        talking = args.chat or not (args.prompt or args.warm or args.replay or args.receipt) and sys.stdin.isatty()
        if talking:
            if args.receipt or args.replay or args.warm:
                raise ValueError("--chat is a conversation: --receipt, --replay and --warm are for one prompt")
            session = start(model, tokenizer, args)
            try:
                sys.exit(
                    chat(session, tokenizer, args, " ".join(args.prompt), restart=lambda: start(model, tokenizer, args))
                )
            except KeyboardInterrupt:
                print(file=sys.stderr)
                sys.exit(130)
        if args.replay:
            prompt = receipt.load(args.replay)["prompt"]
        else:
            prompt = " ".join(args.prompt) or ("" if args.warm else sys.stdin.read())
        sys.exit(run(model, tokenizer, args, prompt))
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
