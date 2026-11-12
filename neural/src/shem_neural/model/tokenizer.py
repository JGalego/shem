"""Render checkpoint chat templates and encode or decode their tokens."""

import json
from datetime import datetime
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer as HfTokenizer


def _raise(message: str):
    raise ValueError(message)


class Tokenizer:
    def __init__(self, directory: Path, spec: dict):
        directory = Path(directory)
        self.inner = HfTokenizer.from_file(str(directory / spec["file"]))
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = lambda value, indent=None, **_: json.dumps(value, ensure_ascii=False, indent=indent)
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: datetime(2026, 1, 1).strftime(fmt)  # the guest's virtual clock
        self.template = env.from_string((directory / spec["chat_template"]).read_text())
        self.eos = set(spec["eos"])
        #: The special tokens the template sees, such as `bos_token`
        self.special = spec.get("special", {})

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> list[int]:
        """The tokens of `messages`, ready for the assistant's reply."""
        return self.encode(self.render(messages, tools))

    def render(self, messages: list[dict], tools: list[dict] | None = None, reply: bool = True) -> str:
        """The text of `messages` in the chat template, offering `tools`, and
        with `reply`, ready for the assistant's reply."""
        return self.template.render(
            messages=messages, tools=tools or None, add_generation_prompt=reply, enable_thinking=False, **self.special
        )

    def stop_texts(self) -> list[str]:
        """The text of each token that ends a reply, such as `<|im_end|>`."""
        return [self.inner.id_to_token(i) for i in self.eos if self.inner.id_to_token(i)]

    def encode(self, text: str) -> list[int]:
        return self.inner.encode(text, add_special_tokens=False).ids

    def prefix(self, messages: list[dict], tools: list[dict] | None = None) -> list[int]:
        """The tokens every conversation that starts with `messages` and
        continues with a user turn begins with. Found by rendering two such
        conversations and keeping what they share, so it works with any
        template, including ones that refuse a conversation with no user turn."""
        a = self.chat(messages + [{"role": "user", "content": "A"}], tools)
        b = self.chat(messages + [{"role": "user", "content": "?"}], tools)
        shared = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        return a[:shared]

    def decode(self, ids: list[int]) -> str:
        return self.inner.decode(ids, skip_special_tokens=True)


class Detokenizer:
    """Turns a stream of tokens into a stream of text, holding back the end
    of a character split across tokens until it is complete."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.sent = 0

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self.tokenizer.decode(self.ids)
        if text.endswith("�"):
            return ""
        new, self.sent = text[self.sent :], len(text)
        return new
