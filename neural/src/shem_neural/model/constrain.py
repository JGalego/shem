"""Constrained decoding: a reply that must be a JSON object.

`JsonPrefix` follows a text character by character and says whether it can
still become a JSON object. `JsonConstraint` uses it to mask a model's logits
at each step: only tokens that keep the reply a prefix of a JSON object can be
chosen, and once the object is closed only the end of the reply can. The reply
that comes out parses, whatever the model would have said.
"""

import copy
import re

import torch

#: A number's text so far can still become a number, and when it is one
_NUMBER_PREFIX = re.compile(r"-?(0|[1-9][0-9]*)?(\.[0-9]*)?([eE][+-]?[0-9]*)?")
_NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
_LITERALS = ("true", "false", "null")
#: How many of the likeliest tokens are tried before every token is
TOP = 256


class JsonPrefix:
    """A text that may be the start of a JSON object, read a character at a time."""

    def __init__(self):
        self.stack: list[str] = []
        #: What comes next: value, value_or_end, key_or_end, key, colon, comma_or_end, done
        self.expect = "object"
        self.string: tuple[bool, bool, int] | None = None  # (is a key, after a backslash, unicode digits left)
        self.literal = ""
        self.number = ""

    def copy(self) -> "JsonPrefix":
        other = copy.copy(self)
        other.stack = list(self.stack)
        return other

    @property
    def done(self) -> bool:
        return self.expect == "done"

    def feed(self, text: str) -> bool:
        """Read `text`; False if the whole is no longer the start of a JSON object."""
        return all(self._char(char) for char in text)

    def _value_done(self) -> None:
        self.expect = "comma_or_end" if self.stack else "done"

    def _char(self, ch: str) -> bool:
        if self.string is not None:
            return self._in_string(ch)
        if self.literal:
            self.literal += ch
            if not any(word.startswith(self.literal) for word in _LITERALS):
                return False
            if self.literal in _LITERALS:
                self.literal = ""
                self._value_done()
            return True
        if self.number:
            if ch in "0123456789.eE+-":
                self.number += ch
                return bool(_NUMBER_PREFIX.fullmatch(self.number))
            if not _NUMBER.fullmatch(self.number):
                return False
            self.number = ""
            self._value_done()
            return self._char(ch)
        if ch in " \t\r\n":
            return True
        return getattr(self, "_" + self.expect)(ch)

    def _in_string(self, ch: str) -> bool:
        assert self.string is not None
        is_key, escaped, unicode_left = self.string
        if unicode_left:
            if ch not in "0123456789abcdefABCDEF":
                return False
            self.string = (is_key, False, unicode_left - 1)
        elif escaped:
            if ch == "u":
                self.string = (is_key, False, 4)
            elif ch in '"\\/bfnrt':
                self.string = (is_key, False, 0)
            else:
                return False
        elif ch == "\\":
            self.string = (is_key, True, 0)
        elif ch == '"':
            self.string = None
            if is_key:
                self.expect = "colon"
            else:
                self._value_done()
        elif ord(ch) < 0x20:
            return False  # a control character must be escaped
        return True

    def _start_value(self, ch: str) -> bool:
        if ch == "{":
            self.stack.append("{")
            self.expect = "key_or_end"
        elif ch == "[":
            self.stack.append("[")
            self.expect = "value_or_end"
        elif ch == '"':
            self.string = (False, False, 0)
        elif ch == "-" or ch.isdigit():
            self.number = ch
        elif ch in "tfn":
            self.literal = ch
        else:
            return False
        return True

    def _object(self, ch: str) -> bool:  # the top level of a reply: an object, and only that
        return ch == "{" and self._start_value(ch)

    def _value(self, ch: str) -> bool:
        return self._start_value(ch)

    def _value_or_end(self, ch: str) -> bool:
        if ch == "]":
            self.stack.pop()
            self._value_done()
            return True
        return self._start_value(ch)

    def _key_or_end(self, ch: str) -> bool:
        if ch == "}":
            self.stack.pop()
            self._value_done()
            return True
        return self._key(ch)

    def _key(self, ch: str) -> bool:
        if ch != '"':
            return False
        self.string = (True, False, 0)
        return True

    def _colon(self, ch: str) -> bool:
        if ch != ":":
            return False
        self.expect = "value"
        return True

    def _comma_or_end(self, ch: str) -> bool:
        inside = self.stack[-1]
        if ch == ",":
            self.expect = "key" if inside == "{" else "value"
        elif ch == ("}" if inside == "{" else "]"):
            self.stack.pop()
            self._value_done()
        else:
            return False
        return True

    def _done(self, ch: str) -> bool:
        return False


class JsonConstraint:
    """The mask on a model's choices that keeps a reply a JSON object."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.state = JsonPrefix()
        self.words: dict[int, str] = {}

    def word(self, token: int) -> str:
        if token not in self.words:
            self.words[token] = self.tokenizer.decode([token])
        return self.words[token]

    def allows(self, token: int) -> bool:
        if token in self.tokenizer.eos:
            return self.state.done
        text = self.word(token)
        # A piece of a character the tokenizer cannot show alone is not a choice shem can check
        return bool(text) and "�" not in text and self.state.copy().feed(text)

    def mask(self, logits: torch.Tensor) -> torch.Tensor:
        """`logits` with every token that would break the object at minus infinity."""
        if self.state.done:
            order = torch.tensor(sorted(self.tokenizer.eos), device=logits.device)
        else:
            order = logits.float().topk(min(TOP, logits.numel())).indices
            if not any(self.allows(int(token)) for token in order):
                order = torch.arange(logits.numel(), device=logits.device)  # rare: scan the vocabulary
        allowed = [int(token) for token in order if self.allows(int(token))]
        masked = torch.full_like(logits, float("-inf"))
        if allowed:
            index = torch.tensor(allowed, device=logits.device)
            masked[index] = logits[index]
        else:
            masked[order[:1]] = 0  # nothing can continue it: end it as the model would
        return masked

    def advance(self, token: int) -> None:
        if token not in self.tokenizer.eos:
            self.state.feed(self.word(token))
