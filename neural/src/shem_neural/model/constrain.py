"""Constrained decoding: a reply that must be a JSON object, and optionally
one that fits a JSON schema.

`JsonPrefix` follows a text character by character and says whether it can
still become a JSON object (of the schema, if it was given one).
`JsonConstraint` uses it to mask a model's logits at each step: only tokens
that keep the reply a prefix of such an object can be chosen, and once the
object is closed only the end of the reply can. The reply that comes out
parses, and fits the schema, whatever the model would have said.

The schemas understood are those of a structured reply: `type` (one, or a
list), `enum`, `const` (of strings, numbers, booleans or null), `properties`,
`required`, `items`, `minItems` and `maxItems`, with `additionalProperties`
only as `false`. Where `properties` are given, only those keys are written.
`check_schema` refuses anything else, so a constraint is never quietly less
than what was asked.
"""

import copy
import json
import re

import torch

#: A number's text so far can still become a number, and when it is one
_NUMBER_PREFIX = re.compile(r"-?(0|[1-9][0-9]*)?(\.[0-9]*)?([eE][+-]?[0-9]*)?")
_NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
_LITERALS = ("true", "false", "null")
#: How many of the likeliest tokens are tried before every token is
TOP = 256
_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_KEYWORDS = {"type", "enum", "const", "properties", "required", "items", "minItems", "maxItems", "additionalProperties"}
#: Keywords that say nothing about which values fit
_NOTES = {"title", "description", "default", "examples", "$schema", "$id", "$comment", "name", "strict"}


def check_schema(schema, top: bool = True) -> None:
    """Raise ValueError unless `schema` is one that `JsonPrefix` enforces in full."""
    if not isinstance(schema, dict):
        raise ValueError("a JSON schema must be an object")
    for key in schema:
        if key not in _KEYWORDS | _NOTES:
            raise ValueError(f"the JSON schema keyword {key!r} is not supported: shem enforces {sorted(_KEYWORDS)}")
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not isinstance(kinds, list) or not kinds or any(kind not in _TYPES for kind in kinds):
            raise ValueError(f"type must be one of {sorted(_TYPES)}, or a list of them")
    if top and kinds != ["object"]:
        raise ValueError('the schema of a reply must be {"type": "object", ...}')
    for key in ("enum", "const"):
        if key in schema:
            options = schema["enum"] if key == "enum" else [schema["const"]]
            if not isinstance(options, list) or not options:
                raise ValueError("enum must be a list with something in it")
            if any(isinstance(item, dict | list) for item in options):
                raise ValueError(f"{key} can hold strings, numbers, booleans and null, not objects or arrays")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("properties must be an object")
    for sub in properties.values():
        check_schema(sub, top=False)
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
        raise ValueError("required must be a list of names")
    if required and not set(required) <= set(properties):
        raise ValueError("required names a property that properties does not")
    if schema.get("additionalProperties", False) is not False:
        raise ValueError("additionalProperties can only be false: keys that properties does not list are never written")
    if "items" in schema:
        check_schema(schema["items"], top=False)
    for key in ("minItems", "maxItems"):
        if key in schema and (isinstance(schema[key], bool) or not isinstance(schema[key], int) or schema[key] < 0):
            raise ValueError(f"{key} must be a number, 0 or more")


class JsonPrefix:
    """A text that may be the start of a JSON object, read a character at a time."""

    def __init__(self, schema: dict | None = None):
        self.stack: list[str] = []
        #: The schema of each open object or array, with what it has taken so far
        self.frames: list[dict] = []
        #: The schema of the value that comes next
        self.want: dict = schema or {}
        #: A text that must be one of these, which is the reply so far
        self.fixed: tuple[list[str], str, str] | None = None
        self.integer = False
        self.last_key: str | None = None
        self.allowed: tuple[str, ...] = _LITERALS
        #: What comes next: value, value_or_end, key_or_end, key, colon, comma_or_end, done
        self.expect = "object"
        self.string: tuple[bool, bool, int] | None = None  # (is a key, after a backslash, unicode digits left)
        self.literal = ""
        self.number = ""

    def copy(self) -> "JsonPrefix":
        other = copy.copy(self)
        other.stack = list(self.stack)
        other.frames = [{**frame, "used": set(frame["used"])} for frame in self.frames]
        return other

    @property
    def done(self) -> bool:
        return self.expect == "done"

    def feed(self, text: str) -> bool:
        """Read `text`; False if the whole is no longer the start of a JSON object."""
        return all(self._char(char) for char in text)

    def _value_done(self) -> None:
        self.expect = "comma_or_end" if self.stack else "done"
        if self.frames:
            self.frames[-1]["count"] += 1

    def _types(self) -> set[str]:
        kinds = self.want.get("type")
        if kinds is None:
            return _TYPES
        kinds = {kinds} if isinstance(kinds, str) else set(kinds)
        return kinds | ({"integer"} if "number" in kinds else set())

    def _fixed_char(self, ch: str) -> bool:
        """One more character of a text that must be one of a few: a key, or an enum's value."""
        assert self.fixed is not None
        options, so_far, kind = self.fixed
        text = so_far + ch
        if any(option.startswith(text) for option in options):
            self.fixed = (options, text, kind)
            if text in options and not any(option != text and option.startswith(text) for option in options):
                self._fixed_done(text, kind)
            return True
        if so_far in options:  # a number that is whole, and this character ends it
            self._fixed_done(so_far, kind)
            return self._char(ch)
        return False

    def _fixed_done(self, text: str, kind: str) -> None:
        self.fixed = None
        if kind == "key":
            self.last_key = json.loads(text)
            self.frames[-1]["used"].add(self.last_key)
            self.expect = "colon"
        else:
            self._value_done()

    def _enter(self, key: str | None = None) -> None:
        """Set what the next value must be, as the container that holds it says."""
        schema = self.frames[-1]["schema"]
        if self.stack[-1] == "[":
            self.want = schema.get("items", {})
        else:
            self.want = schema.get("properties", {}).get(key, {})

    def _char(self, ch: str) -> bool:
        if self.fixed is not None:
            return self._fixed_char(ch)
        if self.string is not None:
            return self._in_string(ch)
        if self.literal:
            self.literal += ch
            if not any(word.startswith(self.literal) for word in self.allowed):
                return False
            if self.literal in self.allowed:
                self.literal = ""
                self._value_done()
            return True
        if self.number:
            if ch in ("0123456789-" if self.integer else "0123456789.eE+-"):
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
                self.last_key = None
                self.expect = "colon"
            else:
                self._value_done()
        elif ord(ch) < 0x20:
            return False  # a control character must be escaped
        return True

    def _start_value(self, ch: str) -> bool:
        want, kinds = self.want, self._types()
        options = want["enum"] if "enum" in want else [want["const"]] if "const" in want else None
        if options is not None:
            self.fixed = ([json.dumps(option, ensure_ascii=False) for option in options], "", "value")
            return self._fixed_char(ch)
        if ch == "{" and "object" in kinds:
            self.stack.append("{")
            self.frames.append({"schema": want, "used": set(), "count": 0})
            self.expect = "key_or_end"
        elif ch == "[" and "array" in kinds:
            self.stack.append("[")
            self.frames.append({"schema": want, "used": set(), "count": 0})
            self.expect = "value_or_end"
            self._enter()
        elif ch == '"' and "string" in kinds:
            self.string = (False, False, 0)
        elif (ch == "-" or ch.isdigit()) and kinds & {"number", "integer"}:
            self.number = ch
            self.integer = "number" not in kinds
        elif ch in "tfn" and kinds & {"boolean", "null"}:
            self.allowed = tuple(
                word
                for word in _LITERALS
                if (word == "null" and "null" in kinds) or (word != "null" and "boolean" in kinds)
            )
            self.literal = ch
            if not any(word.startswith(ch) for word in self.allowed):
                return False
        else:
            return False
        return True

    def _object(self, ch: str) -> bool:  # the top level of a reply: an object, and only that
        return ch == "{" and self._start_value(ch)

    def _value(self, ch: str) -> bool:
        return self._start_value(ch)

    def _close(self) -> None:
        self.stack.pop()
        self.frames.pop()
        self.want = {}
        self._value_done()

    def _ends_array(self) -> bool:
        return self.frames[-1]["count"] >= self.frames[-1]["schema"].get("minItems", 0)

    def _array_room(self) -> bool:
        return self.frames[-1]["count"] < self.frames[-1]["schema"].get("maxItems", float("inf"))

    def _ends_object(self) -> bool:
        frame = self.frames[-1]
        return set(frame["schema"].get("required", ())) <= frame["used"]

    def _keys_left(self) -> list[str]:
        frame = self.frames[-1]
        return [name for name in frame["schema"].get("properties", {}) if name not in frame["used"]]

    def _value_or_end(self, ch: str) -> bool:
        if ch == "]":
            if not self._ends_array():
                return False
            self._close()
            return True
        return self._array_room() and self._start_value(ch)

    def _key_or_end(self, ch: str) -> bool:
        if ch == "}":
            if not self._ends_object():
                return False
            self._close()
            return True
        return self._key(ch)

    def _key(self, ch: str) -> bool:
        schema = self.frames[-1]["schema"]
        if "properties" in schema:  # only the keys it lists, each once
            names = self._keys_left()
            if ch != '"' or not names:
                return False
            self.fixed = ([json.dumps(name, ensure_ascii=False) for name in names], "", "key")
            return self._fixed_char(ch)
        if ch != '"':
            return False
        self.string = (True, False, 0)
        return True

    def _colon(self, ch: str) -> bool:
        if ch != ":":
            return False
        self.expect = "value"
        self._enter(self.last_key)
        return True

    def _comma_or_end(self, ch: str) -> bool:
        inside = self.stack[-1]
        if ch == ",":
            if inside == "{":
                if "properties" in self.frames[-1]["schema"] and not self._keys_left():
                    return False
                self.expect = "key"
            else:
                if not self._array_room():
                    return False
                self.expect = "value"
                self._enter()
        elif ch == "}" and inside == "{":
            if not self._ends_object():
                return False
            self._close()
        elif ch == "]" and inside == "[":
            if not self._ends_array():
                return False
            self._close()
        else:
            return False
        return True

    def _done(self, ch: str) -> bool:
        return False


class JsonConstraint:
    """The mask on a model's choices that keeps a reply a JSON object, one that fits `schema` if given."""

    def __init__(self, tokenizer, schema: dict | None = None):
        if schema is not None:
            check_schema(schema)
        self.tokenizer = tokenizer
        self.state = JsonPrefix(schema)
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
