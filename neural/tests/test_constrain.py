"""Replies that must be a JSON object: the prefix checker against Python's parser, and the mask on a model's choices."""

import json

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from shem_neural.model.constrain import JsonConstraint, JsonPrefix

values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=6), inner, max_size=4),
    max_leaves=15,
)


@settings(max_examples=300, deadline=None)
@given(st.dictionaries(st.text(max_size=6), values, max_size=4), st.booleans())
def test_every_prefix_of_a_json_object_is_one_and_the_whole_is_done(document, ugly):
    text = json.dumps(document, ensure_ascii=ugly, indent=2 if ugly else None)
    state = JsonPrefix()
    for i, char in enumerate(text):
        assert state.feed(char), (text, i)
    assert state.done


@settings(max_examples=400, deadline=None)
@given(st.text(alphabet='{}[]":,. -+0123456789eEtrufalsn\\u\n', max_size=30))
def test_it_accepts_a_text_as_a_whole_object_exactly_when_python_does(text):
    state = JsonPrefix()
    accepted = state.feed(text) and state.done
    try:
        parsed = json.loads(text)
        parses = isinstance(parsed, dict)
    except ValueError:
        parses = False
    assert accepted == parses, text


def test_what_cannot_become_an_object_is_refused_at_once():
    for text in (
        "[1]",
        '"a"',
        "1",
        "{1",
        '{"a" 1',
        '{"a":}',
        '{"a":1,}',
        '{"a":01}',
        '{"a":tru3}',
        '{"a":"\\x"}',
        '{"a":"\n"}',
    ):
        assert not JsonPrefix().feed(text), text
    assert JsonPrefix().feed('{"a": [1, 2.5e3, {"b": null}], "c": "\\u00e9"} ')


class Words:
    """A tokenizer of a small vocabulary of JSON's pieces, and a token that ends the reply."""

    pieces = [
        "{",
        "}",
        "[",
        "]",
        '"a"',
        '"b"',
        ":",
        ",",
        "1",
        "23",
        "-",
        ".",
        "true",
        "null",
        " ",
        "\n",
        "x",
        '"',
        "ab",
        '"k": ',
        "",
        "�",
    ]
    eos = {len(pieces)}

    def decode(self, ids):
        return "".join(self.pieces[i] if i < len(self.pieces) else "" for i in ids)


def test_masked_choices_always_make_a_reply_that_parses():
    tokenizer = Words()
    generator = torch.Generator().manual_seed(0)
    for _ in range(40):
        constraint, text, ended = JsonConstraint(tokenizer), "", False
        for _ in range(200):
            logits = torch.randn(len(Words.pieces) + 1, generator=generator) * 3
            token = int(constraint.mask(logits).argmax())
            if token in tokenizer.eos:
                ended = True
                break
            constraint.advance(token)
            text += tokenizer.decode([token])
        if ended:
            assert isinstance(json.loads(text), dict), text
        else:
            assert JsonPrefix().feed(text), text  # still on its way to an object


def test_once_the_object_is_closed_only_the_end_can_follow():
    tokenizer = Words()
    constraint = JsonConstraint(tokenizer)
    for piece in ("{", "}"):
        constraint.advance(Words.pieces.index(piece))
    masked = constraint.mask(torch.zeros(len(Words.pieces) + 1))
    assert int(masked.argmax()) in tokenizer.eos and (masked > float("-inf")).sum() == 1


# ---- a JSON schema ----------------------------------------------------------


def fits(value, schema) -> bool:
    """Whether `value` is one `schema` allows: the subset shem enforces, written out the plain way."""
    if "enum" in schema:
        return any(value == option and type(value) is type(option) for option in schema["enum"])
    if "const" in schema:
        return value == schema["const"] and type(value) is type(schema["const"])
    kinds = schema.get("type")
    kinds = None if kinds is None else [kinds] if isinstance(kinds, str) else kinds
    if kinds is not None:
        ok = {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
            "null": value is None,
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, int | float) and not isinstance(value, bool),
        }
        if not any(ok[kind] for kind in kinds):
            return False
    if isinstance(value, dict):
        properties = schema.get("properties")
        if properties is not None and not set(value) <= set(properties):
            return False
        if not set(schema.get("required", ())) <= set(value):
            return False
        return all(fits(item, (properties or {}).get(key, {})) for key, item in value.items())
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")):
            return False
        return all(fits(item, schema.get("items", {})) for item in value)
    return True


scalars = st.one_of(
    st.sampled_from(["string", "integer", "number", "boolean", "null"]).map(lambda kind: {"type": kind}),
    st.lists(st.sampled_from(["string", "integer", "boolean", "null"]), min_size=1, max_size=3, unique=True).map(
        lambda kinds: {"type": kinds}
    ),
    st.lists(
        st.text(alphabet="abc", max_size=3) | st.integers(-5, 5) | st.booleans() | st.none(), min_size=1, max_size=3
    ).map(lambda options: {"enum": options}),
)
names = st.text(alphabet="abc", min_size=1, max_size=3)


@st.composite
def object_schemas(draw, depth=2):
    inner = scalars if depth == 0 else st.one_of(scalars, object_schemas(depth - 1), arrays(depth - 1))
    properties = draw(st.dictionaries(names, inner, max_size=3))
    required = draw(st.lists(st.sampled_from(sorted(properties)), unique=True)) if properties else []
    schema = {"type": "object"}
    if properties or draw(st.booleans()):
        schema["properties"] = properties
    if required:
        schema["required"] = required
    return schema


@st.composite
def arrays(draw, depth=1):
    inner = scalars if depth == 0 else st.one_of(scalars, object_schemas(depth - 1))
    low = draw(st.integers(0, 2))
    schema = {"type": "array", "items": draw(inner), "minItems": low}
    if draw(st.booleans()):
        schema["maxItems"] = low + draw(st.integers(0, 2))
    return schema


@st.composite
def instances(draw, schema):
    """A value that fits `schema`."""
    if "enum" in schema:
        return draw(st.sampled_from(schema["enum"]))
    kind = schema.get("type")
    kind = draw(st.sampled_from(kind)) if isinstance(kind, list) else kind
    if kind == "object":
        properties = schema.get("properties")
        if properties is None:
            return draw(st.dictionaries(names, st.integers(), max_size=2))
        required = set(schema.get("required", ()))
        chosen = [key for key in properties if key in required or draw(st.booleans())]
        return {key: draw(instances(properties[key])) for key in draw(st.permutations(chosen))}
    if kind == "array":
        low, high = schema.get("minItems", 0), schema.get("maxItems", 4)
        return draw(st.lists(instances(schema["items"]), min_size=low, max_size=high))
    return draw(
        {
            "string": st.text(max_size=5),
            "integer": st.integers(-1000, 1000),
            "number": st.integers(-1000, 1000) | st.floats(-1e3, 1e3, allow_nan=False),
            "boolean": st.booleans(),
            "null": st.none(),
        }[kind]
    )


@settings(max_examples=300, deadline=None)
@given(st.data(), st.booleans())
def test_every_prefix_of_an_object_that_fits_the_schema_is_one_and_the_whole_is_done(data, spaced):
    schema = data.draw(object_schemas())
    check = JsonConstraint(Words(), schema)  # the schema is one that is understood
    value = data.draw(instances(schema))
    assert fits(value, schema)
    text = json.dumps(value, separators=(", ", ": ") if spaced else (",", ":"), ensure_ascii=False)
    state = JsonPrefix(schema)
    for i, char in enumerate(text):
        assert state.feed(char), (schema, text, i)
    assert state.done and check


@settings(max_examples=300, deadline=None)
@given(st.data())
def test_an_object_that_does_not_fit_the_schema_is_refused_before_it_is_whole(data):
    schema = data.draw(object_schemas())
    text = json.dumps(data.draw(st.dictionaries(names, values, max_size=3)))
    state = JsonPrefix(schema)
    accepted = state.feed(text) and state.done
    assert accepted == fits(json.loads(text), schema), (schema, text)


class Letters:
    """A tokenizer of single characters, which is what a mask has most to do."""

    pieces = list('{}[]":,. -0123456789abcdefghijklmnopqrstuvwxyz') + ["true", "false", "null", '{"', '":']
    eos = {len(pieces)}

    def decode(self, ids):
        return "".join(self.pieces[i] if i < len(self.pieces) else "" for i in ids)


@settings(max_examples=60, deadline=None)
@given(object_schemas(), st.integers(0, 10_000))
def test_masked_choices_make_a_reply_that_fits_the_schema(schema, seed):
    tokenizer = Letters()
    generator = torch.Generator().manual_seed(seed)
    constraint, text, ended = JsonConstraint(tokenizer, schema), "", False
    for _ in range(150):
        logits = torch.randn(len(Letters.pieces) + 1, generator=generator) * 3
        token = int(constraint.mask(logits).argmax())
        if token in tokenizer.eos:
            ended = True
            break
        constraint.advance(token)
        text += tokenizer.decode([token])
    if ended:
        assert fits(json.loads(text), schema), (schema, text)
    else:
        assert JsonPrefix(schema).feed(text), (schema, text)


def test_what_a_schema_does_not_enforce_is_refused_and_not_ignored():
    from shem_neural.model.constrain import check_schema

    for bad in (
        {"type": "array"},  # a reply is an object
        {"type": "object", "pattern": "a"},
        {"type": "object", "anyOf": []},
        {"type": "object", "properties": {"a": {"type": "string", "minLength": 1}}},
        {"type": "object", "properties": {"a": {"enum": [[1]]}}},
        {"type": "object", "additionalProperties": True},
        {"type": "object", "required": ["a"]},
        {"type": "object", "properties": {"a": {"type": "text"}}},
        {"type": "object", "properties": {"a": {"type": "array", "minItems": -1}}},
        [],
    ):
        try:
            check_schema(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    check_schema(
        {"type": "object", "properties": {"a": {"type": ["string", "null"], "description": "x"}}, "required": ["a"]}
    )


def test_a_schema_decides_the_keys_their_order_of_use_and_what_ends_the_object():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"enum": ["a", "b"]}},
        },
        "required": ["name"],
    }

    def ok(text):
        state = JsonPrefix(schema)
        return state.feed(text) and state.done

    assert ok('{"name": "x"}') and ok('{"age": 3, "name": "x", "tags": ["a", "b"]}')
    assert not ok("{}")  # name is required
    assert not ok('{"name": "x", "name": "y"}')  # each key once
    assert not ok('{"name": "x", "extra": 1}') and not ok('{"name": 1}')
    assert not ok('{"name": "x", "age": 1.5}') and not ok('{"name": "x", "tags": ["c"]}')
