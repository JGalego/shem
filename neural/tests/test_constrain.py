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
