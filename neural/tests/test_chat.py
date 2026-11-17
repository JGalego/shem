"""`--chat`: a conversation of turns read from input, one to a line."""

import io
from types import SimpleNamespace

import pytest

from shem_neural.model import generate


@pytest.fixture
def turns(monkeypatch):
    said = []
    monkeypatch.setattr(generate, "converse", lambda session, tokenizer, text, *_: said.append((session, text)))
    return said


ARGS = SimpleNamespace(max_tokens=4, stop=[])


def test_each_line_is_a_turn_of_one_conversation(turns):
    status = generate.chat("session", None, ARGS, lines=io.StringIO("Hi\n\n  Who are you?  \nlast"), err=io.StringIO())
    assert status == 0
    assert turns == [("session", "Hi"), ("session", "Who are you?"), ("session", "last")]


def test_a_prompt_starts_it_and_exit_ends_it(turns):
    generate.chat("s", None, ARGS, "First", io.StringIO("Second\n/exit\nNever\n"), err=io.StringIO())
    assert [text for _, text in turns] == ["First", "Second"]


def test_clear_starts_the_conversation_over(turns):
    generate.chat("old", None, ARGS, lines=io.StringIO("A\n/clear\nB\n"), err=io.StringIO(), restart=lambda: "new")
    assert turns == [("old", "A"), ("new", "B")]


def test_the_options_take_chat():
    assert generate.options(["--chat", "model"]).chat is True
    assert generate.options(["model", "Hi"]).chat is False
