"""Seeded sampling: the same seed draws the same tokens, and a receipt of drawn tokens is checked by drawing again."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from shem_neural.model import generate, receipt, sampling, serve
from shem_neural.model.blocks import Model
from shem_neural.model.sampling import Sampler

TINY = Path(__file__).parent / "sessions" / "model"


def draws(sampler, logits, count=40):
    return [sampler.draw(logits)[0] for _ in range(count)]


def test_the_same_seed_draws_the_same_tokens_and_another_draws_others():
    logits = torch.randn(50, generator=torch.Generator().manual_seed(1))
    one = draws(Sampler(1.0, seed=7), logits)
    assert one == draws(Sampler(1.0, seed=7), logits)
    assert one != draws(Sampler(1.0, seed=8), logits)
    assert len(set(one)) > 3, "a flat distribution is not drawn greedily"


def test_a_cold_distribution_is_nearly_greedy_and_top_k_and_top_p_cut_it():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert set(draws(Sampler(0.05, seed=1), logits)) == {0}
    assert set(draws(Sampler(5.0, top_k=2, seed=1), logits, 200)) == {0, 1}
    # Top-p keeps the smallest set that adds to it: here the first two, whose mass passes 0.7
    assert set(draws(Sampler(1.0, top_p=0.7, seed=1), logits, 200)) <= {0, 1}
    assert 0 in set(draws(Sampler(1.0, top_p=0.01, seed=1), logits)) == {0}


def test_a_margin_is_the_draws_distance_from_the_edge_of_its_share():
    logits = torch.tensor([0.0, 0.0])
    for seed in range(20):
        _, margin = Sampler(1.0, seed=seed).draw(logits)
        assert 0 <= margin <= 0.25


@pytest.mark.parametrize(
    "bad",
    [dict(temperature=-1), dict(temperature=1, top_p=0), dict(temperature=1, top_p=1.5), dict(temperature=1, top_k=-1)],
)
def test_settings_that_make_no_sense_are_refused(bad):
    with pytest.raises(ValueError):
        Sampler(**bad)


def test_the_options_need_a_temperature():
    ok = generate.options(["--temperature", "0.8", "--top-p", "0.9", "--top-k", "20", "--seed", "3", "model"])
    assert generate.sampler(ok).settings() == dict(temperature=0.8, top_p=0.9, top_k=20, seed=3)
    assert generate.sampler(generate.options(["model"])) is None
    with pytest.raises(ValueError, match="need a --temperature"):
        generate.sampler(generate.options(["--seed", "3", "model"]))


class Words:
    """A tokenizer of one token per character, as the tiny model's vocabulary allows."""

    eos: set = set()
    ROLES = {"system": 1, "user": 2, "assistant": 3}

    def chat(self, messages, tools=None):
        return self.prefix(messages) + [self.ROLES["assistant"]]

    def prefix(self, messages):
        return [t for m in messages for t in [self.ROLES[m["role"]]] + [4 + ord(c) % 60 for c in m["content"]]]

    def decode(self, ids):
        return "".join(chr(ord("a") + i % 26) for i in ids)

    def render(self, *_, **__):  # pragma: no cover
        raise NotImplementedError


def sampled_session(seed, temperature=1.5):
    session = generate.Session(Model(TINY))
    session.sampler = Sampler(temperature, seed=seed)
    session.choices = []
    session.feed([3, 17, 42])
    session.messages = [{"role": "user", "content": "x"}]
    return session


def test_a_sampled_receipt_is_checked_by_drawing_again():
    first = sampled_session(11)
    for _ in range(12):
        first.feed([first.choose()])
    first.messages.append({"role": "assistant", "content": Words().decode([t for t, _ in first.choices])})
    written = receipt.record(first, "x")
    assert written["sampling"]["seed"] == 11 and written["sampling"]["temperature"] == 1.5

    again = sampled_session(11)
    for _ in range(12):
        again.feed([again.choose()])
    again.messages.append({"role": "assistant", "content": written["reply"]})
    assert receipt.check(written, again)[0] == receipt.SAME

    other = sampled_session(12)
    for _ in range(12):
        other.feed([other.choose()])
    other.messages.append({"role": "assistant", "content": "z"})
    assert receipt.check(written, other)[0] in (receipt.DIFFERENT, receipt.UNDECIDED)


def test_greedy_receipts_have_no_sampling():
    session = generate.Session(Model(TINY))
    session.choices = []
    session.feed([3, 17])
    session.feed([session.choose()])
    assert "sampling" not in receipt.record(session, "x")
    assert receipt.threshold("float32", True) == sampling.CLOSE["float32"]
    assert receipt.threshold("float32+int8", False) == receipt.CLOSE["float32+int8"]
    assert receipt.threshold("unheard-of", True) == 0


def test_requests_draw_with_the_settings_they_give(monkeypatch):
    model = Model(TINY)
    args = SimpleNamespace(max_tokens=6, stop=[], system=None, temperature=0.0)
    chat = serve.Chat(model, Words(), args)
    messages = [{"role": "user", "content": "Hi"}]
    greedy = chat.complete(messages)
    again = chat.complete(messages)
    assert greedy["text"] == again["text"]
    one = chat.complete(messages, draw={"temperature": 2.0, "seed": 5})
    two = chat.complete(messages, draw={"temperature": 2.0, "seed": 5})
    assert one["text"] == two["text"], "the same seed, the same reply"
    texts = {chat.complete(messages, draw={"temperature": 2.0, "seed": s})["text"] for s in range(8)}
    assert len(texts) > 1
