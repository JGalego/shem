"""Imported models against transformers' own implementations.

These download checkpoints from Hugging Face, so they run only with
SHEM_MODEL_TESTS=1.
"""

import os

import pytest
import torch

from shem_neural.model.blocks import Model
from shem_neural.model.importers import import_model
from shem_neural.model.tokenizer import Tokenizer

pytestmark = pytest.mark.skipif(not os.environ.get("SHEM_MODEL_TESTS"), reason="set SHEM_MODEL_TESTS=1")

REPOS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"]
MESSAGES = [{"role": "user", "content": "Name three uses of clay."}]


def reference(repo: str):
    import transformers

    model = transformers.AutoModelForImageTextToText if "3.5" in repo else transformers.AutoModelForCausalLM
    return model.from_pretrained(repo, dtype=torch.float32).eval()


@pytest.mark.parametrize("repo", REPOS)
def test_imported_models_match_transformers(repo, tmp_path):
    import_model(repo, tmp_path)
    model = Model(tmp_path)
    tokenizer = Tokenizer(tmp_path, model.manifest.tokenizer)
    prompt = tokenizer.chat(MESSAGES)
    ref = reference(repo)

    # The prompt's logits, in one pass
    tokens = torch.tensor([prompt])
    expected = ref(input_ids=tokens).logits
    got = model.forward(tokens, model.init_state(), 0)
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3), (got - expected).abs().max()

    # Greedy decoding, one token at a time from the session's state
    want = ref.generate(input_ids=tokens, max_new_tokens=24, do_sample=False)[0, len(prompt) :].tolist()
    state = model.init_state()
    logits = model.forward(tokens, state, 0)
    reply = []
    for position in range(len(prompt), len(prompt) + len(want)):
        token = int(logits[0, -1].argmax())
        reply.append(token)
        logits = model.forward(torch.tensor([[token]]), state, position)
    assert reply == want, (tokenizer.decode(reply), tokenizer.decode(want))


@pytest.mark.parametrize("repo", REPOS)
def test_saved_sessions_continue_as_if_never_stopped(repo, tmp_path):
    from shem_neural.model.generate import Session, reply

    import_model(repo, tmp_path / "model")
    model = Model(tmp_path / "model")
    tokenizer = Tokenizer(tmp_path / "model", model.manifest.tokenizer)
    system = [{"role": "system", "content": "You are a golem. Answer in one sentence."}]
    conversation = tokenizer.chat(system + MESSAGES)

    fresh = Session(model)
    fresh.feed(conversation)
    want = reply(fresh, tokenizer, 16, out=open(os.devnull, "w"))

    primed = Session(model)
    primed.feed(tokenizer.prefix(system))
    primed.save(tmp_path / "session")
    resumed = Session.load(model, tmp_path / "session")
    assert conversation[: len(resumed.tokens)] == resumed.tokens
    resumed.feed(conversation[len(resumed.tokens) :])
    assert reply(resumed, tokenizer, 16, out=open(os.devnull, "w")) == want
