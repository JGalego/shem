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

REPOS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
#: The imports that are a text model's whole checkpoint: Qwen3.5's leaves out its vision encoder
TEXT_REPOS = [repo for repo in REPOS if "3.5" not in repo]
_IMPORTED: dict = {}


@pytest.fixture(scope="module")
def imports(tmp_path_factory):
    """Each model imported once for the whole module: an import is a gigabyte."""

    def get(repo):
        if repo not in _IMPORTED:
            out = tmp_path_factory.mktemp(repo.replace("/", "-"))
            import_model(repo, out)
            _IMPORTED[repo] = out
        return _IMPORTED[repo]

    return get


MESSAGES = [{"role": "user", "content": "Name three uses of clay."}]


def reference(repo: str):
    import transformers

    model = transformers.AutoModelForImageTextToText if "3.5" in repo else transformers.AutoModelForCausalLM
    return model.from_pretrained(repo, dtype=torch.float32).eval()


@pytest.mark.parametrize("repo", REPOS)
def test_imported_models_match_transformers(repo, imports):
    directory = imports(repo)
    model = Model(directory)
    tokenizer = Tokenizer(directory, model.manifest.tokenizer)
    prompt = tokenizer.chat(MESSAGES)
    ref = reference(repo)

    # The prompt's logits, in one pass
    tokens = torch.tensor([prompt])
    expected = ref(input_ids=tokens).logits
    got = model.forward(tokens, model.init_state(), 0)
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3), (got - expected).abs().max()

    # Greedy decoding, one token at a time from the session's state. The
    # checkpoint's generation defaults are replaced, not overridden, since
    # generate merges them into a config it is given: Qwen2.5's repetition
    # penalty would apply even without sampling
    import transformers

    ref.generation_config = transformers.GenerationConfig()
    want = ref.generate(input_ids=tokens, do_sample=False, max_new_tokens=24)[0, len(prompt) :].tolist()
    state = model.init_state()
    logits = model.forward(tokens, state, 0)
    reply = []
    for position in range(len(prompt), len(prompt) + len(want)):
        token = int(logits[0, -1].argmax())
        reply.append(token)
        logits = model.forward(torch.tensor([[token]]), state, position)
    assert reply == want, (tokenizer.decode(reply), tokenizer.decode(want))


@pytest.mark.parametrize("repo", TEXT_REPOS)
def test_imported_directories_load_in_transformers(repo, imports):
    """An import is the checkpoint sharded by block, with its own tensor
    names: transformers loads it and gives the original's logits."""
    import transformers

    loaded = transformers.AutoModelForCausalLM.from_pretrained(imports(repo), dtype=torch.float32).eval()
    tokens = torch.arange(100, 140)[None, :]
    with torch.no_grad():
        assert torch.equal(loaded(input_ids=tokens).logits, reference(repo)(input_ids=tokens).logits)


@pytest.mark.parametrize("repo", REPOS)
def test_saved_sessions_continue_as_if_never_stopped(repo, imports, tmp_path):
    from shem_neural.model.generate import Session, reply

    model = Model(imports(repo))
    tokenizer = Tokenizer(imports(repo), model.manifest.tokenizer)
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


@pytest.mark.parametrize("repo", REPOS)
def test_conversations_take_turns_and_survive_a_save(repo, imports):
    import io

    from shem_neural.model.generate import Session, converse

    model = Model(imports(repo))
    tokenizer = Tokenizer(imports(repo), model.manifest.tokenizer)
    end_of_turn = tokenizer.inner.token_to_id("<|im_end|>")
    assert end_of_turn in tokenizer.eos, "replies stop at the end of the turn"

    def turn(session, text):
        out = io.StringIO()
        converse(session, tokenizer, text, 24, out)
        return out.getvalue()

    turns = ["My name is Joana.", "What did I just tell you my name is?", "Say it once more."]
    whole = Session(model)
    whole.messages = [{"role": "system", "content": "Answer in one short sentence."}]
    want = [turn(whole, text) for text in turns]

    saved = Session(model)
    saved.messages = list(whole.messages[:1])
    got = [turn(saved, turns[0])]
    for text in turns[1:]:
        saved = Session.from_files(model, saved.to_files())
        got.append(turn(saved, text))
    assert got == want
    assert "Joana" in want[1], want
