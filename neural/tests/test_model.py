"""Imported models against transformers' own implementations.

These download checkpoints from Hugging Face, so they run only with
SHEM_MODEL_TESTS=1. Gated checkpoints, such as Gemma's, are skipped without
access to them. Each test holds one model in memory at a time, so a 1B
model's comparison fits in 16 GB.
"""

import gc
import json
import os
from datetime import datetime
from pathlib import Path

import pytest
import torch

from shem_neural.model.blocks import Model, rope_frequencies
from shem_neural.model.importers import (
    ACTIVATIONS,
    FAMILIES,
    activation,
    import_model,
    layer_types,
    rope,
    sliding_window,
)
from shem_neural.model.manifest import Manifest
from shem_neural.model.tokenizer import Tokenizer

pytestmark = pytest.mark.skipif(not os.environ.get("SHEM_MODEL_TESTS"), reason="set SHEM_MODEL_TESTS=1")

REPOS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "google/gemma-3-270m-it",
    "allenai/OLMo-2-0425-1B-Instruct",
    "ibm-granite/granite-3.1-1b-a400m-instruct",
    "state-spaces/mamba-130m-hf",
]
#: The imports that are a text model's whole checkpoint: Qwen3.5's leaves out its vision encoder
TEXT_REPOS = [repo for repo in REPOS if "3.5" not in repo]
#: The token that ends a turn in each chat template, where it isn't ChatML's
END_OF_TURN = {
    "google/gemma-3-270m-it": "<end_of_turn>",
    "allenai/OLMo-2-0425-1B-Instruct": "<|endoftext|>",
    "ibm-granite/granite-3.1-1b-a400m-instruct": "<|end_of_text|>",
    "state-spaces/mamba-130m-hf": "<|endoftext|>",
}
#: Base models, without a chat template: shem gives them one that runs
#: their messages' text together, and transformers has none to compare
BASE = {"state-spaces/mamba-130m-hf"}
#: Models too small to recall a name from two turns back: transformers,
#: given the same tokens, forgets it the same way
FORGETFUL = {"google/gemma-3-270m-it"} | BASE
_IMPORTED: dict = {}


def gated(call):
    """`call()`, or a skip if the checkpoint is gated and this machine has no access to it."""
    from huggingface_hub.errors import GatedRepoError

    try:
        return call()
    except GatedRepoError as error:
        pytest.skip(f"no access to a gated checkpoint: {error}")


@pytest.fixture(scope="module")
def imports(tmp_path_factory):
    """Each model imported once for the whole module: an import is a gigabyte."""

    def get(repo):
        if repo not in _IMPORTED:
            out = tmp_path_factory.mktemp(repo.replace("/", "-"))
            gated(lambda: import_model(repo, out))
            _IMPORTED[repo] = out
        return _IMPORTED[repo]

    return get


MESSAGES = [{"role": "user", "content": "Name three uses of clay."}]


def reference(repo: str):
    import transformers

    model = transformers.AutoModelForImageTextToText if "3.5" in repo else transformers.AutoModelForCausalLM
    return model.from_pretrained(repo, dtype=torch.float32).eval()


def free(*models) -> None:
    """Let go of a model before the next is loaded."""
    del models
    gc.collect()


@pytest.mark.parametrize("repo", REPOS)
def test_imported_models_match_transformers(repo, imports):
    import transformers

    directory = imports(repo)
    tokenizer = Tokenizer(directory, Manifest.load(directory).tokenizer)
    prompt = tokenizer.chat(MESSAGES)
    tokens = torch.tensor([prompt])

    # transformers' logits for the prompt, and its greedy reply. The
    # checkpoint's generation defaults are replaced, not overridden, since
    # generate merges them into a config it is given: Qwen2.5's repetition
    # penalty would apply even without sampling
    ref = reference(repo)
    with torch.no_grad():
        expected = ref(input_ids=tokens).logits
    ref.generation_config = transformers.GenerationConfig()
    want = ref.generate(input_ids=tokens, do_sample=False, max_new_tokens=24)[0, len(prompt) :].tolist()
    free(ref)
    ref = None

    # The prompt's logits, in one pass
    model = Model(directory)
    got = model.forward(tokens, model.init_state(), 0)
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3), (got - expected).abs().max()

    # Greedy decoding, one token at a time from the session's state
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

    tokens = torch.arange(100, 140)[None, :]
    loaded = transformers.AutoModelForCausalLM.from_pretrained(imports(repo), dtype=torch.float32).eval()
    with torch.no_grad():
        got = loaded(input_ids=tokens).logits
    free(loaded)
    loaded = None
    with torch.no_grad():
        assert torch.equal(got, reference(repo)(input_ids=tokens).logits)


@pytest.mark.parametrize("repo", REPOS)
def test_chat_templates_render_as_in_transformers(repo, imports, monkeypatch):
    """A conversation in the chat template, as shem renders it and as
    transformers does, special tokens included."""
    import transformers

    if repo in BASE:
        pytest.skip("a base model has no chat template to compare")
    tokenizer = Tokenizer(imports(repo), Manifest.load(imports(repo)).tokenizer)
    theirs = transformers.AutoTokenizer.from_pretrained(repo)
    system = [{"role": "system", "content": "You are a golem."}]
    # shem's templates see the guest's virtual clock, so a session renders the
    # same any day: transformers' templates get the same day
    from transformers.utils import chat_template_utils

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(chat_template_utils, "datetime", Clock)
    for messages in (MESSAGES, system + MESSAGES):
        want = theirs.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert tokenizer.render(messages) == want


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
    end_of_turn = tokenizer.inner.token_to_id(END_OF_TURN.get(repo, "<|im_end|>"))
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
    if repo not in FORGETFUL:
        assert "Joana" in want[1], want


#: Checkpoints whose configs are read layer by layer, as transformers reads
#: them: their configs alone are downloaded, so the list can include models
#: too large to compare here, and configs in older formats
CONFIG_REPOS = [
    "unsloth/Llama-3.2-1B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "google/gemma-2-2b-it",
    "google/gemma-3-270m-it",
    "google/gemma-3-4b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "allenai/OLMoE-1B-7B-0125-Instruct",
    "ibm-granite/granite-3.3-2b-instruct",
    "ibm-granite/granite-3.1-1b-a400m-instruct",
]


@pytest.mark.parametrize("repo", CONFIG_REPOS)
def test_configs_read_as_transformers_reads_them(repo):
    """Each layer's kind, window and rotary frequencies, from the config as
    shem reads it, against transformers' reading of the same config."""
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    raw = json.loads(Path(gated(lambda: hf_hub_download(repo, "config.json"))).read_text())
    cfg = FAMILIES[raw["architectures"][0]].config(raw)
    text = transformers.AutoConfig.from_pretrained(repo).get_text_config()

    heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    assert head_dim == (getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads)
    assert (heads, cfg["num_key_value_heads"]) == (text.num_attention_heads, text.num_key_value_heads)
    assert cfg["rms_norm_eps"] == text.rms_norm_eps
    assert activation(cfg) == ACTIVATIONS[getattr(text, "hidden_activation", None) or text.hidden_act]
    multipliers = ("attention_multiplier", "residual_multiplier", "embedding_multiplier", "logits_scaling")
    for key in ("query_pre_attn_scalar", "attn_logit_softcapping", "final_logit_softcapping", *multipliers):
        assert cfg.get(key) == getattr(text, key, None), key

    kinds = getattr(text, "layer_types", None)
    per_kind = all(isinstance(value, dict) for value in text.rope_parameters.values())
    for i, kind in enumerate(layer_types(cfg)):
        if kinds:
            assert kind == kinds[i], i
            window = text.sliding_window if kinds[i] == "sliding_attention" else None
        else:
            window = getattr(text, "sliding_window", None)
        assert (sliding_window(cfg) if kind == "sliding_attention" else None) == window, i
        if kind == "linear_attention":
            continue
        if cfg.get("no_rope_layers"):
            assert cfg["no_rope_layers"][i] == text.no_rope_layers[i], i
            if not text.no_rope_layers[i]:
                continue

        params = {"head_dim": head_dim, **rope(cfg, head_dim, kind)}
        layer_type = kind if per_kind else None
        theirs = text.rope_parameters[kind] if per_kind else text.rope_parameters
        context = theirs.get("original_max_position_embeddings") or 0
        # A pass within the original context, and one past it, which longrope tells apart
        for end in (1, context + 1):
            got, got_scale = rope_frequencies(params, torch.device("cpu"), end)
            if theirs["rope_type"] == "default":
                dim = int(head_dim * theirs.get("partial_rotary_factor", 1.0))
                want = 1.0 / (theirs["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
                want_scale = 1.0
            else:
                function = ROPE_INIT_FUNCTIONS[theirs["rope_type"]]
                want, want_scale = function(text, None, seq_len=end, layer_type=layer_type)
            assert torch.allclose(got, want, rtol=1e-6, atol=0), (i, kind, end)
            assert got_scale == pytest.approx(want_scale), (i, kind, end)


@pytest.mark.parametrize("repo, quantization", [("Qwen/Qwen3-0.6B-GGUF", "Q8_0")])
def test_gguf_imports_match_transformers(repo, quantization, tmp_path):
    """A GGUF file imports as transformers dequantizes it, with its tokenizer
    and chat template, and replies as transformers does."""
    import transformers

    gguf = f"{repo.split('/')[1].removesuffix('-GGUF')}-{quantization}.gguf"
    gated(lambda: import_model(repo, tmp_path, quantization))
    tokenizer = Tokenizer(tmp_path, Manifest.load(tmp_path).tokenizer)
    theirs = transformers.AutoTokenizer.from_pretrained(repo, gguf_file=gguf)
    want = theirs.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert tokenizer.render(MESSAGES) == want
    tokens = torch.tensor([tokenizer.chat(MESSAGES)])
    ref = transformers.AutoModelForCausalLM.from_pretrained(repo, gguf_file=gguf, dtype=torch.float32).eval()
    with torch.no_grad():
        expected = ref(input_ids=tokens).logits
    free(ref)
    ref = None
    model = Model(tmp_path)
    got = model.forward(tokens, model.init_state(), 0)
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3), (got - expected).abs().max()


#: Checkpoints too large to compare in float32 on 16 GB, compared in bfloat16
#: instead, a model at a time: Gemma 2's softcaps, Phi-4-mini's fused
#: projections and longrope, SmolLM3's layers without rotary positions,
#: and dense Granite
BF16_REPOS = [
    "google/gemma-2-2b-it",
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "ibm-granite/granite-3.3-2b-instruct",
]


@pytest.mark.skipif(
    not os.environ.get("SHEM_LARGE_MODEL_TESTS"), reason="20 GB of checkpoints: set SHEM_LARGE_MODEL_TESTS=1"
)
@pytest.mark.parametrize("repo", BF16_REPOS)
def test_large_models_agree_with_transformers_in_bfloat16(repo, imports):
    """In bfloat16 the two can't agree to the last bit, as they round in
    different places, but a block read wrong is far off: the next token they
    predict must agree almost everywhere along a prompt, and their last
    logits must point the same way."""
    import transformers

    directory = imports(repo)
    tokenizer = Tokenizer(directory, Manifest.load(directory).tokenizer)
    tokens = torch.tensor(
        [tokenizer.chat(MESSAGES + [{"role": "assistant", "content": "Bricks, pottery and sculpture."}])]
    )
    ref = transformers.AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        expected = ref(input_ids=tokens).logits[0].float()
    free(ref)
    ref = None
    model = Model(directory, dtype=torch.bfloat16)
    got = model.forward(tokens, model.init_state(), 0)[0].float()
    agree = (got.argmax(-1) == expected.argmax(-1)).float().mean().item()
    assert agree >= 0.9, agree
    assert torch.nn.functional.cosine_similarity(got[-1], expected[-1], dim=0) > 0.99
