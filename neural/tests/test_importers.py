"""Importers against transformers, on tiny random checkpoints.

Each family is built small and random by transformers itself, so these need
no downloads: the imported model must give transformers' logits, in one pass
and token by token from its session state, and its directory must load in
transformers as the checkpoint it came from. `test_model` does the same with
real checkpoints.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from shem_neural.model.blocks import Model
from shem_neural.model.importers import import_checkpoint
from shem_neural.model.manifest import FILE, V1, Manifest

transformers = pytest.importorskip("transformers")

SMALL = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
)
GDN = dict(linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8)
LLAMA3_ROPE = {
    "rope_type": "llama3",
    "rope_theta": 10000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    # Short enough that the frequencies fall in all three of llama3's bands
    "original_max_position_embeddings": 32,
}


#: Token ids inside the tiny vocabulary, for families whose defaults are outside it
TOKENS = dict(pad_token_id=0, bos_token_id=1, eos_token_id=2)
SLIDING_FULL = ["sliding_attention", "full_attention"]


def qwen3_5_vlm():
    """Qwen3.5's vision-language model, whose language model shem imports."""
    text = SMALL | GDN | dict(head_dim=8, layer_types=["linear_attention", "full_attention"])
    vision = dict(depth=1, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=32)
    return transformers.Qwen3_5Config(text_config=text, vision_config=vision)


def gemma3_vlm():
    """Gemma 3's vision-language model, whose language model shem imports."""
    text = SMALL | dict(head_dim=8, sliding_window=5, layer_types=SLIDING_FULL)
    vision = dict(
        hidden_size=16, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14
    )
    return transformers.Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=4)


def as_on_the_hub(checkpoint: Path) -> None:
    """Rewrite a Gemma 3 vision-language checkpoint as the Hub has them:
    tensors under the names transformers used before version 5, and a text
    config that leaves out whatever is the default."""
    renames = {"model.language_model.": "language_model.model.", "model.vision_tower.": "vision_tower."}
    renames["model.multi_modal_projector."] = "multi_modal_projector."
    tensors = load_file(str(checkpoint / "model.safetensors"))
    renamed = {}
    for name, tensor in tensors.items():
        for new, old in renames.items():
            name = name.replace(new, old, 1) if name.startswith(new) else name
        renamed[name] = tensor
    save_file(renamed, str(checkpoint / "model.safetensors"), metadata={"format": "pt"})
    config = json.loads((checkpoint / "config.json").read_text())
    defaults = transformers.Gemma3TextConfig().to_dict()
    config["text_config"] = {k: v for k, v in config["text_config"].items() if defaults.get(k) != v}
    (checkpoint / "config.json").write_text(json.dumps(config))


CONFIGS = {
    # Tied embeddings and Llama 3's rotary scaling
    "llama3": lambda: transformers.LlamaConfig(
        **SMALL, head_dim=8, tie_word_embeddings=True, rope_parameters=LLAMA3_ROPE
    ),
    # Biases on every projection, and linear rotary scaling
    "llama-biases": lambda: transformers.LlamaConfig(
        **SMALL,
        attention_bias=True,
        mlp_bias=True,
        rope_parameters={"rope_type": "linear", "rope_theta": 10000.0, "factor": 4.0},
    ),
    # A window shorter than the test's sequence
    "mistral": lambda: transformers.MistralConfig(**SMALL, sliding_window=5),
    # Biases on queries, keys and values
    "qwen2": lambda: transformers.Qwen2Config(**SMALL),
    # Norms on queries and keys
    "qwen3": lambda: transformers.Qwen3Config(**SMALL, head_dim=8),
    # Gated DeltaNet, gated attention on part of each head, zero-centred norms
    "qwen3.5": lambda: transformers.Qwen3_5TextConfig(
        **SMALL, **GDN, head_dim=8, layer_types=["linear_attention", "full_attention"]
    ),
    "qwen3.5-vlm": qwen3_5_vlm,
    # Norms around every block, GeGLU, scaled embeddings, a query scale of its
    # own, and softcaps small enough to bite
    "gemma2": lambda: transformers.Gemma2Config(
        **SMALL,
        head_dim=8,
        sliding_window=5,
        query_pre_attn_scalar=16,
        attn_logit_softcapping=0.5,
        final_logit_softcapping=1.0,
    ),
    # Norms on queries and keys, and rotary bases and scaling per kind of layer
    "gemma3": lambda: transformers.Gemma3TextConfig(
        **SMALL,
        head_dim=8,
        sliding_window=5,
        query_pre_attn_scalar=16,
        layer_types=SLIDING_FULL,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "linear", "factor": 4.0, "rope_theta": 1e6},
        },
    ),
    "gemma3-vlm": gemma3_vlm,
    # Fused projections, and longrope on part of each head, whose context the
    # test's sequence outgrows
    "phi3": lambda: transformers.Phi3Config(
        **SMALL,
        **TOKENS,
        original_max_position_embeddings=8,
        rope_parameters={
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "short_factor": [1.0, 1.5],
            "long_factor": [3.0, 7.0],
            "original_max_position_embeddings": 8,
        },
    ),
    # Norms after blocks, and on all of the queries and keys at once
    "olmo2": lambda: transformers.Olmo2Config(**SMALL),
    # YaRN in the full-attention layer, and a sliding window in the other
    "olmo3": lambda: transformers.Olmo3Config(
        **SMALL,
        sliding_window=5,
        layer_types=SLIDING_FULL,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 10000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 16,
            },
        },
    ),
    # No rotary positions in the second layer
    "smollm3": lambda: transformers.SmolLM3Config(**SMALL, **TOKENS, no_rope_layers=[1, 0]),
}
#: Vision-language models, whose imports leave the vision encoder out
VLMS = {
    "qwen3.5-vlm": transformers.Qwen3_5ForConditionalGeneration,
    "gemma3-vlm": transformers.Gemma3ForConditionalGeneration,
}
#: Families whose positions change with the length of the pass, so that a
#: session's steps match transformers' steps, not its single pass
STEPPED = {"phi3"}


def tokenizer_files(directory: Path, vocab_size: int) -> None:
    """A word-level tokenizer and a chat template, which an import needs."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(vocab_size)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    template = "{{ bos_token }} {% for m in messages %}{{ m.content }} {{ eos_token }} {% endfor %}"
    settings = {"chat_template": template, "bos_token": "t3", "eos_token": {"content": "t1"}}
    (directory / "tokenizer_config.json").write_text(json.dumps(settings))


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def imported(request, tmp_path_factory):
    """A tiny random checkpoint of one family, the reference model built from
    it, and the checkpoint imported."""
    torch.manual_seed(0)
    config = CONFIGS[request.param]()
    if request.param in VLMS:
        reference = VLMS[request.param](config).eval()
    else:
        reference = transformers.AutoModelForCausalLM.from_config(config).eval()
    # transformers starts biases at zero and norms at one, which would hide a
    # missing bias or a swapped norm: every tensor gets noise of its own
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    checkpoint = tmp_path_factory.mktemp("checkpoint")
    reference.save_pretrained(checkpoint)
    if request.param == "gemma3-vlm":
        as_on_the_hub(checkpoint)
    tokenizer_files(checkpoint, SMALL["vocab_size"])
    out = tmp_path_factory.mktemp("imported")
    import_checkpoint(checkpoint, out)
    return request.param, reference, out


def logits(reference, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return reference(input_ids=tokens).logits


def stepped(reference, tokens: torch.Tensor, split: int) -> torch.Tensor:
    """transformers' logits for `tokens`, `split` of them in one pass and
    then one at a time from its cache, as a session reads them."""
    with torch.no_grad():
        out = reference(input_ids=tokens[:, :split], use_cache=True)
        steps = [out.logits]
        for position in range(split, tokens.shape[1]):
            out = reference(input_ids=tokens[:, position : position + 1], past_key_values=out.past_key_values)
            steps.append(out.logits)
    return torch.cat(steps, dim=1)


def test_imported_models_match_transformers(imported):
    name, reference, out = imported
    tokens = torch.randint(0, SMALL["vocab_size"], (1, 12), generator=torch.Generator().manual_seed(1))
    expected = logits(reference, tokens)
    model = Model(out)

    # The whole sequence in one pass
    got = model.forward(tokens, model.init_state(), 0)
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()

    # Part of it, then a token at a time from the session's state
    state = model.init_state()
    got = [model.forward(tokens[:, :7], state, 0)]
    for position in range(7, tokens.shape[1]):
        got.append(model.forward(tokens[:, position : position + 1], state, position))
    got = torch.cat(got, dim=1)
    if name in STEPPED:
        expected = stepped(reference, tokens, 7)
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()


def test_imported_directories_are_checkpoints(imported):
    """transformers loads the imported directory as it does the original:
    the same tensors, under the same names, found through the index."""
    name, reference, out = imported
    if name in VLMS:
        pytest.skip("the import leaves out the vision encoder, so it is a text model's checkpoint")
    loaded = transformers.AutoModelForCausalLM.from_pretrained(out).eval()
    tokens = torch.arange(10)[None, :] % SMALL["vocab_size"]
    assert torch.equal(logits(loaded, tokens), logits(reference, tokens))


def test_each_block_is_a_file_of_its_own_tensors(imported):
    _, _, out = imported
    manifest = Manifest.load(out)
    weight_map = manifest.weight_map(out)
    for i, block in enumerate(manifest.blocks):
        files = {weight_map[name] for name in block.tensors.values()}
        assert files == {f"blocks/{i:03d}-{block.type}.safetensors"}
        with safe_open(str(out / files.pop()), "pt") as handle:
            assert set(handle.keys()) == set(block.tensors.values())


def test_templates_see_the_special_tokens(imported):
    """Chat templates get the tokenizer's special tokens, as transformers
    gives them: Gemma's starts with `bos_token`, OLMo 2's ends turns with
    `eos_token`."""
    from shem_neural.model.tokenizer import Tokenizer

    _, _, out = imported
    tokenizer = Tokenizer(out, Manifest.load(out).tokenizer)
    assert tokenizer.render([{"role": "user", "content": "t5"}], reply=False) == "t3 t5 t1 "


def test_v1_manifests_still_load(imported, tmp_path):
    """A model in the v1 layout, each block's file holding its tensors under
    its type's names, runs as its v2 import does."""
    _, _, out = imported
    manifest = Manifest.load(out)
    weight_map = manifest.weight_map(out)
    for name in ("tokenizer.json", "chat_template.jinja"):
        (tmp_path / name).write_bytes((out / name).read_bytes())
    (tmp_path / "blocks").mkdir()

    def to_v1(spec, file):
        tensors = {}
        for local, full in spec.tensors.items():
            with safe_open(str(out / weight_map[full]), "pt") as handle:
                tensors[local] = handle.get_tensor(full)
        save_file(tensors, str(tmp_path / file))
        spec.file, spec.tensors = file, {}

    to_v1(manifest.embed, "embed.safetensors")
    to_v1(manifest.head, "head.safetensors")
    for i, block in enumerate(manifest.blocks):
        to_v1(block, f"blocks/{i:03d}.safetensors")
    manifest.format, manifest.weights = V1, None
    manifest.save(tmp_path)
    assert json.loads((tmp_path / FILE).read_text())["format"] == V1

    tokens = torch.arange(9)[None, :] % SMALL["vocab_size"]
    v1, v2 = Model(tmp_path), Model(out)
    assert torch.equal(v1.forward(tokens, v1.init_state(), 0), v2.forward(tokens, v2.init_state(), 0))


def test_unsupported_rotary_scaling_is_refused(tmp_path):
    config = transformers.LlamaConfig(
        **SMALL, rope_parameters={"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 4.0}
    )
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path / "checkpoint")
    tokenizer_files(tmp_path / "checkpoint", SMALL["vocab_size"])
    with pytest.raises(ValueError, match="'dynamic' is not supported"):
        import_checkpoint(tmp_path / "checkpoint", tmp_path / "out")


@pytest.mark.parametrize(
    "architecture, config_class",
    [
        ("Gemma2ForCausalLM", "Gemma2Config"),
        ("Gemma3ForCausalLM", "Gemma3TextConfig"),
        ("Olmo3ForCausalLM", "Olmo3Config"),
        ("SmolLM3ForCausalLM", "SmolLM3Config"),
    ],
)
def test_derived_layer_patterns_match_transformers(architecture, config_class):
    """Which layers slide, and which have rotary positions, when a config
    leaves them to transformers to derive."""
    from shem_neural.model.importers import FAMILIES, layer_types

    config = getattr(transformers, config_class)(num_hidden_layers=12)
    # What transformers derives, and what it derives it from when that is a default too
    left_out = ("layer_types", "no_rope_layers", "no_rope_layer_interval")
    raw = {key: value for key, value in config.to_dict().items() if key not in left_out}
    cfg = FAMILIES[architecture].config(raw)
    assert layer_types(cfg) == config.layer_types
    if hasattr(config, "no_rope_layers"):
        assert cfg["no_rope_layers"] == config.no_rope_layers


@pytest.mark.parametrize("kind", ["su", "yarn", "longrope"])
def test_phi3_longrope_by_any_name(kind):
    """Phi-3's configs call longrope by older names, which transformers reads as longrope."""
    from shem_neural.model.importers import FAMILIES, rope

    scaling = {"type": kind, "short_factor": [1.0, 2.0], "long_factor": [3.0, 4.0]}
    raw = {"max_position_embeddings": 64, "original_max_position_embeddings": 16, "rope_scaling": scaling}
    params = rope(FAMILIES["Phi3ForCausalLM"].config(raw), 4)
    assert params["rope_scaling"]["type"] == "longrope"
    assert params["rope_scaling"]["attention_factor"] == pytest.approx((1 + math.log(4) / math.log(16)) ** 0.5)
