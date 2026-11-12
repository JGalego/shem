"""Importers against transformers, on tiny random checkpoints.

Each family is built small and random by transformers itself, so these need
no downloads: the imported model must give transformers' logits, in one pass
and token by token from its session state, and its directory must load in
transformers as the checkpoint it came from. `test_model` does the same with
real checkpoints.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

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


def qwen3_5_vlm():
    """Qwen3.5's vision-language model, whose language model shem imports."""
    text = SMALL | GDN | dict(head_dim=8, layer_types=["linear_attention", "full_attention"])
    vision = dict(depth=1, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=32)
    return transformers.Qwen3_5Config(text_config=text, vision_config=vision)


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
}


def tokenizer_files(directory: Path, vocab_size: int) -> None:
    """A word-level tokenizer and a chat template, which an import needs."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(vocab_size)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    template = "{% for m in messages %}{{ m.content }} {% endfor %}"
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": template, "eos_token": "t1"}))


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def imported(request, tmp_path_factory):
    """A tiny random checkpoint of one family, the reference model built from
    it, and the checkpoint imported."""
    torch.manual_seed(0)
    config = CONFIGS[request.param]()
    if request.param == "qwen3.5-vlm":
        reference = transformers.Qwen3_5ForConditionalGeneration(config).eval()
    else:
        reference = transformers.AutoModelForCausalLM.from_config(config).eval()
    # transformers starts biases at zero and norms at one, which would hide a
    # missing bias or a swapped norm: every tensor gets noise of its own
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    checkpoint = tmp_path_factory.mktemp("checkpoint")
    reference.save_pretrained(checkpoint)
    tokenizer_files(checkpoint, SMALL["vocab_size"])
    out = tmp_path_factory.mktemp("imported")
    import_checkpoint(checkpoint, out)
    return request.param, reference, out


def logits(reference, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return reference(input_ids=tokens).logits


def test_imported_models_match_transformers(imported):
    _, reference, out = imported
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
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()


def test_imported_directories_are_checkpoints(imported):
    """transformers loads the imported directory as it does the original:
    the same tensors, under the same names, found through the index."""
    name, reference, out = imported
    if name == "qwen3.5-vlm":
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
        **SMALL, rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0}
    )
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path / "checkpoint")
    tokenizer_files(tmp_path / "checkpoint", SMALL["vocab_size"])
    with pytest.raises(ValueError, match="'yarn' is not supported"):
        import_checkpoint(tmp_path / "checkpoint", tmp_path / "out")
