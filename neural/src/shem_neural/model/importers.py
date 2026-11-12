"""Importers: Hugging Face checkpoints into shem's model layout.

    python -m shem_neural.model.importers Qwen/Qwen3-0.6B out/model [--revision REV]
    python -m shem_neural.model.importers CHECKPOINT_DIR out/model [--in-place]

Each importer maps one architecture's tensors onto shem's block types and
writes one weights file per block. The tensors keep their names and types,
and an index says which file holds each, so the output is also a sharded
checkpoint that transformers loads as it would the original.

In place, the checkpoint's files stay as they are, and the import writes
only what a model needs besides them: the manifest, and an index and a chat
template if the checkpoint has none. A ModelPack artifact's image twin is
its files, unchanged, and a layer with these.

Families are registered by the `architectures` name in the checkpoint's
`config.json`. The decoders shem reads share one layout (Llama's) and differ
in what their checkpoint holds and their config turns on, which `decoder`
reads: where the norms are, fused projections, biases, sliding windows,
rotary scaling. A family's `config` fills in what transformers derives
rather than stores, such as which layers slide. A family with another
layout needs a reader of its own, and new block types in `blocks` if its
layers are new.
"""

import argparse
import json
import math
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from .manifest import INDEX, BlockSpec, Manifest

PATTERNS = ["*.json", "*.safetensors", "*.jinja", "LICENSE*", "NOTICE*", "README.md"]
#: The checkpoint's own files that travel with the model, so that its
#: directory loads as a checkpoint: its configs and its tokenizer
CHECKPOINT_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
# Hugging Face licence tags as SPDX identifiers, for the image's licence label
SPDX = {"apache-2.0": "Apache-2.0", "mit": "MIT", "bsd-3-clause": "BSD-3-Clause", "cc-by-4.0": "CC-BY-4.0"}


class Checkpoint:
    """A checkpoint's directory: its config and its tensors by name, found
    through its index when it has one, which may name files in directories
    of their own (as shem's imports do), or else in its safetensors files."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.config = json.loads((directory / "config.json").read_text())
        self.files = {}
        index = directory / INDEX
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self.files = {name: directory / file for name, file in weight_map.items()}
            return
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(str(path), "pt") as handle:
                for name in handle.keys():
                    self.files[name] = path

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(str(self.files[name]), "pt") as handle:
            return handle.get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.files

    def shape(self, name: str) -> list[int]:
        with safe_open(str(self.files[name]), "pt") as handle:
            return handle.get_slice(name).get_shape()

    def info(self, name: str) -> tuple[int, str]:
        """How many numbers the tensor `name` holds, and their type, by
        safetensors' name for it, without reading it."""
        with safe_open(str(self.files[name]), "pt") as handle:
            tensor = handle.get_slice(name)
            return math.prod(tensor.get_shape()), tensor.get_dtype()

    def weight_map(self) -> dict[str, str]:
        """Which of the checkpoint's own files holds each tensor."""
        return {name: path.relative_to(self.directory).as_posix() for name, path in sorted(self.files.items())}

    def size(self) -> int:
        """The bytes of all the checkpoint's tensors."""
        total = 0
        for path in sorted(set(self.files.values())):
            with safe_open(str(path), "pt") as handle:
                for name in handle.keys():
                    tensor = handle.get_slice(name)
                    count = math.prod(tensor.get_shape())
                    total += count * torch.empty(0, dtype=DTYPES[tensor.get_dtype()]).element_size()
        return total


#: safetensors' dtype names, as ModelPack's config names precisions
PRECISIONS = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3",
    "F8_E5M2": "float8_e5m2",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}
#: safetensors' dtype names, as torch's
DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class Writer:
    """Writes block files into the output directory, numbered in order, and
    keeps the index of which file holds each tensor. In place, it writes no
    weights, and the index is the checkpoint's own files."""

    def __init__(self, out: Path, checkpoint: Checkpoint, in_place: bool = False):
        self.out, self.ckpt, self.in_place = out, checkpoint, in_place
        self.blocks: list[BlockSpec] = []
        self.weight_map: dict[str, str] = {}
        self.size = 0
        if not in_place:
            (out / "blocks").mkdir(parents=True, exist_ok=True)

    def file(self, name: str, tensors: dict[str, str]) -> dict[str, str]:
        """Save the checkpoint's tensors `tensors` (block-local name to
        checkpoint name) as `name`, under the checkpoint's names, and return
        `tensors`. In place, the tensors stay where they are."""
        if self.in_place:
            return dict(tensors)
        saved = {full: self.ckpt.tensor(full).contiguous() for full in tensors.values()}
        save_file(saved, str(self.out / name), metadata={"format": "pt"})
        for full, tensor in saved.items():
            self.weight_map[full] = name
            self.size += tensor.numel() * tensor.element_size()
        return dict(tensors)

    def block(self, kind: str, params: dict, tensors: dict[str, str]) -> None:
        name = f"blocks/{len(self.blocks):03d}-{kind}.safetensors"
        self.blocks.append(BlockSpec(kind, params, self.file(name, tensors)))

    def biases(self, names: dict[str, str]) -> dict[str, str]:
        """The biases the checkpoint has for the projections in `names`, as
        `NAME_bias`, which blocks add when they are there."""
        found = {}
        for local, full in names.items():
            bias = full.removesuffix(".weight") + ".bias"
            if full.endswith(".weight") and local != "norm" and self.ckpt.has(bias):
                found[f"{local}_bias"] = bias
        return found

    def index(self) -> None:
        """Write the checkpoint index, as Hugging Face names and lays it out.
        In place, a checkpoint with an index of its own keeps it, and one
        without gets one for its own files, every tensor in them included."""
        weight_map, size = dict(sorted(self.weight_map.items())), self.size
        if self.in_place:
            if (self.ckpt.directory / INDEX).exists():
                return
            weight_map, size = self.ckpt.weight_map(), self.ckpt.size()
        index = {"metadata": {"total_size": size}, "weight_map": weight_map}
        (self.out / INDEX).write_text(json.dumps(index, indent=2) + "\n")


def rope(cfg: dict, head_dim: int, kind: str = "full_attention") -> dict:
    """The rotary parameters of an attention block in a layer of `kind`,
    from any of the ways a config gives them: `rope_parameters`
    (transformers 5), for all layers or per kind of layer, or `rope_theta`
    with an optional `rope_scaling`. What scaling needs from the rest of the
    config, such as YaRN's and longrope's attention factors, is worked out
    here, so a block gets numbers rather than recipes. Kinds of scaling the
    blocks can't run are refused rather than run wrong."""
    given = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    if isinstance(given.get(kind), dict):
        given = given[kind]
    elif any(isinstance(value, dict) for value in given.values()):
        raise ValueError(f"the config has no RoPE parameters for {kind} layers")
    theta = given.get("rope_theta") or cfg.get("rope_theta") or 10000.0
    fraction = given.get("partial_rotary_factor") or cfg.get("partial_rotary_factor") or 1.0
    params = {"rope_theta": theta, "rotary_dim": int(head_dim * fraction)}
    scaling = given.get("rope_type") or given.get("type") or "default"
    longest = cfg.get("max_position_embeddings")
    context = given.get("original_max_position_embeddings") or cfg.get("original_max_position_embeddings") or longest
    if scaling == "linear":
        params["rope_scaling"] = {"type": "linear", "factor": given["factor"]}
    elif scaling == "llama3":
        params["rope_scaling"] = {
            "type": "llama3",
            "factor": given["factor"],
            "low_freq_factor": given["low_freq_factor"],
            "high_freq_factor": given["high_freq_factor"],
            "original_max_position_embeddings": context,
        }
    elif scaling == "yarn":
        factor = given.get("factor") or longest / context

        def mscale(scale: float, m: float = 1.0) -> float:
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

        attention = given.get("attention_factor")
        if attention is None:
            m, m_all = given.get("mscale"), given.get("mscale_all_dim")
            attention = mscale(factor, m) / mscale(factor, m_all) if m and m_all else mscale(factor)
        params["rope_scaling"] = {
            "type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": context,
            "beta_fast": given.get("beta_fast") or 32,
            "beta_slow": given.get("beta_slow") or 1,
            "truncate": given.get("truncate", True),
            "attention_factor": attention,
        }
    elif scaling in ("longrope", "su"):
        factor = given.get("factor") or longest / context
        attention = given.get("attention_factor")
        if attention is None:
            attention = 1.0 if factor <= 1 else math.sqrt(1 + math.log(factor) / math.log(context))
        params["rope_scaling"] = {
            "type": "longrope",
            "short_factor": given["short_factor"],
            "long_factor": given["long_factor"],
            "original_max_position_embeddings": context,
            "attention_factor": attention,
        }
    elif scaling != "default":
        raise ValueError(f"RoPE scaling {scaling!r} is not supported yet; shem has linear, llama3, yarn and longrope")
    return params


def sliding_window(cfg: dict) -> int | None:
    """The window of sliding-attention layers, when the config turns them on:
    Mistral whenever it has a `sliding_window`, Qwen2 and Qwen3 only with
    `use_sliding_window`."""
    return cfg.get("sliding_window") if cfg.get("use_sliding_window", True) else None


def layer_types(cfg: dict) -> list[str]:
    """Each layer's kind of token mixer: the config's `layer_types`, or else
    full attention, sliding from layer `max_window_layers` on (Qwen2 and
    Qwen3) or in every layer (Mistral and Phi-3) when there is a sliding
    window. Families whose pattern transformers derives fill it in first."""
    if cfg.get("layer_types"):
        return cfg["layer_types"]
    first = cfg.get("max_window_layers", 0) if "use_sliding_window" in cfg else 0
    window = sliding_window(cfg)
    return ["sliding_attention" if window and i >= first else "full_attention" for i in range(cfg["num_hidden_layers"])]


#: Activation names in configs, as the `mlp` block's gates
ACTIVATIONS = {"silu": "silu", "swish": "silu", "gelu": "gelu", "gelu_pytorch_tanh": "gelu_tanh"}


def activation(cfg: dict) -> str:
    name = cfg.get("hidden_activation") or cfg.get("hidden_act") or "silu"
    if name not in ACTIVATIONS:
        raise ValueError(f"MLP activation {name!r} is not supported yet; shem has {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]


def norms(w: Writer, layer: str) -> tuple[dict[str, str], dict[str, str]]:
    """The norms of a layer's token mixer and of its MLP, by what the
    checkpoint holds: one before each (Llama), one around each (Gemma) or
    one after each (OLMo 2)."""

    def named(**names: str) -> dict[str, str]:
        return {local: f"{layer}{name}.weight" for local, name in names.items()}

    if w.ckpt.has(f"{layer}pre_feedforward_layernorm.weight"):
        mixer = named(norm="input_layernorm", post_norm="post_attention_layernorm")
        return mixer, named(norm="pre_feedforward_layernorm", post_norm="post_feedforward_layernorm")
    if w.ckpt.has(f"{layer}post_feedforward_layernorm.weight"):
        return named(post_norm="post_attention_layernorm"), named(post_norm="post_feedforward_layernorm")
    return named(norm="input_layernorm"), named(norm="post_attention_layernorm")


def attention_block(
    w: Writer, cfg: dict, layer: str, kind: str, norm_names: dict[str, str], rotary: bool
) -> tuple[dict, dict[str, str]]:
    """An attention layer's block: its tensors, and its parameters from the
    config and from the tensors' shapes."""
    attn = layer + "self_attn."
    if w.ckpt.has(attn + "qkv_proj.weight"):
        names = {"qkv": attn + "qkv_proj.weight", "o": attn + "o_proj.weight"}
    else:
        names = {part: f"{attn}{part}_proj.weight" for part in "qkvo"}
    names |= w.biases(names) | norm_names
    if w.ckpt.has(attn + "q_norm.weight"):
        names |= {"q_norm": attn + "q_norm.weight", "k_norm": attn + "k_norm.weight"}
    heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    params = {
        "heads": heads,
        "kv_heads": cfg["num_key_value_heads"],
        "head_dim": head_dim,
        **rope(cfg, head_dim, kind),
        # A gated query projection is twice as wide: a query and a gate per head
        "output_gate": "q" in names and w.ckpt.shape(names["q"])[0] == 2 * heads * head_dim,
        "eps": cfg["rms_norm_eps"],
    }
    if not rotary:
        params["rotary_dim"] = 0
        params.pop("rope_scaling", None)
    if "q_norm" in names and w.ckpt.shape(names["q_norm"])[0] != head_dim:
        params["qk_norm"] = "projection"
    if cfg.get("query_pre_attn_scalar"):
        params["scale"] = cfg["query_pre_attn_scalar"] ** -0.5
    if cfg.get("attn_logit_softcapping"):
        params["softcap"] = cfg["attn_logit_softcapping"]
    window = sliding_window(cfg)
    if kind == "sliding_attention" and window:
        params["sliding_window"] = window
    return params, names


def decoder_layers(w: Writer, cfg: dict, prefix: str, offset: float) -> None:
    """The decoder layers of Llama and the families that share its layout:
    each an attention or Gated DeltaNet block, then an MLP block."""
    eps = cfg["rms_norm_eps"]
    gate = activation(cfg)
    with_rope = cfg.get("no_rope_layers")
    for i, kind in enumerate(layer_types(cfg)):
        layer = f"{prefix}layers.{i}."
        mixer_norms, mlp_norms = norms(w, layer)
        if kind in ("full_attention", "sliding_attention"):
            params, names = attention_block(w, cfg, layer, kind, mixer_norms, not with_rope or bool(with_rope[i]))
            w.block("attention", params | {"norm_offset": offset}, names)
        elif kind == "linear_attention":
            gdn = layer + "linear_attn."
            names = mixer_norms | {
                "in_qkv": gdn + "in_proj_qkv.weight",
                "in_z": gdn + "in_proj_z.weight",
                "in_b": gdn + "in_proj_b.weight",
                "in_a": gdn + "in_proj_a.weight",
                "conv": gdn + "conv1d.weight",
                "dt_bias": gdn + "dt_bias",
                "A_log": gdn + "A_log",
                "out_norm": gdn + "norm.weight",
                "out": gdn + "out_proj.weight",
            }
            params = {
                "k_heads": cfg["linear_num_key_heads"],
                "v_heads": cfg["linear_num_value_heads"],
                "k_dim": cfg["linear_key_head_dim"],
                "v_dim": cfg["linear_value_head_dim"],
                "conv_kernel": cfg["linear_conv_kernel_dim"],
                "eps": eps,
                "norm_offset": offset,
            }
            w.block("gated_deltanet", params, names)
        else:
            raise ValueError(f"layer {i}: unknown layer type {kind!r}")
        mlp = layer + "mlp."
        if w.ckpt.has(mlp + "gate_up_proj.weight"):
            names = {"gate_up": mlp + "gate_up_proj.weight", "down": mlp + "down_proj.weight"}
        else:
            names = {part: f"{mlp}{part}_proj.weight" for part in ("gate", "up", "down")}
        names |= w.biases(names) | mlp_norms
        params = {"eps": eps, "norm_offset": offset}
        if gate != "silu":
            params["activation"] = gate
        w.block("mlp", params, names)


def decoder(w: Writer, family: "Family") -> tuple[BlockSpec, BlockSpec, dict]:
    """A decoder in Llama's layout: an embedding, the layers, a final norm
    and a head, tied to the embedding or not, read as `family` says."""
    cfg = family.config(w.ckpt.config)
    prefix = next((p for p in family.prefixes if w.ckpt.has(p + "embed_tokens.weight")), family.prefixes[0])
    embed_params = {"scale": cfg["hidden_size"] ** 0.5} if family.embed_scale else {}
    embed_names = {"weight": prefix + "embed_tokens.weight"}
    embed = BlockSpec("embedding", embed_params, w.file("embed.safetensors", embed_names))
    decoder_layers(w, cfg, prefix, family.offset)
    head = "lm_head.weight"
    tied = cfg.get("tie_word_embeddings")
    if tied is None:
        tied = not w.ckpt.has(head)
    names = {"norm": prefix + "norm.weight"}
    if not tied:
        names["weight"] = head
    params = {"eps": cfg["rms_norm_eps"], "norm_offset": family.offset, "tied": bool(tied)}
    if cfg.get("final_logit_softcapping"):
        params["softcap"] = cfg["final_logit_softcapping"]
    return embed, BlockSpec("head", params, w.file("head.safetensors", names)), cfg


def per_kind_rope(cfg: dict, sliding_theta: float, full_theta: float) -> dict:
    """`cfg` with rotary parameters per kind of layer, as transformers reads
    an older config of a family that scales only its full-attention layers
    (Gemma 3 and OLMo 3)."""
    if cfg.get("rope_parameters"):
        return cfg
    full = {"rope_theta": cfg.get("rope_theta") or full_theta} | (cfg.get("rope_scaling") or {})
    return cfg | {"rope_parameters": {"full_attention": full, "sliding_attention": {"rope_theta": sliding_theta}}}


def pattern(cfg: dict, period: int) -> dict:
    """`cfg` with `layer_types`, if it has none: full attention in every
    `period`th layer, and sliding attention between."""
    if cfg.get("layer_types"):
        return cfg
    kinds = ["sliding_attention" if (i + 1) % period else "full_attention" for i in range(cfg["num_hidden_layers"])]
    return cfg | {"layer_types": kinds}


def phi3_config(config: dict) -> dict:
    """Phi-3 and Phi-4-mini call their longrope scaling `su` or `yarn`."""
    cfg = dict(config)
    for key in ("rope_scaling", "rope_parameters"):
        given = cfg.get(key) or {}
        if (given.get("rope_type") or given.get("type")) in ("su", "yarn"):
            cfg[key] = {k: v for k, v in given.items() if k != "type"} | {"rope_type": "longrope"}
    return cfg


def olmo3_config(config: dict) -> dict:
    """OLMo 3 slides in three layers of every four, and scales positions only
    in the fourth."""
    theta = config.get("rope_theta") or 500000.0
    return per_kind_rope(pattern(config, 4), theta, theta)


def smollm3_config(config: dict) -> dict:
    """SmolLM3 has no rotary positions in every fourth layer, unless its
    config lists which."""
    if config.get("no_rope_layers"):
        return config
    interval = config.get("no_rope_layer_interval", 4)
    return config | {"no_rope_layers": [int((i + 1) % interval != 0) for i in range(config["num_hidden_layers"])]}


def gemma2_config(config: dict) -> dict:
    """Gemma 2 slides in every other layer."""
    return pattern(config, 2)


#: Gemma 3's text model as transformers fills it in, for the keys a config
#: leaves out: its vision-language checkpoints give only a few
GEMMA3_TEXT = {
    "vocab_size": 262208,
    "hidden_size": 2304,
    "intermediate_size": 9216,
    "num_hidden_layers": 26,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "query_pre_attn_scalar": 256,
    "sliding_window": 4096,
    "sliding_window_pattern": 6,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 131072,
}


def gemma3_config(config: dict) -> dict:
    """Gemma 3's text model, alone or inside its vision-language model: five
    sliding layers for every global one, which alone scale positions, from a
    base of their own."""
    cfg = GEMMA3_TEXT | config.get("text_config", config)
    if cfg.get("use_bidirectional_attention"):
        raise ValueError("Gemma 3 with bidirectional attention is an embedding model, which shem doesn't run")
    if "text_config" in config:
        cfg["tie_word_embeddings"] = config.get("tie_word_embeddings", True)
    cfg = pattern(cfg, cfg["sliding_window_pattern"])
    return per_kind_rope(cfg, cfg.get("rope_local_base_freq", 10000.0), 1e6)


def text_config(config: dict) -> dict:
    """The language model's config inside a vision-language model's."""
    return config.get("text_config", config)


@dataclass
class Family:
    """How to read the checkpoints of one architecture."""

    #: The language model's config, with what transformers derives filled in
    config: Callable[[dict], dict] = lambda config: config
    #: Where the language model's tensors are, as transformers 5 names them
    #: and as older checkpoints did
    prefixes: tuple[str, ...] = ("model.",)
    #: 1 for zero-centred norm weights
    offset: float = 0.0
    #: Whether embeddings are scaled by the square root of the width, as Gemma's are
    embed_scale: bool = False


GEMMA = {"offset": 1.0, "embed_scale": True}
#: The architectures shem imports, by the name in their config. Llama,
#: Mistral, Qwen2, Qwen3 and OLMo 2 differ only in what their checkpoints
#: and configs turn on; Phi-3 fuses projections, which the blocks split;
#: Gemma puts norms around its blocks and softcaps
FAMILIES: dict[str, Family] = {
    "LlamaForCausalLM": Family(),
    "MistralForCausalLM": Family(),
    "Qwen2ForCausalLM": Family(),
    "Qwen3ForCausalLM": Family(),
    "Olmo2ForCausalLM": Family(),
    "Olmo3ForCausalLM": Family(olmo3_config),
    "Phi3ForCausalLM": Family(phi3_config),
    "SmolLM3ForCausalLM": Family(smollm3_config),
    "Gemma2ForCausalLM": Family(gemma2_config, **GEMMA),
    "Gemma3ForCausalLM": Family(gemma3_config, **GEMMA),
    "Gemma3ForConditionalGeneration": Family(
        gemma3_config, ("model.language_model.", "language_model.model."), **GEMMA
    ),
    "Qwen3_5ForCausalLM": Family(offset=1.0),
    "Qwen3_5ForConditionalGeneration": Family(text_config, ("model.language_model.",), offset=1.0),
}


def eos_tokens(checkpoint: Checkpoint, cfg: dict) -> list[int]:
    """The tokens that end a reply: the generation config's and the model
    config's end-of-sequence tokens, and the tokenizer's `eos_token`, which
    for chat models is the end of a turn (`<|im_end|>`) and is sometimes
    declared nowhere else."""
    ids = []
    generation = checkpoint.directory / "generation_config.json"
    for source in (json.loads(generation.read_text()) if generation.exists() else {}, cfg, checkpoint.config):
        value = source.get("eos_token_id")
        ids += value if isinstance(value, list) else [value] if value is not None else []
    settings = checkpoint.directory / "tokenizer_config.json"
    if settings.exists():
        token = json.loads(settings.read_text()).get("eos_token")
        token = token.get("content") if isinstance(token, dict) else token
        added = json.loads((checkpoint.directory / "tokenizer.json").read_text()).get("added_tokens", [])
        ids += [entry["id"] for entry in added if token and entry["content"] == token]
    return sorted(set(ids))


#: The special tokens transformers hands a chat template, as `bos_token` and so on
SPECIAL_TOKENS = ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"]


def special_tokens(directory: Path) -> dict[str, str]:
    """The checkpoint's special tokens, as its chat template sees them: many
    templates start with `bos_token` or end turns with `eos_token`. They are
    in the tokenizer's config, or in an older special tokens map."""
    found = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        path = directory / name
        settings = json.loads(path.read_text()) if path.exists() else {}
        for key in SPECIAL_TOKENS:
            token = settings.get(key)
            token = token.get("content") if isinstance(token, dict) else token
            if token:
                found[key] = token
    return found


def model_license(directory: Path) -> dict:
    """The licence a checkpoint's model card declares in its front matter: the
    Hugging Face tag, its SPDX identifier where there is one, and its link."""
    card = directory / "README.md"
    if not card.exists():
        return {}
    text = card.read_text(errors="replace")
    if not text.startswith("---"):
        return {}
    front = text.split("---", 2)[1]
    fields = dict(line.split(":", 1) for line in front.splitlines() if ":" in line and not line.startswith(" "))
    tag = fields.get("license", "").strip()
    if not tag:
        return {}
    found = {"license": tag, "spdx": SPDX.get(tag, f"LicenseRef-{tag}")}
    if fields.get("license_link"):
        found["license_link"] = fields["license_link"].strip()
    return found


def import_model(repo: str, out: Path, revision: str | None = None) -> Manifest:
    """Download `repo` from Hugging Face and write it to `out` as a shem model."""
    directory = Path(snapshot_download(repo, revision=revision, allow_patterns=PATTERNS))
    return import_checkpoint(directory, out, {"repo": repo, "revision": directory.name})


def import_checkpoint(directory: Path, out: Path, source: dict | None = None, in_place: bool = False) -> Manifest:
    """Write the checkpoint in `directory` to `out` as a shem model. `source`
    says where it came from, for the manifest. In place, `out` gets only what
    the model needs besides the checkpoint's own files, and the model is
    `out`'s files laid over `directory`'s."""
    directory, out = Path(directory), Path(out)
    checkpoint = Checkpoint(directory)
    architecture = checkpoint.config["architectures"][0]
    if architecture not in FAMILIES:
        raise SystemExit(f"{directory}: no importer for {architecture}; shem has {sorted(FAMILIES)}")
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out, checkpoint, in_place)
    embed, head, cfg = decoder(writer, FAMILIES[architecture])
    writer.index()

    if not in_place:
        for name in CHECKPOINT_FILES:
            if (directory / name).exists():
                shutil.copy(directory / name, out / name)
        # The weights' licence and notices travel with them
        for notice in sorted(directory.glob("LICENSE*")) + sorted(directory.glob("NOTICE*")):
            shutil.copy(notice, out / notice.name)
    template = directory / "chat_template.jinja"
    if template.exists():
        if not in_place:
            shutil.copy(template, out / "chat_template.jinja")
    else:
        settings = json.loads((directory / "tokenizer_config.json").read_text())
        (out / "chat_template.jinja").write_text(settings["chat_template"])
    # What the model holds: the tensors it runs, the vision encoder of a
    # vision-language model's checkpoint not among them
    used = {name for spec in [embed, *writer.blocks, head] for name in spec.tensors.values()}
    infos = [checkpoint.info(name) for name in sorted(used)]
    family = {"family": checkpoint.config["model_type"]} if checkpoint.config.get("model_type") else {}
    manifest = Manifest(
        hidden_size=cfg["hidden_size"],
        tokenizer={
            "file": "tokenizer.json",
            "chat_template": "chat_template.jinja",
            "eos": eos_tokens(checkpoint, cfg),
            "special": special_tokens(directory),
        },
        embed=embed,
        blocks=writer.blocks,
        head=head,
        source=(source or {}) | {"architecture": architecture} | family | model_license(directory),
        parameters=sum(count for count, _ in infos),
        precision=sorted({PRECISIONS.get(dtype, dtype.lower()) for _, dtype in infos}),
    )
    manifest.save(out)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="a Hugging Face model, such as Qwen/Qwen3-0.6B, or a checkpoint's directory")
    parser.add_argument("out", type=Path, help="the directory to write the model to")
    parser.add_argument("--revision", help="a branch, tag or commit of the Hugging Face model")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="for a checkpoint's directory: leave its files where they are, and write only the rest of the model",
    )
    args = parser.parse_args()
    if Path(args.source).is_dir():
        manifest = import_checkpoint(Path(args.source), args.out, in_place=args.in_place)
    elif args.in_place:
        raise SystemExit("--in-place imports a checkpoint's directory")
    else:
        manifest = import_model(args.source, args.out, args.revision)
    # Progress goes to stderr: a pull may import a model on its way to a run
    print(f"{args.source}: {len(manifest.blocks)} blocks", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
