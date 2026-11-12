"""Importers: Hugging Face checkpoints into shem's model layout.

    python -m shem_neural.model.importers Qwen/Qwen3-0.6B out/model [--revision REV]

Each importer maps one architecture's tensors onto shem's block types and
writes one weights file per block. The tensors keep their names and types,
and an index says which file holds each, so the output is also a sharded
checkpoint that transformers loads as it would the original.

Importers are registered by the `architectures` name in the checkpoint's
`config.json`. Most decoders share one layout (Llama's) and differ in what
their config turns on, which `decoder` reads; a family that differs more
gets its own function here, and new block types in `blocks` if its layers
are new.
"""

import argparse
import json
import shutil
from collections.abc import Callable
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
    """A downloaded checkpoint: its config and its tensors by name."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.config = json.loads((directory / "config.json").read_text())
        self.files = {}
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


class Writer:
    """Writes block files into the output directory, numbered in order, and
    keeps the index of which file holds each tensor."""

    def __init__(self, out: Path, checkpoint: Checkpoint):
        self.out, self.ckpt = out, checkpoint
        self.blocks: list[BlockSpec] = []
        self.weight_map: dict[str, str] = {}
        self.size = 0
        (out / "blocks").mkdir(parents=True, exist_ok=True)

    def file(self, name: str, tensors: dict[str, str]) -> dict[str, str]:
        """Save the checkpoint's tensors `tensors` (block-local name to
        checkpoint name) as `name`, under the checkpoint's names, and return
        `tensors`."""
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
        """Write the checkpoint index, as Hugging Face names and lays it out."""
        index = {"metadata": {"total_size": self.size}, "weight_map": dict(sorted(self.weight_map.items()))}
        (self.out / INDEX).write_text(json.dumps(index, indent=2) + "\n")


def rope(cfg: dict, head_dim: int) -> dict:
    """An attention block's rotary parameters, from either way a config
    gives them: `rope_parameters` (transformers 5) or `rope_theta` with an
    optional `rope_scaling`. Positions are scaled linearly or as Llama 3
    does; other kinds of scaling are refused rather than run wrong."""
    given = dict(cfg.get("rope_parameters") or cfg.get("rope_scaling") or {})
    if any(isinstance(value, dict) for value in given.values()):
        raise ValueError("RoPE parameters per layer type are not supported yet")
    theta = given.get("rope_theta") or cfg.get("rope_theta") or 10000.0
    fraction = given.get("partial_rotary_factor") or cfg.get("partial_rotary_factor") or 1.0
    params = {"rope_theta": theta, "rotary_dim": int(head_dim * fraction)}
    kind = given.get("rope_type") or given.get("type") or "default"
    if kind == "linear":
        params["rope_scaling"] = {"type": "linear", "factor": given["factor"]}
    elif kind == "llama3":
        context = given.get("original_max_position_embeddings") or cfg["max_position_embeddings"]
        params["rope_scaling"] = {
            "type": "llama3",
            "factor": given["factor"],
            "low_freq_factor": given["low_freq_factor"],
            "high_freq_factor": given["high_freq_factor"],
            "original_max_position_embeddings": context,
        }
    elif kind != "default":
        raise ValueError(f"RoPE scaling {kind!r} is not supported yet; shem has linear and llama3")
    return params


def sliding_window(cfg: dict) -> int | None:
    """The window of sliding-attention layers, when the config turns them on:
    Mistral whenever it has a `sliding_window`, Qwen2 and Qwen3 only with
    `use_sliding_window`."""
    return cfg.get("sliding_window") if cfg.get("use_sliding_window", True) else None


def layer_types(cfg: dict) -> list[str]:
    """Each layer's kind of token mixer: the config's `layer_types`, or else
    full attention, sliding from layer `max_window_layers` on (Qwen2 and
    Qwen3) or in every layer (Mistral) when there is a sliding window."""
    if cfg.get("layer_types"):
        return cfg["layer_types"]
    first = cfg.get("max_window_layers", 0) if "use_sliding_window" in cfg else 0
    window = sliding_window(cfg)
    return ["sliding_attention" if window and i >= first else "full_attention" for i in range(cfg["num_hidden_layers"])]


def decoder_layers(w: Writer, cfg: dict, prefix: str, offset: float) -> None:
    """The decoder layers of Llama and the families that share its layout:
    each an attention or Gated DeltaNet block, then an MLP block."""
    eps = cfg["rms_norm_eps"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    rotary = rope(cfg, head_dim)
    activation = cfg.get("hidden_act", "silu")
    if activation not in ("silu", "swish"):
        raise ValueError(f"MLP activation {activation!r} is not supported yet; shem has SwiGLU")
    window = sliding_window(cfg)
    for i, kind in enumerate(layer_types(cfg)):
        layer = f"{prefix}layers.{i}."
        if kind in ("full_attention", "sliding_attention"):
            attn = layer + "self_attn."
            names = {"norm": layer + "input_layernorm.weight"}
            names |= {part: f"{attn}{part}_proj.weight" for part in "qkvo"}
            names |= w.biases(names)
            if w.ckpt.has(attn + "q_norm.weight"):
                names |= {"q_norm": attn + "q_norm.weight", "k_norm": attn + "k_norm.weight"}
            heads = cfg["num_attention_heads"]
            params = {
                "heads": heads,
                "kv_heads": cfg["num_key_value_heads"],
                "head_dim": head_dim,
                **rotary,
                # A gated query projection is twice as wide: a query and a gate per head
                "output_gate": w.ckpt.shape(names["q"])[0] == 2 * heads * head_dim,
                "eps": eps,
                "norm_offset": offset,
            }
            if kind == "sliding_attention" and window:
                params["sliding_window"] = window
            w.block("attention", params, names)
        elif kind == "linear_attention":
            gdn = layer + "linear_attn."
            names = {
                "norm": layer + "input_layernorm.weight",
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
        names = {"norm": layer + "post_attention_layernorm.weight"}
        names |= {part: f"{mlp}{part}_proj.weight" for part in ("gate", "up", "down")}
        names |= w.biases(names)
        w.block("mlp", {"eps": eps, "norm_offset": offset}, names)


def decoder(w: Writer, cfg: dict, prefix: str, offset: float) -> tuple[BlockSpec, BlockSpec, dict]:
    """A decoder in Llama's layout: an embedding, the layers under `prefix`,
    a final norm and a head, tied to the embedding or not. `offset` is 1 for
    families whose norm weights are zero-centred."""
    embed = BlockSpec("embedding", tensors=w.file("embed.safetensors", {"weight": prefix + "embed_tokens.weight"}))
    decoder_layers(w, cfg, prefix, offset)
    tied = bool(cfg.get("tie_word_embeddings", False))
    names = {"norm": prefix + "norm.weight"}
    if not tied:
        names["weight"] = "lm_head.weight"
    params = {"eps": cfg["rms_norm_eps"], "norm_offset": offset, "tied": tied}
    head = BlockSpec("head", params, w.file("head.safetensors", names))
    return embed, head, cfg


def llama(w: Writer) -> tuple[BlockSpec, BlockSpec, dict]:
    """Llama, Mistral, Qwen2 and Qwen3, which differ only in what their
    configs turn on: biases, norms on queries and keys, sliding windows and
    rotary scaling."""
    return decoder(w, w.ckpt.config, "model.", 0.0)


def qwen3_5(w: Writer) -> tuple[BlockSpec, BlockSpec, dict]:
    """Qwen3.5's language model, alone or inside its vision-language model,
    whose vision encoder and multi-token prediction layer are left out."""
    cfg = w.ckpt.config
    if "text_config" in cfg:
        return decoder(w, cfg["text_config"], "model.language_model.", 1.0)
    return decoder(w, cfg, "model.", 1.0)


IMPORTERS: dict[str, Callable[[Writer], tuple[BlockSpec, BlockSpec, dict]]] = {
    "LlamaForCausalLM": llama,
    "MistralForCausalLM": llama,
    "Qwen2ForCausalLM": llama,
    "Qwen3ForCausalLM": llama,
    "Qwen3_5ForCausalLM": qwen3_5,
    "Qwen3_5ForConditionalGeneration": qwen3_5,
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


def import_checkpoint(directory: Path, out: Path, source: dict | None = None) -> Manifest:
    """Write the checkpoint in `directory` to `out` as a shem model. `source`
    says where it came from, for the manifest."""
    directory, out = Path(directory), Path(out)
    checkpoint = Checkpoint(directory)
    architecture = checkpoint.config["architectures"][0]
    if architecture not in IMPORTERS:
        raise SystemExit(f"{directory}: no importer for {architecture}; shem has {sorted(IMPORTERS)}")
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out, checkpoint)
    embed, head, cfg = IMPORTERS[architecture](writer)
    writer.index()

    for name in CHECKPOINT_FILES:
        if (directory / name).exists():
            shutil.copy(directory / name, out / name)
    # The weights' licence and notices travel with them
    for notice in sorted(directory.glob("LICENSE*")) + sorted(directory.glob("NOTICE*")):
        shutil.copy(notice, out / notice.name)
    template = directory / "chat_template.jinja"
    if template.exists():
        shutil.copy(template, out / "chat_template.jinja")
    else:
        settings = json.loads((directory / "tokenizer_config.json").read_text())
        (out / "chat_template.jinja").write_text(settings["chat_template"])
    manifest = Manifest(
        hidden_size=cfg["hidden_size"],
        tokenizer={
            "file": "tokenizer.json",
            "chat_template": "chat_template.jinja",
            "eos": eos_tokens(checkpoint, cfg),
        },
        embed=embed,
        blocks=writer.blocks,
        head=head,
        source=(source or {}) | {"architecture": architecture} | model_license(directory),
    )
    manifest.save(out)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo", help="a Hugging Face model, such as Qwen/Qwen3-0.6B")
    parser.add_argument("out", type=Path, help="the directory to write the model to")
    parser.add_argument("--revision", help="a branch, tag or commit of the repo")
    args = parser.parse_args()
    manifest = import_model(args.repo, args.out, args.revision)
    print(f"{args.repo}: {len(manifest.blocks)} blocks", flush=True)


if __name__ == "__main__":
    main()
