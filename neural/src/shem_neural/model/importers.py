"""Importers: Hugging Face checkpoints into shem's model layout.

    python -m shem_neural.model.importers Qwen/Qwen3-0.6B out/model [--revision REV]

Each importer maps one architecture's tensors onto shem's block types and
writes one weights file per block, in the checkpoint's own type. They are
registered by the `architectures` name in the checkpoint's `config.json`, so
supporting a new model family means adding one function here, and new block
types in `blocks` if its layers are new.
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

from .manifest import BlockSpec, Manifest

PATTERNS = ["*.json", "*.safetensors", "*.jinja", "LICENSE*", "NOTICE*", "README.md"]
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


class Writer:
    """Writes block files into the output directory, numbered in order."""

    def __init__(self, out: Path, checkpoint: Checkpoint):
        self.out, self.ckpt = out, checkpoint
        self.blocks: list[BlockSpec] = []
        (out / "blocks").mkdir(parents=True, exist_ok=True)

    def file(self, name: str, tensors: dict[str, str]) -> str:
        """Save the checkpoint's tensors `tensors` (block-local name to
        checkpoint name) as `name`, and return its path."""
        save_file({local: self.ckpt.tensor(full).contiguous() for local, full in tensors.items()}, str(self.out / name))
        return name

    def block(self, kind: str, params: dict, tensors: dict[str, str]) -> None:
        name = f"blocks/{len(self.blocks):03d}-{kind}.safetensors"
        self.blocks.append(BlockSpec(kind, self.file(name, tensors), params))


def qwen_layers(w: Writer, cfg: dict, prefix: str, offset: float) -> None:
    """The decoder layers of the Qwen3 family: each an attention or Gated
    DeltaNet block, then an MLP block."""
    eps = cfg["rms_norm_eps"]
    rope = cfg.get("rope_parameters") or {"rope_theta": cfg.get("rope_theta")}
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kinds = cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]
    for i, kind in enumerate(kinds):
        layer = f"{prefix}layers.{i}."
        if kind == "full_attention":
            attn = layer + "self_attn."
            names = {"norm": layer + "input_layernorm.weight"}
            names |= {part: f"{attn}{part}_proj.weight" for part in "qkvo"}
            if w.ckpt.has(attn + "q_norm.weight"):
                names |= {"q_norm": attn + "q_norm.weight", "k_norm": attn + "k_norm.weight"}
            params = {
                "heads": cfg["num_attention_heads"],
                "kv_heads": cfg["num_key_value_heads"],
                "head_dim": head_dim,
                "rope_theta": rope["rope_theta"],
                "rotary_dim": int(head_dim * rope.get("partial_rotary_factor", 1.0)),
                "output_gate": bool(cfg.get("attn_output_gate", False)),
                "eps": eps,
                "norm_offset": offset,
            }
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
        w.block("mlp", {"eps": eps, "norm_offset": offset}, names)


def qwen(w: Writer, cfg: dict, prefix: str, offset: float) -> tuple[BlockSpec, BlockSpec, dict]:
    embed = BlockSpec("embedding", w.file("embed.safetensors", {"weight": prefix + "embed_tokens.weight"}))
    qwen_layers(w, cfg, prefix, offset)
    tied = bool(cfg.get("tie_word_embeddings", False))
    names = {"norm": prefix + "norm.weight"}
    if not tied:
        names["weight"] = "lm_head.weight"
    params = {"eps": cfg["rms_norm_eps"], "norm_offset": offset, "tied": tied}
    head = BlockSpec("head", w.file("head.safetensors", names), params)
    return embed, head, cfg


def qwen3(w: Writer) -> tuple[BlockSpec, BlockSpec, dict]:
    return qwen(w, w.ckpt.config, "model.", 0.0)


def qwen3_5(w: Writer) -> tuple[BlockSpec, BlockSpec, dict]:
    """Qwen3.5's language model. Its vision encoder and its multi-token
    prediction layer are left out."""
    return qwen(w, w.ckpt.config["text_config"], "model.language_model.", 1.0)


IMPORTERS: dict[str, Callable[[Writer], tuple[BlockSpec, BlockSpec, dict]]] = {
    "Qwen3ForCausalLM": qwen3,
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
    checkpoint = Checkpoint(directory)
    architecture = checkpoint.config["architectures"][0]
    if architecture not in IMPORTERS:
        raise SystemExit(f"{repo}: no importer for {architecture}; shem has {sorted(IMPORTERS)}")
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out, checkpoint)
    embed, head, cfg = IMPORTERS[architecture](writer)

    shutil.copy(directory / "tokenizer.json", out / "tokenizer.json")
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
        source={"repo": repo, "revision": directory.name, "architecture": architecture} | model_license(directory),
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
