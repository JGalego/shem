"""The model manifest, `shem.json`: what a model is, as files.

    {
      "format": "shem.model/v2",
      "source": {"repo": "Qwen/Qwen3-0.6B", "revision": "...", "architecture": "Qwen3ForCausalLM"},
      "hidden_size": 1024,
      "tokenizer": {"file": "tokenizer.json", "chat_template": "chat_template.jinja", "eos": [151645]},
      "weights": "model.safetensors.index.json",
      "embed": {"type": "embedding", "tensors": {"weight": "model.embed_tokens.weight"}},
      "blocks": [{"type": "attention", "params": {...},
                  "tensors": {"q": "model.layers.0.self_attn.q_proj.weight", ...}}, ...],
      "head": {"type": "head", "params": {"tied": true, ...}, "tensors": {"norm": "model.norm.weight"}}
    }

A block maps the names its type uses to the checkpoint's own tensor names.
`weights` is a Hugging Face sharded-checkpoint index, which says which
safetensors file holds each tensor. So the model's directory is also a
checkpoint that transformers can load, and whether the files follow the
blocks or the checkpoint's own shards is a choice of layout, not of format.

Paths are relative to the manifest's directory. Block types are names that
`blocks` registers, and their parameters are whatever that type needs.

`shem.model/v1` manifests still load: each of their blocks names a `file`
that holds its tensors under the block type's names.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

FORMAT = "shem.model/v2"
V1 = "shem.model/v1"
FILE = "shem.json"
#: The checkpoint index a v2 model's tensors are found through, by Hugging Face's name for it
INDEX = "model.safetensors.index.json"


@dataclass
class BlockSpec:
    type: str
    params: dict = field(default_factory=dict)
    #: The block's tensors: its type's names for them, to the checkpoint's names
    tensors: dict[str, str] = field(default_factory=dict)
    #: In a v1 manifest, the file that holds the block's tensors under its type's names
    file: str | None = None

    def to_json(self) -> dict:
        data = {"type": self.type, "params": self.params}
        if self.file is not None:
            data["file"] = self.file
        else:
            data["tensors"] = self.tensors
        return data

    @classmethod
    def from_json(cls, data: dict) -> "BlockSpec":
        return cls(data["type"], data.get("params", {}), data.get("tensors", {}), data.get("file"))


@dataclass
class Manifest:
    hidden_size: int
    tokenizer: dict
    embed: BlockSpec
    blocks: list[BlockSpec]
    head: BlockSpec
    source: dict = field(default_factory=dict)
    #: The checkpoint index, in a v2 manifest
    weights: str | None = INDEX
    format: str = FORMAT

    def save(self, directory: Path) -> None:
        data = {
            "format": self.format,
            "source": self.source,
            "hidden_size": self.hidden_size,
            "tokenizer": self.tokenizer,
        }
        if self.weights is not None:
            data["weights"] = self.weights
        data |= {
            "embed": self.embed.to_json(),
            "blocks": [block.to_json() for block in self.blocks],
            "head": self.head.to_json(),
        }
        (Path(directory) / FILE).write_text(json.dumps(data, indent=1) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "Manifest":
        data = json.loads((Path(directory) / FILE).read_text())
        if data.get("format") not in (FORMAT, V1):
            raise ValueError(f"{directory}: not a {FORMAT} model")
        return cls(
            hidden_size=data["hidden_size"],
            tokenizer=data["tokenizer"],
            embed=BlockSpec.from_json(data["embed"]),
            blocks=[BlockSpec.from_json(b) for b in data["blocks"]],
            head=BlockSpec.from_json(data["head"]),
            source=data.get("source", {}),
            weights=data.get("weights"),
            format=data["format"],
        )

    def weight_map(self, directory: Path) -> dict[str, str]:
        """Which file holds each tensor, by the checkpoint's name for it."""
        if self.weights is None:
            return {}
        return json.loads((Path(directory) / self.weights).read_text())["weight_map"]


def locate(path: Path) -> tuple[Path, Path]:
    """The model's directory and the image's root, for `path`: an image's
    root, whose model is at `/model` (where shem imports models) or at `/`
    (where a ModelPack artifact's files land); or a model's own directory,
    whose image is the one around it when it is `/model`."""
    path = Path(path)
    if (path / "model" / FILE).is_file():
        return path / "model", path
    if (path / FILE).is_file():
        return path, path.parent if path.name == "model" else path
    raise FileNotFoundError(f"{path}: no {FILE} here or in model/")
