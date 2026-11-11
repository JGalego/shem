"""The model manifest, `shem.json`: what a model is, as files.

    {
      "format": "shem.model/v1",
      "source": {"repo": "Qwen/Qwen3-0.6B", "revision": "...", "architecture": "Qwen3ForCausalLM"},
      "hidden_size": 1024,
      "tokenizer": {"file": "tokenizer.json", "chat_template": "chat_template.jinja", "eos": [151645]},
      "embed": {"type": "embedding", "file": "embed.safetensors", "params": {}},
      "blocks": [{"type": "attention", "file": "blocks/000.safetensors", "params": {...}}, ...],
      "head": {"type": "head", "file": "head.safetensors", "params": {"tied": true, ...}}
    }

Paths are relative to the manifest's directory. Block types are names that
`blocks` registers, and their parameters are whatever that type needs.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

FORMAT = "shem.model/v1"
FILE = "shem.json"


@dataclass
class BlockSpec:
    type: str
    file: str
    params: dict = field(default_factory=dict)


@dataclass
class Manifest:
    hidden_size: int
    tokenizer: dict
    embed: BlockSpec
    blocks: list[BlockSpec]
    head: BlockSpec
    source: dict = field(default_factory=dict)
    format: str = FORMAT

    def save(self, directory: Path) -> None:
        (directory / FILE).write_text(json.dumps(asdict(self), indent=1) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "Manifest":
        data = json.loads((Path(directory) / FILE).read_text())
        if data.get("format") != FORMAT:
            raise ValueError(f"{directory}: not a {FORMAT} model")
        spec = lambda d: BlockSpec(**d)  # noqa: E731
        return cls(
            hidden_size=data["hidden_size"],
            tokenizer=data["tokenizer"],
            embed=spec(data["embed"]),
            blocks=[spec(b) for b in data["blocks"]],
            head=spec(data["head"]),
            source=data.get("source", {}),
        )
