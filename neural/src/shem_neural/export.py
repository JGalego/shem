"""Export the compiled CPU as a model repository, for the Hugging Face Hub.

    python -m shem_neural.export OUT [--version X.Y.Z]

writes `model.safetensors`, the network's weights as `compile.compile_weights`
makes them; `config.json`, which describes the machine they implement; and
`README.md`, the model card. Nothing is trained, so the same version of shem
always exports the same weights. Each release publishes them.
"""

import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from . import isa
from .compile import compile_weights, save

CARD = """---
license: mit
library_name: shem
tags:
- shem
- riscv
- rv32im
- compiled
- neural-cpu
---

# shem RV32IM

An RV32IM CPU compiled into network weights: one instruction per forward pass, exact. It is the neural vCPU of [shem](https://github.com/JGalego/shem), which runs containers and VMs inside a model's weights, and these are its weights for shem {version}.

Nothing here is learned. The weight compiler turns the instruction set, {count} instructions described as data, into the lookup and routing tables of the network's linear layers: decoding is a layer that counts mismatching bits, operands are attention reads of a register file, and results are routed by the decoded instruction. Memory and registers live in delta-rule state, as Gated DeltaNet layers hold memory. The network uses only linear maps, gating, clamps and one-hot attention, in float64, so every value it computes is exact.

## Verified

- All 60 official riscv-tests (rv32ui, rv32um, rv32ua) end in the same state as on shem's reference interpreter.
- Random programs, snapshots resumed from the reference interpreter, forking containers, and a no-MMU Linux kernel running a shell command, about 180,000 instructions, end in identical states, all 64 MiB of RAM included.
- On GPUs, every mode (eager, compiled, CUDA graphs) is bit-exact against the CPU.

## Use

```sh
uvx --from shem-neural shem-neural-server --socket /tmp/shem-neural.sock &
shem run --vcpu neural --neural-socket /tmp/shem-neural.sock program
```

The server compiles these same weights itself; this repository is the published copy, for inspection and for images that carry their CPU as a layer.

## Files

| File | What it is |
|---|---|
| `model.safetensors` | The weights, {tensors} tensors, {values:,} values, float64 |
| `config.json` | The machine: its instruction set and the names of the units the routing tables select |
"""


def export(out: Path, version: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    weights = compile_weights()
    save(weights, out / "model.safetensors")
    config = {
        "format": "shem.cpu/v1",
        "isa": "rv32im",
        "shem_version": version,
        "instructions": [ins.name for ins in isa.INSTRUCTIONS],
        "formats": list(isa.FORMATS),
        "results": list(isa.RESULTS),
        "next_pc": list(isa.NEXT_PC),
        "branches": list(isa.BRANCHES),
        "exits": list(isa.EXITS),
        "sha256": hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    card = CARD.format(
        version=version,
        count=len(isa.INSTRUCTIONS),
        tensors=len(weights),
        values=sum(t.numel() for t in weights.values()),
    )
    (out / "README.md").write_text(card)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--version", help="the shem version (default: this package's)")
    args = parser.parse_args()
    try:
        version = args.version or package_version("shem-neural")
    except PackageNotFoundError:
        version = "unknown"
    config = export(args.out, version)
    print(f"{args.out}: rv32im weights for shem {version}, sha256 {config['sha256'][:16]}")


if __name__ == "__main__":
    main()
