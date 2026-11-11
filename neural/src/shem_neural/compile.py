"""The weight compiler: turn the ISA description into the network's tensors.

Nothing here is learned. Each tensor is a lookup or routing table derived from
`isa.INSTRUCTIONS`, and `cpu.NeuralCpu` uses them as the weights of its linear
layers. `compile_weights` builds them, and `save` and `load` store them as
safetensors, the format a shem machine image carries weights in.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from . import isa

DTYPE = torch.float64


def _matcher(patterns: list[tuple[int, int]], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights and bias of a layer that counts mismatches: for bits `x`,
    `x @ w.T + c` is how many masked bits differ from each pattern's match.
    `relu(1 - that)` is then a one-hot over the patterns."""
    w = torch.zeros(len(patterns), width, dtype=DTYPE)
    c = torch.zeros(len(patterns), dtype=DTYPE)
    for k, (mask, match) in enumerate(patterns):
        for i in range(width):
            if mask >> i & 1:
                bit = match >> i & 1
                w[k, i] = 1 - 2 * bit
                c[k] += bit
    return w, c


def _immediate(fmt: str) -> torch.Tensor:
    """A (32, 32) selection matrix from instruction bits to immediate bits,
    with sign extension: row j says which instruction bit is immediate bit j."""
    source: dict[int, int] = {}
    if fmt == "I":
        source = {j: 20 + j for j in range(11)}
        sign = 11
    elif fmt == "S":
        source = {j: 7 + j for j in range(5)} | {j: 20 + j for j in range(5, 11)}
        sign = 11
    elif fmt == "B":
        source = {j: 7 + j for j in range(1, 5)} | {j: 20 + j for j in range(5, 11)} | {11: 7}
        sign = 12
    elif fmt == "U":
        source = {j: j for j in range(12, 32)}
        sign = 32
    elif fmt == "J":
        source = {j: 20 + j for j in range(1, 11)} | {11: 20} | {j: j for j in range(12, 20)}
        sign = 20
    source |= {j: 31 for j in range(sign, 32)}
    select = torch.zeros(32, 32, dtype=DTYPE)
    for out, bit in source.items():
        select[out, bit] = 1
    return select


def _route(values: list[str | None], names: tuple[str, ...]) -> torch.Tensor:
    """One row per instruction, one-hot over `names` (all zeros for None)."""
    route = torch.zeros(len(values), len(names), dtype=DTYPE)
    for k, value in enumerate(values):
        if value is not None:
            route[k, names.index(value)] = 1
    return route


def compile_weights() -> dict[str, torch.Tensor]:
    """Build every tensor the network uses."""
    ins = isa.INSTRUCTIONS
    decode_w, decode_c = _matcher([(i.mask, i.match) for i in ins], 32)
    index_w, index_c = _matcher([(0x1F, j) for j in range(32)], 5)

    sll = torch.zeros(32, 32, 32, dtype=DTYPE)  # [shift, out bit, in bit]
    srl = torch.zeros(32, 32, 32, dtype=DTYPE)
    sra_fill = torch.zeros(32, 32, dtype=DTYPE)  # [shift, out bit]: copies of the sign bit
    for s in range(32):
        for i in range(32):
            if i - s >= 0:
                sll[s, i, i - s] = 1
            if i + s < 32:
                srl[s, i, i + s] = 1
            else:
                sra_fill[s, i] = 1

    byte_bits = torch.tensor([[v >> j & 1 for j in range(8)] for v in range(256)], dtype=DTYPE)
    return {
        "decode_w": decode_w,
        "decode_c": decode_c,
        "index_w": index_w,
        "index_c": index_c,
        "imm_bits": torch.stack([_immediate(f) for f in isa.FORMATS]),
        "format_route": _route([i.format for i in ins], isa.FORMATS),
        "uses_imm": torch.tensor([float(i.uses_imm) for i in ins], dtype=DTYPE),
        "result_route": _route([i.result for i in ins], isa.RESULTS),
        "next_pc_route": _route([i.next_pc for i in ins], isa.NEXT_PC),
        "branch_route": _route([i.branch for i in ins], isa.BRANCHES),
        "exit_route": _route([i.exit for i in ins], isa.EXITS),
        "load_size": torch.tensor([float(i.load) for i in ins], dtype=DTYPE),
        "store_size": torch.tensor([float(i.store) for i in ins], dtype=DTYPE),
        "sll": sll,
        "srl": srl,
        "sra_fill": sra_fill,
        "byte_bits": byte_bits,
        "powers": torch.tensor([2.0**i for i in range(32)], dtype=DTYPE),
    }


def save(weights: dict[str, torch.Tensor], path: Path) -> None:
    save_file({name: t.contiguous() for name, t in weights.items()}, str(path))


def load(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))
