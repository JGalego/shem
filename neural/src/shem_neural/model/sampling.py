"""Sampling a reply's tokens, seeded, so that a sampled reply can be repeated.

The driver decodes greedily unless given a temperature. Then each token is
drawn from the model's distribution (its logits over the temperature, cut to
the `top_k` likeliest and then to the smallest set whose probabilities add to
`top_p`) with a number from a generator that `seed` starts. The generator
lives on the CPU and draws one number a token, so the same seed makes the
same draws on any device, and a receipt that records the settings and the
seed can be checked by drawing again.

A drawn token's margin says how firmly it was drawn: how far the draw was,
in probability, from the edge of the token's share of the distribution. A
draw near an edge may land on the next token if another device rounds the
logits differently, as a close call in greedy decoding may.
"""

import secrets
from dataclasses import dataclass, field

import torch

#: The margin, in probability, under which another device may draw another token, by the type the model computes in.
#: Devices' logits part by under the greedy margins' `CLOSE` (see `receipt`), and a logit off by that much moves
#: no edge of the distribution's cumulative sums by more than it: the same numbers, in probability. A flat
#: distribution, whose shares are all thinner than that, is close everywhere, as a draw from it should be.
CLOSE = {
    "float32": 2e-3,
    "bfloat16": 0.5,
    "float32+int8": 0.5,
    "bfloat16+int8": 0.5,
    "float32+int4": 0.5,
    "bfloat16+int4": 0.5,
}


@dataclass
class Sampler:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    generator: torch.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be above 0 and at most 1")
        if self.top_k < 0:
            raise ValueError("top_k must not be negative")
        if self.seed is None:
            self.seed = secrets.randbits(62)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    def settings(self) -> dict:
        """What a receipt records to draw again."""
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "seed": self.seed}

    def draw(self, logits: torch.Tensor) -> tuple[int, float]:
        """The token drawn from `logits` (a vocabulary's worth), and its margin."""
        z = logits.detach().float().cpu() / self.temperature
        if self.top_k:
            keep = z.topk(min(self.top_k, z.numel())).values[-1]
            z = z.masked_fill(z < keep, float("-inf"))
        probabilities, order = z.softmax(-1).double().sort(descending=True)
        if self.top_p < 1:
            before = probabilities.cumsum(-1) - probabilities
            probabilities = probabilities.masked_fill(before >= self.top_p, 0.0)
        total = probabilities.sum()
        edges = probabilities.cumsum(-1)
        u = float(torch.rand((), generator=self.generator, dtype=torch.float64)) * float(total)
        at = min(int((edges <= u).sum()), probabilities.numel() - 1)
        low = float(edges[at - 1]) if at else 0.0
        high = float(edges[at])
        return int(order[at]), min(u - low, high - u) / float(total)
