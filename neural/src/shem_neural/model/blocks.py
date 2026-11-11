"""Block types, registered by name, and the model they make up.

Every block reads the residual stream and adds its output to it, with its own
pre-norm. A block that remembers anything keeps it in a state the caller
owns, one per block per session: the host's to inspect, snapshot and restore.

- `attention`: grouped-query attention with rotary positions on all or part
  of each head, optional RMSNorm on queries and keys, and an optional sigmoid
  gate on its output. Its state is the key and value cache.
- `gated_deltanet`: the Gated DeltaNet token mixer, a causal depthwise
  convolution and then a per-head matrix memory written with the gated delta
  rule. Its state is the convolution's last inputs and the memory matrices,
  a fixed size however long the session.
- `mlp`: a SwiGLU feed-forward block, with no state.

RMSNorms take an `offset` added to their weights: 0 for the usual norm, 1
for zero-centred weights. Weights load in their stored type and compute in
`dtype`, float32 by default, on the CPU or a GPU (`device`, where `auto`
picks a GPU if there is one). TF32 is off on GPUs, so a prompt gets the same
reply each time on the same device.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .manifest import BlockSpec, Manifest


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, offset: float = 0.0) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (offset + weight)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Block:
    """A block's weights and parameters. Subclasses implement `forward`."""

    def __init__(self, params: dict, tensors: dict[str, torch.Tensor]):
        self.p = params
        self.t = tensors

    def init_state(self, batch: int):
        return None

    def forward(self, x: torch.Tensor, state, start: int) -> torch.Tensor:
        """Add the block's output to the stream `x` (batch, tokens, hidden),
        whose first token is at position `start`, updating `state`."""
        raise NotImplementedError


class Attention(Block):
    def init_state(self, batch):
        return {}

    def forward(self, x, state, start):
        p, t = self.p, self.t
        batch, length, _ = x.shape
        heads, kv_heads, head_dim = p["heads"], p["kv_heads"], p["head_dim"]
        offset, eps = p.get("norm_offset", 0.0), p["eps"]
        h = rmsnorm(x, t["norm"], eps, offset)

        q = h @ t["q"].T
        gate = None
        if p.get("output_gate"):
            q, gate = q.view(batch, length, heads, 2 * head_dim).chunk(2, dim=-1)
            gate = gate.reshape(batch, length, heads * head_dim)
        q = q.reshape(batch, length, heads, head_dim)
        k = (h @ t["k"].T).view(batch, length, kv_heads, head_dim)
        v = (h @ t["v"].T).view(batch, length, kv_heads, head_dim)
        if "q_norm" in t:
            q = rmsnorm(q, t["q_norm"], eps, offset)
            k = rmsnorm(k, t["k_norm"], eps, offset)

        # Rotary positions on the first `rotary_dim` dimensions of each head
        rotary = p.get("rotary_dim", head_dim)
        positions = torch.arange(start, start + length, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (p["rope_theta"] ** (torch.arange(0, rotary, 2, device=x.device).float() / rotary))
        freqs = positions[:, None] * inv_freq
        angles = torch.cat((freqs, freqs), dim=-1)
        cos, sin = angles.cos().to(x.dtype)[:, None, :], angles.sin().to(x.dtype)[:, None, :]

        def rope(z):
            turned = z[..., :rotary] * cos + rotate_half(z[..., :rotary]) * sin
            return torch.cat((turned, z[..., rotary:]), dim=-1)

        q, k = rope(q).transpose(1, 2), rope(k).transpose(1, 2)
        v = v.transpose(1, 2)
        if "k" in state:
            k = torch.cat((state["k"], k), dim=2)
            v = torch.cat((state["v"], v), dim=2)
        state["k"], state["v"] = k, v

        group = heads // kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = (q @ k.transpose(2, 3)) * head_dim**-0.5
        total = k.shape[2]
        future = (
            torch.arange(total, device=x.device)[None, :]
            > torch.arange(total - length, total, device=x.device)[:, None]
        )
        scores = scores.masked_fill(future, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, heads * head_dim)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return x + out @ t["o"].T


class GatedDeltaNet(Block):
    def init_state(self, batch):
        p = self.p
        conv_dim = 2 * p["k_heads"] * p["k_dim"] + p["v_heads"] * p["v_dim"]
        dtype, device = self.t["out"].dtype, self.t["out"].device
        return {
            "conv": torch.zeros(batch, conv_dim, p["conv_kernel"] - 1, dtype=dtype, device=device),
            "memory": torch.zeros(batch, p["v_heads"], p["k_dim"], p["v_dim"], dtype=dtype, device=device),
        }

    def forward(self, x, state, start):
        p, t = self.p, self.t
        batch, length, _ = x.shape
        k_heads, v_heads, k_dim, v_dim = p["k_heads"], p["v_heads"], p["k_dim"], p["v_dim"]
        h = rmsnorm(x, t["norm"], p["eps"], p.get("norm_offset", 0.0))

        # Causal depthwise convolution over the projected queries, keys and values
        mixed = torch.cat((state["conv"], (h @ t["in_qkv"].T).transpose(1, 2)), dim=-1)
        state["conv"] = mixed[:, :, -(p["conv_kernel"] - 1) :]
        mixed = F.silu(F.conv1d(mixed, t["conv"], groups=mixed.shape[1])).transpose(1, 2)
        q, k, v = mixed.split([k_heads * k_dim, k_heads * k_dim, v_heads * v_dim], dim=-1)
        q = q.reshape(batch, length, k_heads, k_dim)
        k = k.reshape(batch, length, k_heads, k_dim)
        v = v.reshape(batch, length, v_heads, v_dim)
        if v_heads > k_heads:
            q = q.repeat_interleave(v_heads // k_heads, dim=2)
            k = k.repeat_interleave(v_heads // k_heads, dim=2)
        q = l2norm(q) / k_dim**0.5
        k = l2norm(k)

        beta = torch.sigmoid(h @ t["in_b"].T)
        decay = torch.exp(-torch.exp(t["A_log"]) * F.softplus(h @ t["in_a"].T + t["dt_bias"]))

        # The gated delta rule, one token at a time: decay the memory, then
        # write the value's correction at the key, then read at the query
        memory = state["memory"]
        outputs = []
        for i in range(length):
            memory = memory * decay[:, i, :, None, None]
            k_i = k[:, i]
            recalled = torch.einsum("bhk,bhkv->bhv", k_i, memory)
            delta = (v[:, i] - recalled) * beta[:, i, :, None]
            memory = memory + k_i[..., None] * delta[..., None, :]
            outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, i], memory))
        state["memory"] = memory
        out = torch.stack(outputs, dim=1)

        z = (h @ t["in_z"].T).view(batch, length, v_heads, v_dim)
        out = rmsnorm(out, t["out_norm"], p["eps"]) * F.silu(z)
        return x + out.reshape(batch, length, v_heads * v_dim) @ t["out"].T


class MLP(Block):
    def forward(self, x, state, start):
        t = self.t
        h = rmsnorm(x, t["norm"], self.p["eps"], self.p.get("norm_offset", 0.0))
        return x + (F.silu(h @ t["gate"].T) * (h @ t["up"].T)) @ t["down"].T


BLOCKS: dict[str, type[Block]] = {
    "attention": Attention,
    "gated_deltanet": GatedDeltaNet,
    "mlp": MLP,
}


def resolve_device(device: str) -> torch.device:
    """`auto` is a GPU when there is one, and the CPU otherwise."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class Model:
    """A model loaded from its manifest's directory, ready to run sessions."""

    def __init__(self, directory: Path, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        self.directory = Path(directory)
        self.manifest = Manifest.load(self.directory)
        self.dtype = dtype
        self.device = resolve_device(device)
        device = str(self.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        def tensors(spec: BlockSpec) -> dict[str, torch.Tensor]:
            loaded = load_file(str(self.directory / spec.file), device=device)
            return {name: tensor.to(dtype) for name, tensor in loaded.items()}

        m = self.manifest
        self.embed = tensors(m.embed)["weight"]
        self.blocks: list[Block] = []
        for spec in m.blocks:
            if spec.type not in BLOCKS:
                raise ValueError(f"unknown block type {spec.type!r}: this runtime has {sorted(BLOCKS)}")
            self.blocks.append(BLOCKS[spec.type](spec.params, tensors(spec)))
        head = tensors(m.head) if m.head.file else {}
        self.head_norm = head["norm"]
        self.head_weight = self.embed if m.head.params.get("tied") else head["weight"]
        self.head_params = m.head.params

    def init_state(self, batch: int = 1) -> list:
        return [block.init_state(batch) for block in self.blocks]

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, state: list, start: int) -> torch.Tensor:
        """Next-token logits (batch, tokens, vocabulary) for `tokens`, which
        continue a session at position `start` whose blocks' state is `state`."""
        x = self.embed[tokens.to(self.device)]
        for block, block_state in zip(self.blocks, state, strict=True):
            x = block.forward(x, block_state, start)
        p = self.head_params
        return rmsnorm(x, self.head_norm, p["eps"], p.get("norm_offset", 0.0)) @ self.head_weight.T
