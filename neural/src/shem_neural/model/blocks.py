"""Residual blocks and model loading.

Blocks add their output to the residual stream. State is caller-owned, one
entry per block per session:

- `attention`: key/value caches, growing with the token count.
- `gated_deltanet`: fixed-size convolution history and matrix memory.
- `mlp`: stateless gated feed-forward (SwiGLU or GeGLU).

An attention or MLP block normalises its input when it has a `norm` and its
output when it has a `post_norm`: Llama has the first, OLMo 2 the second and
Gemma both. A projection `x` adds its bias `x_bias` when the block has one.
RMSNorm's weight offset is 0 for ordinary weights and 1 for zero-centred
weights. Computation defaults to float32; CUDA execution disables TF32.
"""

import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from .manifest import BlockSpec, Manifest


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, offset: float = 0.0) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (offset + weight)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_frequencies(p: dict, device: torch.device, end: int) -> tuple[torch.Tensor, float]:
    """The rotary frequencies of an attention block, one per pair of its
    first `rotary_dim` dimensions, for a pass whose last position is
    `end - 1`, and the factor its cosines and sines are scaled by.

    `rope_scaling` stretches them for contexts longer than the model was
    trained on: `linear` divides them all by the factor; `llama3` divides
    only the slow ones and blends around them; `yarn` blends the same way by
    dimension; `longrope` multiplies each wavelength by a factor of its own,
    from a long list once a pass reaches past the original context and a
    short list before."""
    rotary = p.get("rotary_dim", p["head_dim"])
    theta = p["rope_theta"]
    exponents = torch.arange(0, rotary, 2, device=device).float() / rotary
    inv_freq = 1.0 / (theta**exponents)
    scaling = p.get("rope_scaling")
    if scaling is None:
        return inv_freq, 1.0
    kind, context = scaling["type"], scaling.get("original_max_position_embeddings")
    if kind == "linear":
        return inv_freq / scaling["factor"], 1.0
    if kind == "llama3":
        factor, low, high = scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"]
        wavelength = 2 * math.pi / inv_freq
        scaled = torch.where(wavelength > context / low, inv_freq / factor, inv_freq)
        smooth = (context / wavelength - low) / (high - low)
        blended = (1 - smooth) * scaled / factor + smooth * scaled
        between = (wavelength >= context / high) & (wavelength <= context / low)
        return torch.where(between, blended, scaled), 1.0
    if kind == "yarn":

        def dimension(rotations):
            return rotary * math.log(context / (rotations * 2 * math.pi)) / (2 * math.log(theta))

        low, high = dimension(scaling["beta_fast"]), dimension(scaling["beta_slow"])
        if scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, rotary - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(rotary // 2, dtype=torch.float32, device=device) - low) / (high - low)).clamp(0, 1)
        positions = theta**exponents
        extrapolated = 1 - ramp
        inv_freq = (1.0 / (scaling["factor"] * positions)) * (1 - extrapolated) + (1.0 / positions) * extrapolated
        return inv_freq, scaling["attention_factor"]
    if kind == "longrope":
        factors = scaling["long_factor"] if end > context else scaling["short_factor"]
        factors = torch.tensor(factors, dtype=torch.float32, device=device)
        return 1.0 / (factors * theta**exponents), scaling["attention_factor"]
    raise ValueError(f"unknown RoPE scaling {kind!r}")


def project(h: torch.Tensor, t: dict, name: str) -> torch.Tensor:
    """`h` through the projection `name`, and its bias if it has one."""
    out = h @ t[name].T
    bias = t.get(f"{name}_bias")
    return out if bias is None else out + bias


def softcap(x: torch.Tensor, cap: float | None) -> torch.Tensor:
    """`x` squashed smoothly into (-cap, cap), as Gemma 2 bounds its scores."""
    return x if not cap else torch.tanh(x / cap) * cap


#: The gates of `mlp` blocks: SwiGLU's SiLU, and GeGLU's GELU, exact or tanh-approximated
ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
}


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
        h = rmsnorm(x, t["norm"], eps, offset) if "norm" in t else x

        gate = None
        if "qkv" in t:
            # One projection for queries, keys and values together (Phi-3)
            sizes = [heads * head_dim, kv_heads * head_dim, kv_heads * head_dim]
            q, k, v = project(h, t, "qkv").split(sizes, dim=-1)
        else:
            q, k, v = project(h, t, "q"), project(h, t, "k"), project(h, t, "v")
            if p.get("output_gate"):
                q, gate = q.view(batch, length, heads, 2 * head_dim).chunk(2, dim=-1)
                gate = gate.reshape(batch, length, heads * head_dim)
        per_head = p.get("qk_norm", "head") == "head"
        if "q_norm" in t and not per_head:
            # Norms over every head's queries and keys at once (OLMo 2)
            q, k = rmsnorm(q, t["q_norm"], eps, offset), rmsnorm(k, t["k_norm"], eps, offset)
        q = q.reshape(batch, length, heads, head_dim)
        k = k.reshape(batch, length, kv_heads, head_dim)
        v = v.reshape(batch, length, kv_heads, head_dim)
        if "q_norm" in t and per_head:
            q, k = rmsnorm(q, t["q_norm"], eps, offset), rmsnorm(k, t["k_norm"], eps, offset)

        # Rotary positions on the first `rotary_dim` dimensions of each head,
        # if any: a block with none has no positions but the mask (SmolLM3)
        rotary = p.get("rotary_dim", head_dim)
        if rotary:
            inv_freq, scale = rope_frequencies(p, x.device, start + length)
            positions = torch.arange(start, start + length, dtype=torch.float32, device=x.device)
            freqs = positions[:, None] * inv_freq
            angles = torch.cat((freqs, freqs), dim=-1)
            cos = (angles.cos() * scale).to(x.dtype)[:, None, :]
            sin = (angles.sin() * scale).to(x.dtype)[:, None, :]

            def rope(z):
                turned = z[..., :rotary] * cos + rotate_half(z[..., :rotary]) * sin
                return torch.cat((turned, z[..., rotary:]), dim=-1)

            q, k = rope(q), rope(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if "k" in state:
            k = torch.cat((state["k"], k), dim=2)
            v = torch.cat((state["v"], v), dim=2)
        state["k"], state["v"] = k, v

        group = heads // kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = softcap((q @ k.transpose(2, 3)) * p.get("scale", head_dim**-0.5), p.get("softcap"))
        total = k.shape[2]
        # Each query sees the keys up to its own, or with a sliding window,
        # only the last `sliding_window` of them
        keys = torch.arange(total, device=x.device)[None, :]
        queries = torch.arange(total - length, total, device=x.device)[:, None]
        hidden = keys > queries
        if p.get("sliding_window"):
            hidden |= keys <= queries - p["sliding_window"]
        scores = scores.masked_fill(hidden, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, heads * head_dim)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        out = project(out, t, "o")
        if "post_norm" in t:
            out = rmsnorm(out, t["post_norm"], eps, offset)
        return x + out


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
        p, t = self.p, self.t
        offset, eps = p.get("norm_offset", 0.0), p["eps"]
        h = rmsnorm(x, t["norm"], eps, offset) if "norm" in t else x
        if "gate_up" in t:
            # One projection for the gate and the values together (Phi-3)
            gate, up = project(h, t, "gate_up").chunk(2, dim=-1)
        else:
            gate, up = project(h, t, "gate"), project(h, t, "up")
        out = project(ACTIVATIONS[p.get("activation", "silu")](gate) * up, t, "down")
        if "post_norm" in t:
            out = rmsnorm(out, t["post_norm"], eps, offset)
        return x + out


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

        m = self.manifest
        weight_map = m.weight_map(self.directory)

        def tensors(spec: BlockSpec) -> dict[str, torch.Tensor]:
            if spec.file is not None:
                # A v1 manifest: the block's file, under its type's names
                loaded = load_file(str(self.directory / spec.file), device=device)
            else:
                # The block's tensors from whichever files hold them
                by_file = defaultdict(list)
                for local, name in spec.tensors.items():
                    by_file[weight_map[name]].append((local, name))
                loaded = {}
                for file, names in by_file.items():
                    with safe_open(str(self.directory / file), "pt", device=device) as handle:
                        for local, name in names:
                            loaded[local] = handle.get_tensor(name)
            return {name: tensor.to(dtype) for name, tensor in loaded.items()}

        self.embed = tensors(m.embed)["weight"]
        #: What embeddings are multiplied by before the first block: Gemma's square root of the width
        self.embed_scale = m.embed.params.get("scale")
        self.blocks: list[Block] = []
        for spec in m.blocks:
            if spec.type not in BLOCKS:
                raise ValueError(f"unknown block type {spec.type!r}: this runtime has {sorted(BLOCKS)}")
            self.blocks.append(BLOCKS[spec.type](spec.params, tensors(spec)))
        head = tensors(m.head)
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
        if self.embed_scale:
            x = x * self.embed_scale
        for block, block_state in zip(self.blocks, state, strict=True):
            x = block.forward(x, block_state, start)
        p = self.head_params
        logits = rmsnorm(x, self.head_norm, p["eps"], p.get("norm_offset", 0.0)) @ self.head_weight.T
        return softcap(logits, p.get("softcap"))
