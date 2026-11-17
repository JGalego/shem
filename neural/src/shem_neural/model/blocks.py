"""Residual blocks and model loading.

Blocks add their output to the residual stream. State is caller-owned, one
entry per block per session:

- `attention`: key/value caches, growing with the token count.
- `gated_deltanet`: fixed-size convolution history and matrix memory.
- `mamba`: fixed-size convolution history and selective-scan state.
- `mlp`: stateless gated feed-forward (SwiGLU or GeGLU).
- `moe`: stateless mixture of gated feed-forward experts, a few per token.

An attention or MLP block normalises its input when it has a `norm` and its
output when it has a `post_norm`: Llama has the first, OLMo 2 the second and
Gemma both. A projection `x` adds its bias `x_bias` when the block has one.
A block's `residual_scale` scales what it adds to the stream (Granite).
RMSNorm's weight offset is 0 for ordinary weights and 1 for zero-centred
weights. Computation defaults to float32; CUDA execution disables TF32.
"""

import hashlib
import json
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
    out = linear(h, t[name])
    bias = t.get(f"{name}_bias")
    return out if bias is None else out + bias


def softcap(x: torch.Tensor, cap: float | None) -> torch.Tensor:
    """`x` squashed smoothly into (-cap, cap), as Gemma 2 bounds its scores."""
    return x if not cap else torch.tanh(x / cap) * cap


class QWeight:
    """A linear layer's weight as int8 with a scale for each output row:
    a quarter of float32's memory, and half of bfloat16's. It is applied
    with PyTorch's int8 kernel where there is one, and otherwise dequantized
    for the one matrix being applied, never for the whole model."""

    __slots__ = ("q", "scale", "shape", "dtype", "device")

    def __init__(self, weight: torch.Tensor, dtype: torch.dtype):
        w = weight.float()
        self.scale = (w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127).squeeze(-1)
        self.q = (w / self.scale[..., None]).round().clamp(-127, 127).to(torch.int8)
        self.shape, self.dtype, self.device = weight.shape, dtype, self.q.device

    def __getitem__(self, index) -> "QWeight":
        """One expert of a stack of them."""
        part = object.__new__(QWeight)
        part.q, part.scale, part.dtype, part.device = self.q[index], self.scale[index], self.dtype, self.device
        part.shape = part.q.shape
        return part

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        """`h` times the transposed weight, as `F.linear` does."""
        if hasattr(torch, "_weight_int8pack_mm") and h.device.type == "cpu" and self.q.dim() == 2:
            try:
                flat = h.reshape(-1, h.shape[-1])
                out = torch._weight_int8pack_mm(flat.contiguous(), self.q, self.scale.to(flat.dtype))
                return out.reshape(*h.shape[:-1], -1)
            except (RuntimeError, TypeError):
                pass  # a build or type without the kernel
        return F.linear(h, (self.q.to(h.dtype) * self.scale.to(h.dtype)[..., None]))


def linear(h: torch.Tensor, weight) -> torch.Tensor:
    """`h` through the weight, quantized or not."""
    return weight.apply(h) if isinstance(weight, QWeight) else F.linear(h, weight)


#: The weights quantization applies to: those of each block's projections, not norms, biases or recurrent parameters
LINEAR = {
    "attention": {"q", "k", "v", "qkv", "o"},
    "gated_deltanet": {"in_qkv", "in_z", "out"},
    "mlp": {"gate", "up", "gate_up", "down"},
    "moe": {"gate_up", "down", "shared_gate", "shared_up", "shared_down"},
}


def quantizable(kind: str, name: str) -> bool:
    return name in LINEAR.get(kind, ()) or (
        kind == "moe" and name.split(".", 1)[0] in {"gate", "up", "down"} and "." in name
    )


#: The gates of `mlp` blocks: SwiGLU's SiLU, and GeGLU's GELU, exact or tanh-approximated
ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
}


class Block:
    """A block's weights and parameters. Subclasses implement `forward`."""

    #: The version of what the block computes and keeps as state. A saved
    #: session records it for each type of block, and a runtime whose blocks
    #: have another reads the session's tokens again rather than trust its
    #: state. Bump it when a session saved before a change would resume
    #: otherwise than a cold start, as `test_sessions` checks: even a
    #: stateless block changes what the blocks after it keep.
    VERSION = 1

    def __init__(self, params: dict, tensors: dict[str, torch.Tensor]):
        self.p = params
        self.t = tensors

    def init_state(self, batch: int):
        return None

    def forward(self, x: torch.Tensor, state, start: int) -> torch.Tensor:
        """Add the block's output to the stream `x` (batch, tokens, hidden),
        whose first token is at position `start`, updating `state`."""
        raise NotImplementedError

    def forward_batch(self, x: torch.Tensor, states: list, starts: list[int]) -> torch.Tensor:
        """One token of each of several sessions, `x` (sessions, 1, hidden),
        whose positions are `starts` and whose states are `states`. This
        steps them one after the other; a block that can do it at once says so."""
        return torch.cat([self.forward(x[i : i + 1], states[i], starts[i]) for i in range(len(states))], dim=0)


class Attention(Block):
    def __init__(self, params, tensors):
        super().__init__(params, tensors)
        #: Rotary frequencies, by device and whether the pass reaches the long ones: they hold for every pass alike
        self._freqs: dict = {}

    def init_state(self, batch):
        return {}

    def _rotary(self, x, start, length):
        """The cosines and sines for positions `start` to `start + length`."""
        p = self.p
        scaling = p.get("rope_scaling") or {}
        long = scaling.get("type") == "longrope" and start + length > scaling.get("original_max_position_embeddings", 0)
        key = (x.device, long)
        if key not in self._freqs:
            self._freqs[key] = rope_frequencies(p, x.device, start + length)
        inv_freq, scale = self._freqs[key]
        positions = torch.arange(start, start + length, dtype=torch.float32, device=x.device)
        freqs = positions[:, None] * inv_freq
        angles = torch.cat((freqs, freqs), dim=-1)
        return (angles.cos() * scale).to(x.dtype)[:, None, :], (angles.sin() * scale).to(x.dtype)[:, None, :]

    @staticmethod
    def _append(state, k, v):
        """Add the new keys and values to the cache, which are views of buffers
        with room to grow, so that a token does not copy the whole cache. The
        buffers are state a session does not save, as its names start with an
        underscore; `state["k"]` and `state["v"]` are what it keeps."""
        length = k.shape[2]
        n = state["k"].shape[2] if "k" in state else 0
        buffers = state.get("_buffers")
        fits = (
            buffers is not None
            and buffers[0].shape[2] >= n + length
            and (n == 0 or buffers[0].data_ptr() == state["k"].data_ptr())
        )
        if not fits:
            room = n + length + max(256, (n + length) // 2)
            buffers = (
                k.new_empty(k.shape[0], k.shape[1], room, k.shape[3]),
                v.new_empty(v.shape[0], v.shape[1], room, v.shape[3]),
            )
            if n:
                buffers[0][:, :, :n], buffers[1][:, :, :n] = state["k"], state["v"]
            state["_buffers"] = buffers
        buffers[0][:, :, n : n + length], buffers[1][:, :, n : n + length] = k, v
        state["k"], state["v"] = buffers[0][:, :, : n + length], buffers[1][:, :, : n + length]

    def _split(self, x):
        """The queries, keys and values of `x` (batch, tokens, hidden), each
        by head and before their rotary positions, and the output gate if
        the block has one."""
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
        return q, k, v, gate

    def _turn(self, q, k, cos, sin):
        """Rotary positions on the first `rotary_dim` dimensions of each head,
        if any: a block with none has no positions but the mask (SmolLM3)."""
        rotary = self.p.get("rotary_dim", self.p["head_dim"])
        if not rotary:
            return q, k

        def rope(z):
            turned = z[..., :rotary] * cos + rotate_half(z[..., :rotary]) * sin
            return torch.cat((turned, z[..., rotary:]), dim=-1)

        return rope(q), rope(k)

    def _attend(self, q, k, v, mask, group):
        """Attention of `q` over the keys and values, which `mask` (True: see)
        limits, if it does; `group` queries share each key."""
        p = self.p
        scale = p.get("scale", p["head_dim"] ** -0.5)
        if p.get("softcap"):
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)
            scores = softcap((q @ k.transpose(2, 3)) * scale, p["softcap"])
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            return scores.softmax(-1) @ v
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=group > 1)

    def _finish(self, x, out, gate):
        """The block's output from the attention's, added to the stream `x`."""
        p, t = self.p, self.t
        batch, length, _ = x.shape
        out = out.transpose(1, 2).reshape(batch, length, p["heads"] * p["head_dim"])
        if gate is not None:
            out = out * torch.sigmoid(gate)
        out = project(out, t, "o")
        if "post_norm" in t:
            out = rmsnorm(out, t["post_norm"], p["eps"], p.get("norm_offset", 0.0))
        return x + out * p.get("residual_scale", 1.0)

    def forward(self, x, state, start):
        p = self.p
        batch, length, _ = x.shape
        q, k, v, gate = self._split(x)
        if p.get("rotary_dim", p["head_dim"]):
            cos, sin = self._rotary(x, start, length)
            q, k = self._turn(q, k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        self._append(state, k, v)
        k, v = state["k"], state["v"]

        group = p["heads"] // p["kv_heads"]
        total = k.shape[2]
        window = p.get("sliding_window")
        if length == 1 and window and total > window:
            # The one query sees only the last `window` keys
            k, v, total = k[:, :, -window:], v[:, :, -window:], window
        # Each query sees the keys up to its own, or with a sliding window,
        # only the last `window` of them. A single query sees all it has
        mask = None
        if length > 1:
            keys = torch.arange(total, device=x.device)[None, :]
            queries = torch.arange(total - length, total, device=x.device)[:, None]
            hidden = keys > queries
            if window:
                hidden |= keys <= queries - window
            mask = ~hidden
        return self._finish(x, self._attend(q, k, v, mask, group), gate)

    def forward_batch(self, x, states, starts):
        """A token of each of several sessions at once: their queries, keys
        and values are projected together, and each attends over its own
        cache, padded to the longest and masked, so that the weights are read
        once for them all."""
        p = self.p
        if (p.get("rope_scaling") or {}).get("type") == "longrope":
            return super().forward_batch(x, states, starts)  # its frequencies depend on how far each has read
        count = len(states)
        q, k, v, gate = self._split(x)
        if p.get("rotary_dim", p["head_dim"]):
            key = (x.device, False)
            if key not in self._freqs:
                self._freqs[key] = rope_frequencies(p, x.device, max(starts) + 1)
            inv_freq, scale = self._freqs[key]
            freqs = torch.tensor(starts, dtype=torch.float32, device=x.device)[:, None] * inv_freq
            angles = torch.cat((freqs, freqs), dim=-1)
            cos = (angles.cos() * scale).to(x.dtype)[:, None, None, :]
            sin = (angles.sin() * scale).to(x.dtype)[:, None, None, :]
            q, k = self._turn(q, k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        for i, state in enumerate(states):
            self._append(state, k[i : i + 1], v[i : i + 1])
        window = p.get("sliding_window")
        lengths = [state["k"].shape[2] for state in states]
        # The keys each session sees: all of them, or the last `window`
        seen = [min(n, window) if window else n for n in lengths]
        longest = max(seen)
        keys = k.new_zeros(count, p["kv_heads"], longest, p["head_dim"])
        values = v.new_zeros(count, p["kv_heads"], longest, p["head_dim"])
        mask = torch.zeros(count, 1, 1, longest, dtype=torch.bool, device=x.device)
        for i, state in enumerate(states):
            keys[i, :, : seen[i]] = state["k"][0, :, lengths[i] - seen[i] :]
            values[i, :, : seen[i]] = state["v"][0, :, lengths[i] - seen[i] :]
            mask[i, :, :, : seen[i]] = True
        out = self._attend(q, keys, values, mask, p["heads"] // p["kv_heads"])
        return self._finish(x, out, gate)


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
        mixed = torch.cat((state["conv"], linear(h, t["in_qkv"]).transpose(1, 2)), dim=-1)
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

        z = linear(h, t["in_z"]).view(batch, length, v_heads, v_dim)
        out = rmsnorm(out, t["out_norm"], p["eps"]) * F.silu(z)
        return x + linear(out.reshape(batch, length, v_heads * v_dim), t["out"])


class MLP(Block):
    def forward_batch(self, x, states, starts):
        return self.forward(x, None, 0)  # no state, and no token depends on another

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
        return x + out * p.get("residual_scale", 1.0)


class MoE(Block):
    """A mixture of experts: a router picks `top_k` of `experts` gated
    feed-forward networks for each token and weighs their outputs, either
    from a softmax over all the router's logits, renormalised over those
    picked when `normalize` (Mixtral, Qwen), or from a softmax over the
    picked logits alone (`routing: topk_softmax`, GraniteMoe). Experts are
    tensors of their own (`gate.E`, `up.E`, `down.E`) or stacked (`gate_up`
    and `down`). A shared expert (`shared_*`), weighed by a sigmoid gate,
    adds to every token's output (Qwen2-MoE)."""

    def forward_batch(self, x, states, starts):
        return self.forward(x, None, 0)  # no state, and each token is routed on its own

    def forward(self, x, state, start):
        p, t = self.p, self.t
        offset, eps = p.get("norm_offset", 0.0), p["eps"]
        h = rmsnorm(x, t["norm"], eps, offset) if "norm" in t else x
        flat = h.reshape(-1, h.shape[-1])
        logits = F.linear(flat, t["router"]).float()
        if p.get("routing") == "topk_softmax":
            top, index = logits.topk(p["top_k"], dim=-1)
            weights = top.softmax(dim=-1)
        else:
            weights, index = logits.softmax(dim=-1).topk(p["top_k"], dim=-1)
            if p.get("normalize"):
                weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(flat.dtype)
        act = ACTIVATIONS[p.get("activation", "silu")]
        out = torch.zeros_like(flat)
        stacked = "gate_up" in t
        for expert in range(p["experts"]):
            slot, token = torch.where(index.T == expert)
            if token.numel() == 0:
                continue
            chosen = flat[token]
            if stacked:
                gate, up = linear(chosen, t["gate_up"][expert]).chunk(2, dim=-1)
                down = t["down"][expert]
            else:
                gate, up = linear(chosen, t[f"gate.{expert}"]), linear(chosen, t[f"up.{expert}"])
                down = t[f"down.{expert}"]
            y = linear(act(gate) * up, down) * weights[token, slot, None]
            out.index_add_(0, token, y)
        if "shared_up" in t:
            shared = linear(act(linear(flat, t["shared_gate"])) * linear(flat, t["shared_up"]), t["shared_down"])
            out = out + torch.sigmoid(F.linear(flat, t["shared_router"])) * shared
        out = out.view_as(h)
        if "post_norm" in t:
            out = rmsnorm(out, t["post_norm"], eps, offset)
        return x + out * p.get("residual_scale", 1.0)


class Mamba(Block):
    """Mamba: a projection into a stream and its gate, a causal depthwise
    convolution over the stream, then a selective scan, whose state is a
    `state_size` vector per channel, decayed and written at each token with
    step sizes, inputs and outputs that depend on the token."""

    def init_state(self, batch):
        p = self.p
        dtype, device = self.t["out"].dtype, self.t["out"].device
        return {
            "conv": torch.zeros(batch, p["inner"], p["conv_kernel"] - 1, dtype=dtype, device=device),
            "ssm": torch.zeros(batch, p["inner"], p["state_size"], dtype=torch.float32, device=device),
        }

    def forward(self, x, state, start):
        p, t = self.p, self.t
        h = rmsnorm(x, t["norm"], p["eps"], p.get("norm_offset", 0.0))
        stream, gate = project(h, t, "in").transpose(1, 2).chunk(2, dim=1)
        mixed = torch.cat((state["conv"], stream), dim=-1)
        state["conv"] = mixed[:, :, -(p["conv_kernel"] - 1) :]
        u = F.silu(F.conv1d(mixed, t["conv"], t.get("conv_bias"), groups=p["inner"]))
        rank, n = p["time_step_rank"], p["state_size"]
        step, b, c = (u.transpose(1, 2) @ t["x"].T).split([rank, n, n], dim=-1)
        delta = F.softplus(t["dt"] @ step.transpose(1, 2) + t["dt_bias"][..., None].float())
        a = -torch.exp(t["A_log"].float())
        decay = torch.exp(a[None, :, None, :] * delta[:, :, :, None])
        written = delta[:, :, :, None] * b[:, None, :, :].float() * u[:, :, :, None].float()
        ssm = state["ssm"]
        outputs = []
        for i in range(u.shape[-1]):
            ssm = decay[:, :, i, :] * ssm + written[:, :, i, :]
            outputs.append(torch.matmul(ssm.to(u.dtype), c[:, i, :].unsqueeze(-1))[:, :, 0])
        state["ssm"] = ssm
        y = torch.stack(outputs, dim=-1) + u * t["D"][None, :, None]
        y = y * F.silu(gate)
        return x + project(y.transpose(1, 2), t, "out")


BLOCKS: dict[str, type[Block]] = {
    "attention": Attention,
    "gated_deltanet": GatedDeltaNet,
    "mamba": Mamba,
    "mlp": MLP,
    "moe": MoE,
}


def resolve_device(device: str) -> torch.device:
    """`auto` is a GPU when there is one, and the CPU otherwise."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class Adapter:
    """A LoRA adapter, as PEFT saves one: low-rank matrices for some of a
    model's weights, `adapter_model.safetensors` and `adapter_config.json` in a
    directory. Merged into the weights as the model loads, it costs nothing
    to run, and the image that holds it holds a few megabytes beside the
    model's gigabytes, which every adapter's image shares."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        config = json.loads((self.directory / "adapter_config.json").read_text())
        if config.get("peft_type", "LORA") != "LORA":
            raise ValueError(f"{directory}: shem merges LoRA adapters, not {config['peft_type']}")
        for option in ("use_dora", "fan_in_fan_out"):
            if config.get(option):
                raise ValueError(f"{directory}: shem does not merge adapters made with {option}")
        if config.get("modules_to_save"):
            raise ValueError(f"{directory}: shem does not merge adapters that train whole modules (modules_to_save)")
        rank = config["r"]
        self.scale = config["lora_alpha"] / (rank**0.5 if config.get("use_rslora") else rank)
        pairs: dict[str, dict] = {}
        for key, tensor in load_file(str(self.directory / "adapter_model.safetensors")).items():
            module, _, rest = key.removeprefix("base_model.model.").partition(".lora_")
            kind, _, _ = rest.partition(".")
            if kind not in ("A", "B"):
                raise ValueError(f"{directory}: no meaning for the adapter's tensor {key}")
            pairs.setdefault(module + ".weight", {})[kind] = tensor
        if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
            raise ValueError(f"{directory}: the adapter has no complete pair of low-rank matrices")
        self.pairs = pairs
        self.digest = hashlib.sha256((self.directory / "adapter_model.safetensors").read_bytes()).hexdigest()[:16]

    @property
    def name(self) -> str:
        return f"{self.directory.name}@{self.digest}"

    def merge(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """`weight`, the checkpoint's tensor `name`, with the adapter's change to it, if it has one."""
        pair = self.pairs.get(name)
        if pair is None:
            return weight
        delta = (pair["B"].float() @ pair["A"].float()) * self.scale
        if delta.shape != weight.shape:
            raise ValueError(f"{self.directory}: the adapter's change to {name} is {tuple(delta.shape)}, not its shape")
        return (weight.float() + delta).to(weight.dtype)

    def unused(self, names: set[str]) -> list[str]:
        """The weights it changes that the model has none of."""
        return sorted(set(self.pairs) - names)


class Model:
    """A model loaded from its manifest's directory, ready to run sessions."""

    def __init__(
        self,
        directory: Path,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        quantize: str | None = None,
        verify: bool = False,
        adapters: list[Path] | None = None,
    ):
        """The model in `directory`. With `verify`, its tensors are checked
        against the fingerprint its manifest records, as they are read: a
        model that several containers share must be the one it says it is."""
        if quantize not in (None, "int8"):
            raise ValueError(f"cannot quantize to {quantize!r}: shem has int8")
        self.directory = Path(directory)
        self.manifest = Manifest.load(self.directory)
        self.dtype = dtype
        self.quantize = quantize
        #: LoRA adapters merged into the weights as they load, in order
        self.adapters = [Adapter(path) for path in adapters or []]
        used: set[str] = set()
        self.device = resolve_device(device)
        device = str(self.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        m = self.manifest
        weight_map = m.weight_map(self.directory)
        digest = hashlib.sha256() if verify and m.fingerprint else None

        def tensors(spec: BlockSpec) -> dict[str, torch.Tensor]:
            if spec.file is not None:
                # A v1 manifest: the block's file, under its type's names
                loaded = load_file(str(self.directory / spec.file), device=device)
            else:
                # The block's tensors from whichever files hold them
                by_file = defaultdict(list)
                for local, name in spec.tensors.items():
                    by_file[weight_map[name]].append((local, name))
                loaded, named = {}, {}
                for file, names in by_file.items():
                    with safe_open(str(self.directory / file), "pt", device=device) as handle:
                        for local, name in names:
                            loaded[local] = named[name] = handle.get_tensor(name)
                if digest is not None:
                    # As the importer did when it made the fingerprint
                    digest.update(json.dumps([spec.type, spec.params, spec.tensors], sort_keys=True).encode())
                    for name in sorted(set(spec.tensors.values())):
                        tensor = named[name].cpu().contiguous()
                        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
                        digest.update(tensor.view(torch.uint8).numpy().tobytes() if tensor.numel() else b"")
            for adapter in self.adapters:
                for local, name in spec.tensors.items():
                    if name in adapter.pairs and local in loaded:
                        loaded[local] = adapter.merge(name, loaded[local])
                        used.add(name)
            if quantize:
                # Quantized as loaded, from the checkpoint's own precision
                return {
                    name: QWeight(tensor, dtype) if quantizable(spec.type, name) else tensor.to(dtype)
                    for name, tensor in loaded.items()
                }
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
        for adapter in self.adapters:
            if missing := adapter.unused(used):
                raise ValueError(
                    f"{adapter.directory}: the adapter changes {len(missing)} weights this model has none of, "
                    f"such as {missing[0]}: is it for another model?"
                )
        if digest is not None and "sha256:" + digest.hexdigest() != m.fingerprint:
            raise ValueError(f"{self.directory}: the weights are not the ones the manifest's fingerprint names")
        self.head_norm = head["norm"]
        self.head_weight = self.embed if m.head.params.get("tied") else head["weight"]
        self.head_params = m.head.params

    def init_state(self, batch: int = 1) -> list:
        return [block.init_state(batch) for block in self.blocks]

    @property
    def versions(self) -> dict[str, int]:
        """The `VERSION` of each type of block the model has."""
        return {spec.type: BLOCKS[spec.type].VERSION for spec in self.manifest.blocks}

    @property
    def precision(self) -> str:
        """The type the model computes in, as ModelPack names it, such as `float32`."""
        name = str(self.dtype).removeprefix("torch.")
        return f"{name}+{self.quantize}" if self.quantize else name

    @property
    def adapter_names(self) -> list[str]:
        """The adapters merged in, by name and a digest of their weights."""
        return [adapter.name for adapter in self.adapters]

    @torch.no_grad()
    def forward_batch(self, tokens: list[int], states: list[list], starts: list[int]) -> torch.Tensor:
        """Next-token logits (sessions, vocabulary) for one more token of each
        of several sessions: `tokens`, which continue sessions at positions
        `starts`, whose blocks' states are `states`. The sessions' weights are
        read once, and each session's state is updated as `forward` updates it."""
        x = self.embed[torch.tensor(tokens, device=self.device)][:, None, :]
        if self.embed_scale:
            x = x * self.embed_scale
        for index, block in enumerate(self.blocks):
            x = block.forward_batch(x, [state[index] for state in states], starts)
        p = self.head_params
        logits = rmsnorm(x[:, 0], self.head_norm, p["eps"], p.get("norm_offset", 0.0)) @ self.head_weight.T
        if p.get("logit_divisor"):
            logits = logits / p["logit_divisor"]
        return softcap(logits, p.get("softcap"))

    @torch.no_grad()
    def hidden(self, tokens: torch.Tensor, state: list, start: int) -> torch.Tensor:
        """What the model has made of `tokens` (batch, tokens, hidden), after
        its last block and its final norm: what the head reads, and what an
        embedding is taken from. They continue a session at position `start`
        whose blocks' state is `state`."""
        x = self.embed[tokens.to(self.device)]
        if self.embed_scale:
            x = x * self.embed_scale
        for block, block_state in zip(self.blocks, state, strict=True):
            x = block.forward(x, block_state, start)
        p = self.head_params
        return rmsnorm(x, self.head_norm, p["eps"], p.get("norm_offset", 0.0))

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, state: list, start: int) -> torch.Tensor:
        """Next-token logits (batch, tokens, vocabulary) for `tokens`, which
        continue a session at position `start` whose blocks' state is `state`."""
        p = self.head_params
        logits = self.hidden(tokens, state, start) @ self.head_weight.T
        if p.get("logit_divisor"):
            logits = logits / p["logit_divisor"]
        return softcap(logits, p.get("softcap"))
