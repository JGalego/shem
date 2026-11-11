"""The network: one forward pass executes one instruction for every guest in
the batch.

It is built only from operations a compiled transformer uses: products with
the compiled weight tensors (linear layers and one-hot attention reads),
element-wise sums and products (gating), and clamps, which are
piecewise-linear like ReLU. There is no data-dependent control flow in the
arithmetic. Every unit computes its result for every guest, and the decoded
instruction selects one, so guests running different instructions cost the
same as guests running the same one. That is what lets one batch carry many
unrelated guests.

Values are integers held exactly in float64. A 32-bit word is stored as bits
in registers and as four bytes in memory. Arithmetic is done on integer
values (linear in the bits) and turned back into bits by a stack of threshold
layers, `decompose`.

The register file and memory are delta-rule states with one-hot keys (see
`memory.py`). Every write is gated by `commit`, which is 0 when the
instruction exits to the host, so an exiting instruction changes nothing,
matching the reference vCPU.

Like memory, the deepest units have two executors. The dense one is the
network as described. The fused one computes the same function with integer
and indexing operations: bit decomposition by shifts instead of 35 threshold
layers, table lookups by index instead of one-hot products, and division by
integer division instead of 32 unrolled layers. Tests check that the two agree
unit by unit and over whole programs; the fused one is several times faster.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import isa
from .memory import DeltaMemory

DTYPE = torch.float64
TWO32 = 2.0**32
BYTE_PLACES = torch.tensor([1.0, 256.0, 65536.0, 16777216.0], dtype=DTYPE)
LEVELS = torch.arange(256, dtype=DTYPE)

# Exit reasons, in the order `step` reports them.
ECALL, EBREAK, ILLEGAL, FETCH_FAULT, MEMORY_FAULT = range(5)

_MUL = [isa.RESULTS.index(r) for r in ("mul", "mulh", "mulhsu", "mulhu")]
_DIV = [isa.RESULTS.index(r) for r in ("div", "divu", "rem", "remu")]


def clamp01(x: torch.Tensor) -> torch.Tensor:
    """relu(x) - relu(x - 1): for integers, 1 if x >= 1 and 0 if x <= 0."""
    return x.clamp(0, 1)


def decompose(x: torch.Tensor, n: int) -> torch.Tensor:
    """The bits of integers 0 <= x < 2**n, least significant first, computed by
    n threshold layers from the top bit down."""
    bits = []
    for i in range(n - 1, -1, -1):
        bit = clamp01(x - (2.0**i - 1))
        x = x - bit * 2.0**i
        bits.append(bit)
    return torch.stack(bits[::-1], dim=-1)


@dataclass
class State:
    """Everything that defines a batch of machines between instructions."""

    pc: torch.Tensor  # (B,)
    pc_low: torch.Tensor  # (B, 2): the low bits of pc, for the alignment check
    regs: torch.Tensor  # (B, 32, 32): one row of bits per register
    mem: DeltaMemory
    instret: torch.Tensor  # (B,)

    @classmethod
    def zeros(cls, batch: int, mem_size: int, dense: bool = False) -> "State":
        return cls(
            pc=torch.zeros(batch, dtype=DTYPE),
            pc_low=torch.zeros(batch, 2, dtype=DTYPE),
            regs=torch.zeros(batch, 32, 32, dtype=DTYPE),
            mem=DeltaMemory(batch, mem_size, dense),
            instret=torch.zeros(batch, dtype=DTYPE),
        )

    def clone(self) -> "State":
        return State(
            self.pc.clone(),
            self.pc_low.clone(),
            self.regs.clone(),
            self.mem.clone(),
            self.instret.clone(),
        )


class NeuralCpu:
    def __init__(self, weights: dict[str, torch.Tensor], fused: bool = True):
        self.w = weights
        self.fused = fused
        self.powers = weights["powers"]
        self.clear_bit0 = torch.ones(32, dtype=DTYPE)
        self.clear_bit0[0] = 0

    def value(self, bits: torch.Tensor) -> torch.Tensor:
        return bits @ self.powers[: bits.shape[-1]]

    def index(self, bits5: torch.Tensor) -> torch.Tensor:
        """A 5-bit field as a one-hot over 0..31."""
        if self.fused:
            return F.one_hot(self.value(bits5).long(), 32).to(DTYPE)
        return torch.relu(1 - (bits5 @ self.w["index_w"].T + self.w["index_c"]))

    def byte_bits(self, b: torch.Tensor) -> torch.Tensor:
        """Bytes as bits: a one-hot over the 256 values, then a table lookup."""
        if self.fused:
            return self.w["byte_bits"][b.long()]
        onehot = clamp01(1 - (b[..., None] - LEVELS).abs())
        return onehot @ self.w["byte_bits"]

    def decompose(self, x: torch.Tensor, n: int) -> torch.Tensor:
        if self.fused:
            return (x.long()[..., None] >> torch.arange(n) & 1).to(DTYPE)
        return decompose(x, n)

    def wrap_bits(self, x: torch.Tensor) -> torch.Tensor:
        """The bits of x mod 2**32, for integers -2**34 < x < 2**34."""
        return self.decompose(x + 2.0**34, 35)[..., :32]

    def step(self, s: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one instruction for every guest, updating `s` in place.
        Returns the exits, a (B, 5) one-hot or zero row in the order of the
        reason constants above, and each exit's detail: the faulting address,
        or the instruction word for an illegal instruction."""
        w = self.w
        batch = s.pc.shape[0]
        pc = s.pc
        mem_size = float(s.mem.size)

        # Fetch
        fetch_fault = clamp01(clamp01(pc + 4 - mem_size) + s.pc_low.sum(-1))
        fetched = s.mem.read(pc, 4)
        ibits = self.byte_bits(fetched).reshape(batch, 32)
        word = fetched @ BYTE_PLACES

        # Decode
        is_ = torch.relu(1 - (ibits @ w["decode_w"].T + w["decode_c"]))
        legal = is_.sum(-1)
        rd = self.index(ibits[:, 7:12])
        a_bits = torch.einsum("br,brk->bk", self.index(ibits[:, 15:20]), s.regs)
        b_bits = torch.einsum("br,brk->bk", self.index(ibits[:, 20:25]), s.regs)
        imm_bits = torch.einsum("bf,foi,bi->bo", is_ @ w["format_route"], w["imm_bits"], ibits)
        uses_imm = (is_ @ w["uses_imm"])[:, None]
        o_bits = uses_imm * imm_bits + (1 - uses_imm) * b_bits
        a, b, o, imm = (self.value(x) for x in (a_bits, b_bits, o_bits, imm_bits))

        # Integer and logic units
        add = a + o
        sub = a - o
        and_bits = a_bits * o_bits
        or_bits = a_bits + o_bits - and_bits
        xor_bits = or_bits - and_bits
        shamt = self.index(o_bits[:, :5])
        sll = self.value(torch.einsum("bs,sij,bj->bi", shamt, w["sll"], a_bits))
        srl_bits = torch.einsum("bs,sij,bj->bi", shamt, w["srl"], a_bits)
        fill = torch.einsum("bs,si->bi", shamt, w["sra_fill"]) * a_bits[:, 31:32]
        signed_a = a - TWO32 * a_bits[:, 31]
        signed_o = o - TWO32 * o_bits[:, 31]
        lt = clamp01(signed_o - signed_a)
        ltu = clamp01(o - a)
        eq = clamp01(1 - (a - o).abs())

        # Multiply and divide are deep, so they run only when some guest needs
        # them. Otherwise their results would be routed nowhere, so skipping
        # them changes nothing.
        route = is_ @ w["result_route"]
        zero = torch.zeros(batch, dtype=DTYPE)
        mul = mulh = mulhsu = mulhu = div = divu = rem = remu = zero
        if route[:, _MUL].any():
            mul, mulh, mulhsu, mulhu = self.multiply(a_bits, o_bits, a, o)
        if route[:, _DIV].any():
            div, divu, rem, remu = self.divide(a_bits, o_bits, a, o)

        # Memory
        addr = add - TWO32 * clamp01(add - TWO32 + 1)
        load_size = is_ @ w["load_size"]
        store_size = is_ @ w["store_size"]
        size = load_size + store_size
        memory_fault = clamp01(size) * clamp01(addr + size - mem_size)
        loaded = s.mem.read(addr, 4)
        sign_byte = self.byte_bits(loaded[:, 0])[:, 7]
        sign_half = self.byte_bits(loaded[:, 1])[:, 7]
        lbu = loaded[:, 0]
        lhu = lbu + 256 * loaded[:, 1]
        lw = loaded @ BYTE_PLACES
        lb = lbu + (TWO32 - 256) * sign_byte
        lh = lhu + (TWO32 - 65536) * sign_half

        # Exits, in the reference vCPU's order: fetch, decode, then execute
        ecall, ebreak = (is_ @ w["exit_route"]).unbind(-1)
        ok = 1 - fetch_fault
        exits = torch.stack([ok * ecall, ok * ebreak, ok * (1 - legal), fetch_fault, ok * legal * memory_fault], -1)
        commit = 1 - exits.sum(-1)
        info = exits[:, ILLEGAL] * word + exits[:, FETCH_FAULT] * pc + exits[:, MEMORY_FAULT] * addr

        # Results, wrapped to 32 bits
        results = torch.stack(
            [
                imm, pc + imm, pc + 4, add, sub, sll, lt, ltu,
                self.value(xor_bits), self.value(srl_bits), self.value(srl_bits + fill),
                self.value(or_bits), self.value(and_bits),
                mul, mulh, mulhsu, mulhu, div, divu, rem, remu,
                lb, lh, lw, lbu, lhu,
            ],
            -1,
        )  # fmt: skip
        rd_value = (route * results).sum(-1)
        wrapped = self.wrap_bits(torch.stack([rd_value, pc + 4, pc + imm, add], -1))
        rd_bits, seq_bits, relative_bits, jalr_bits = wrapped.unbind(1)
        jalr_bits = jalr_bits * self.clear_bit0

        # Next pc
        compare = torch.stack([eq, 1 - eq, lt, 1 - lt, ltu, 1 - ltu], -1)
        taken = ((is_ @ w["branch_route"]) * compare).sum(-1, keepdim=True)
        branch_bits = taken * relative_bits + (1 - taken) * seq_bits
        candidates = torch.stack([seq_bits, branch_bits, relative_bits, jalr_bits], 1)
        next_bits = torch.einsum("bc,bci->bi", is_ @ w["next_pc_route"], candidates)

        # Commit: delta-rule writes to the register file and memory
        writes = route.sum(-1) * (1 - rd[:, 0]) * commit
        s.regs = s.regs + writes[:, None, None] * rd[:, :, None] * (rd_bits[:, None, :] - s.regs)
        store_bytes = b_bits.reshape(batch, 4, 8) @ self.powers[:8]
        gates = commit[:, None] * clamp01(store_size[:, None] - torch.arange(4))
        s.mem.write(addr, store_bytes, gates)
        keep = (1 - commit)[:, None]
        s.pc = commit * self.value(next_bits) + (1 - commit) * pc
        s.pc_low = (1 - keep) * next_bits[:, :2] + keep * s.pc_low
        s.instret = s.instret + commit
        return exits, info

    def multiply(self, a_bits, o_bits, a, o):
        """The 64-bit product from 16-bit halves, so every partial value stays
        below 2**53 and exact."""
        a0, a1 = self.value(a_bits[:, :16]), self.value(a_bits[:, 16:])
        o0, o1 = self.value(o_bits[:, :16]), self.value(o_bits[:, 16:])
        mid = self.decompose(a0 * o1 + a1 * o0, 33)
        low = self.decompose(a0 * o0 + self.value(mid[:, :16]) * 65536, 33)
        mulhu = a1 * o1 + self.value(mid[:, 16:]) + low[:, 32]
        mul = self.value(low[:, :32])
        # Signed high words from the unsigned one, modulo 2**32
        mulh = mulhu - a_bits[:, 31] * o - o_bits[:, 31] * a
        mulhsu = mulhu - a_bits[:, 31] * o
        return mul, mulh, mulhsu, mulhu

    def divide(self, a_bits, o_bits, a, o):
        """Restoring division, unrolled into 32 compare-and-subtract layers, on
        two lanes at once: the unsigned operands, and the signed operands'
        magnitudes. Dividing by zero falls out of the unsigned lane as RISC-V
        defines it (all ones, remainder the dividend); the signed lane needs it
        set explicitly."""
        neg_a, neg_o = a_bits[:, 31], o_bits[:, 31]
        abs_a = a + neg_a * (TWO32 - 2 * a)
        abs_o = o + neg_o * (TWO32 - 2 * o)
        divisor = torch.stack([o, abs_o], 1)
        if self.fused:
            n, d = torch.stack([a, abs_a], 1).long(), divisor.long()
            safe = d.clamp(min=1)
            q = torch.where(d == 0, 0xFFFF_FFFF, n // safe).to(DTYPE)
            r = torch.where(d == 0, n, n % safe).to(DTYPE)
        else:
            numerator = torch.stack([a_bits, decompose(abs_a, 32)], 1)
            r = torch.zeros_like(divisor)
            q = torch.zeros_like(divisor)
            for i in range(31, -1, -1):
                r = 2 * r + numerator[:, :, i]
                fits = clamp01(r - divisor + 1)
                r = r - fits * divisor
                q = 2 * q + fits
        divu, remu = q[:, 0], r[:, 0]
        q_signed, r_signed = q[:, 1], r[:, 1]
        negative = neg_a + neg_o - 2 * neg_a * neg_o
        by_zero = clamp01(1 - o)
        div = by_zero * (TWO32 - 1) + (1 - by_zero) * (q_signed + negative * (TWO32 - 2 * q_signed))
        rem = by_zero * a + (1 - by_zero) * (r_signed + neg_a * (TWO32 - 2 * r_signed))
        return div, divu, rem, remu
