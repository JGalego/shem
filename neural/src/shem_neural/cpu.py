"""RV32IM network: one forward pass executes one instruction per guest.

The dense executor uses compiled linear maps, one-hot attention, gating and
clamps. Registers hold bit vectors and memory holds bytes; arithmetic uses
integer values represented exactly in float64. `decompose` converts values
back to bits through threshold layers.

The fused executor replaces those layers with equivalent indexing and integer
operations. It skips multiply/divide units when no guest needs them, unless
`static` is set for graph capture.

Register and memory writes use the delta rule, gated by `commit`. An exiting
instruction commits no state changes, matching the reference vCPU.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import isa
from .memory import DeltaMemory, FlatMemory, PagedMemory, PagePool

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


def _row(regs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Each guest's register `index[b]`, as a row of bits."""
    return regs.gather(1, index[:, None, None].expand(-1, 1, regs.shape[2])).squeeze(1)


@dataclass
class State:
    """Everything that defines a batch of machines between instructions."""

    pc: torch.Tensor  # (B,)
    pc_low: torch.Tensor  # (B, 2): the low bits of pc, for the alignment check
    regs: torch.Tensor  # (B, 32, 32): one row of bits per register
    mem: DeltaMemory | PagedMemory | FlatMemory
    instret: torch.Tensor  # (B,)
    base: torch.Tensor  # (B,): the guest address where memory starts

    @classmethod
    def zeros(
        cls,
        batch: int,
        mem_size: int,
        dense: bool = False,
        pool: PagePool | None = None,
        device: torch.device | str = "cpu",
        flat: bool = False,
        base: int = 0,
    ) -> "State":
        """Machines with zeroed registers and memory at `base`: a delta-rule
        state matrix with `dense`, a row of bytes per guest on `device` with
        `flat`, and otherwise pages from `pool`."""
        if dense:
            mem = DeltaMemory(batch, mem_size)
        elif flat:
            mem = FlatMemory(batch, mem_size, device)
        else:
            mem = PagedMemory(batch, mem_size, pool)
        zeros = lambda *shape: torch.zeros(*shape, dtype=DTYPE, device=device)  # noqa: E731
        return cls(
            pc=zeros(batch),
            pc_low=zeros(batch, 2),
            regs=zeros(batch, 32, 32),
            mem=mem,
            instret=zeros(batch),
            base=torch.full((batch,), float(base), dtype=DTYPE, device=device),
        )

    @classmethod
    def stack(cls, states: list["State"]) -> "State":
        """One batch of the given single machines, which hands their page
        tables to the batch until `unstack` hands them back."""
        return cls(
            pc=torch.cat([s.pc for s in states]),
            pc_low=torch.cat([s.pc_low for s in states]),
            regs=torch.cat([s.regs for s in states]),
            mem=PagedMemory.stack([s.mem for s in states]),
            instret=torch.cat([s.instret for s in states]),
            base=torch.cat([s.base for s in states]),
        )

    def unstack(self, states: list["State"]) -> None:
        """Write each row of this batch back to the single machine it came from."""
        for row, state in enumerate(states):
            rows = slice(row, row + 1)
            state.pc, state.pc_low = self.pc[rows].clone(), self.pc_low[rows].clone()
            state.regs, state.instret = self.regs[rows].clone(), self.instret[rows].clone()
            state.mem.table = self.mem.table[rows].clone()

    def clone(self) -> "State":
        return State(
            self.pc.clone(),
            self.pc_low.clone(),
            self.regs.clone(),
            self.mem.clone(),
            self.instret.clone(),
            self.base.clone(),
        )


class NeuralCpu:
    def __init__(self, weights: dict[str, torch.Tensor], fused: bool = True, static: bool = False):
        """With `static`, every unit runs on every step, even the multiplier and
        divider when no guest needs them: no control flow depends on data, so
        the step can be compiled into a fixed graph of kernels. The weights'
        device is where the network runs."""
        self.w = weights
        self.fused = fused
        self.static = static
        self.device = weights["powers"].device
        self.powers = weights["powers"]
        self.byte_places = BYTE_PLACES.to(self.device)
        self.levels = LEVELS.to(self.device)
        self.clear_bit0 = torch.ones(32, dtype=DTYPE, device=self.device)
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
        onehot = clamp01(1 - (b[..., None] - self.levels).abs())
        return onehot @ self.w["byte_bits"]

    def decompose(self, x: torch.Tensor, n: int) -> torch.Tensor:
        if self.fused:
            return (x.long()[..., None] >> torch.arange(n, device=x.device) & 1).to(DTYPE)
        return decompose(x, n)

    @staticmethod
    def relative(x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """x - base modulo 2**32: an address's offset into memory. Below the
        base it wraps to beyond the end, so it faults, as in the reference."""
        return x - base + TWO32 * clamp01(base - x)

    def wrap_bits(self, x: torch.Tensor) -> torch.Tensor:
        """The bits of x mod 2**32, for integers -2**34 < x < 2**34."""
        return self.decompose(x + 2.0**34, 35)[..., :32]

    def step(self, s: State, active: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one instruction for every guest, updating `s` in place.
        Returns the exits, a (B, 5) one-hot or zero row in the order of the
        reason constants above, and each exit's detail: the faulting address,
        or the instruction word for an illegal instruction. Guests where the
        optional 0/1 mask `active` is 0 are computed but commit nothing."""
        w = self.w
        batch = s.pc.shape[0]
        pc = s.pc
        mem_size = float(s.mem.size)

        # Fetch, from memory at the base address
        pc_offset = self.relative(pc, s.base)
        fetch_fault = clamp01(clamp01(pc_offset + 4 - mem_size) + s.pc_low.sum(-1))
        fetched = s.mem.read(pc_offset, 4)
        ibits = self.byte_bits(fetched).reshape(batch, 32)
        word = fetched @ self.byte_places

        # Decode
        is_ = torch.relu(1 - (ibits @ w["decode_w"].T + w["decode_c"]))
        legal = is_.sum(-1)
        if self.fused:
            # Gather the operands' rows, rather than multiply by one-hots
            rd_index, rs1, rs2 = (self.value(ibits[:, at : at + 5]).long() for at in (7, 15, 20))
            a_bits, b_bits = _row(s.regs, rs1), _row(s.regs, rs2)
        else:
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
        sll, srl, sra = self.shift(a_bits, o_bits, a)
        signed_a = a - TWO32 * a_bits[:, 31]
        signed_o = o - TWO32 * o_bits[:, 31]
        lt = clamp01(signed_o - signed_a)
        ltu = clamp01(o - a)
        eq = clamp01(1 - (a - o).abs())

        # Multiply and divide are deep, so they run only when some guest needs
        # them. Otherwise their results would be routed nowhere, so skipping
        # them changes nothing.
        route = is_ @ w["result_route"]
        zero = torch.zeros_like(pc)
        mul = mulh = mulhsu = mulhu = div = divu = rem = remu = zero
        if self.static or route[:, _MUL].any():
            mul, mulh, mulhsu, mulhu = self.multiply(a_bits, o_bits, a, o)
        if self.static or route[:, _DIV].any():
            div, divu, rem, remu = self.divide(a_bits, o_bits, a, o)

        # Memory
        addr = add - TWO32 * clamp01(add - TWO32 + 1)
        load_size = is_ @ w["load_size"]
        store_size = is_ @ w["store_size"]
        size = load_size + store_size
        offset = self.relative(addr, s.base)
        memory_fault = clamp01(size) * clamp01(offset + size - mem_size)
        loaded = s.mem.read(offset, 4)
        sign_byte = self.byte_bits(loaded[:, 0])[:, 7]
        sign_half = self.byte_bits(loaded[:, 1])[:, 7]
        lbu = loaded[:, 0]
        lhu = lbu + 256 * loaded[:, 1]
        lw = loaded @ self.byte_places
        lb = lbu + (TWO32 - 256) * sign_byte
        lh = lhu + (TWO32 - 65536) * sign_half

        # Exits, in the reference vCPU's order: fetch, decode, then execute
        ecall, ebreak = (is_ @ w["exit_route"]).unbind(-1)
        ok = 1 - fetch_fault
        exits = torch.stack([ok * ecall, ok * ebreak, ok * (1 - legal), fetch_fault, ok * legal * memory_fault], -1)
        commit = 1 - exits.sum(-1)
        if active is not None:
            commit = commit * active
        info = exits[:, ILLEGAL] * word + exits[:, FETCH_FAULT] * pc + exits[:, MEMORY_FAULT] * addr

        # Results, wrapped to 32 bits
        results = torch.stack(
            [
                imm, pc + imm, pc + 4, add, sub, sll, lt, ltu,
                self.value(xor_bits), srl, sra,
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
        if self.fused:
            # The same write, to the one row it can change
            writes = route.sum(-1) * (rd_index != 0) * commit
            target = rd_index[:, None, None].expand(-1, 1, s.regs.shape[2])
            old = s.regs.gather(1, target)
            s.regs = s.regs.scatter(1, target, old + writes[:, None, None] * (rd_bits[:, None, :] - old))
        else:
            writes = route.sum(-1) * (1 - rd[:, 0]) * commit
            s.regs = s.regs + writes[:, None, None] * rd[:, :, None] * (rd_bits[:, None, :] - s.regs)
        store_bytes = b_bits.reshape(batch, 4, 8) @ self.powers[:8]
        gates = commit[:, None] * clamp01(store_size[:, None] - torch.arange(4, device=self.device))
        s.mem.write(offset, store_bytes, gates)
        keep = (1 - commit)[:, None]
        s.pc = commit * self.value(next_bits) + (1 - commit) * pc
        s.pc_low = (1 - keep) * next_bits[:, :2] + keep * s.pc_low
        s.instret = s.instret + commit
        return exits, info

    def shift(self, a_bits, o_bits, a):
        """sll, srl and sra of `a` by the low five bits of the second operand:
        attention over the shift matrices, with copies of the sign bit for
        sra, or for the fused executor, integer shifts."""
        if self.fused:
            amount = self.value(o_bits[:, :5]).long()
            whole = a.long()
            signed = whole - (whole >> 31 << 32)
            sll = (whole << amount) & 0xFFFF_FFFF
            sra = (signed >> amount) & 0xFFFF_FFFF
            return sll.to(DTYPE), (whole >> amount).to(DTYPE), sra.to(DTYPE)
        w = self.w
        shamt = self.index(o_bits[:, :5])
        sll = self.value(torch.einsum("bs,sij,bj->bi", shamt, w["sll"], a_bits))
        srl_bits = torch.einsum("bs,sij,bj->bi", shamt, w["srl"], a_bits)
        fill = torch.einsum("bs,si->bi", shamt, w["sra_fill"]) * a_bits[:, 31:32]
        return sll, self.value(srl_bits), self.value(srl_bits + fill)

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
