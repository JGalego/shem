"""The neural vCPU behind the interface the shem host drives (`Vcpu` in
`crates/shem/src/vcpu.rs`): run until an exit, read and write registers and
memory, reset, and snapshot or restore in the canonical machine-state format.

Register and memory access touch the state tensors directly, the way the
architecture has the host read and write DeltaNet rows without decoding tokens.
"""

import struct

import torch

from .compile import compile_weights
from .cpu import DTYPE, EBREAK, ECALL, FETCH_FAULT, ILLEGAL, MEMORY_FAULT, NeuralCpu, State
from .memory import PagePool

# Exit reasons on the wire, as `ExitReason` in the host
BUDGET, EXIT_ECALL, EXIT_EBREAK, EXIT_ILLEGAL, EXIT_MEMORY_FAULT = range(5)
_EXIT_CODES = {
    ECALL: EXIT_ECALL,
    EBREAK: EXIT_EBREAK,
    ILLEGAL: EXIT_ILLEGAL,
    FETCH_FAULT: EXIT_MEMORY_FAULT,
    MEMORY_FAULT: EXIT_MEMORY_FAULT,
}

SNAPSHOT_MAGIC = b"SHEM"
SNAPSHOT_VERSION = 1
_SNAPSHOT_HEADER = struct.Struct("<4sII32IQI")


class MemoryFault(Exception):
    """The host touched guest memory outside the guest's address space."""


def _bits(value: int) -> torch.Tensor:
    return torch.tensor([value >> i & 1 for i in range(32)], dtype=DTYPE)


class NeuralVcpu:
    def __init__(self, mem_size: int, cpu: NeuralCpu | None = None, dense: bool = False, pool: PagePool | None = None):
        """With `dense`, run the network as described, one-hot products and all;
        otherwise use the fused executor, which computes the same thing faster,
        with memory paged from `pool`, shared with other vCPUs."""
        self.cpu = cpu or NeuralCpu(compile_weights(), fused=not dense)
        self.dense = dense
        self.state = State.zeros(1, mem_size, dense, pool)

    @property
    def mem_size(self) -> int:
        return self.state.mem.size

    def run(self, budget: int) -> tuple[int, int, int]:
        """Execute at most `budget` instructions. Returns the exit reason, its
        detail, and the number of instructions retired."""
        steps = 0
        while steps < budget:
            exits, info = self.cpu.step(self.state)
            if exits.any():
                return _EXIT_CODES[int(exits[0].argmax())], int(info[0]), steps
            steps += 1
        return BUDGET, 0, steps

    @property
    def pc(self) -> int:
        return int(self.state.pc[0])

    @pc.setter
    def pc(self, value: int) -> None:
        self.state.pc[0] = value
        self.state.pc_low[0] = torch.tensor([value & 1, value >> 1 & 1], dtype=DTYPE)

    def reg(self, index: int) -> int:
        return int(self.cpu.value(self.state.regs[0, index]))

    def set_reg(self, index: int, value: int) -> None:
        if index != 0:
            self.state.regs[0, index] = _bits(value)

    def _range(self, addr: int, length: int) -> slice:
        if addr < 0 or addr + length > self.mem_size:
            raise MemoryFault(f"{length} bytes at {addr:#010x}")
        return slice(addr, addr + length)

    def read_mem(self, addr: int, length: int) -> bytes:
        self._range(addr, length)
        return self.state.mem.get_bytes(0, addr, length)

    def write_mem(self, addr: int, data: bytes) -> None:
        self._range(addr, len(data))
        self.state.mem.set_bytes(0, addr, data)

    def reset(self) -> None:
        pool = None if self.dense else self.state.mem.pool
        self.state.mem.release()
        self.state = State.zeros(1, self.mem_size, self.dense, pool)

    def release(self) -> None:
        """Give back this vCPU's memory, before dropping it."""
        self.state.mem.release()

    def snapshot(self) -> bytes:
        regs = [self.reg(i) for i in range(32)]
        header = _SNAPSHOT_HEADER.pack(
            SNAPSHOT_MAGIC,
            SNAPSHOT_VERSION,
            self.pc,
            *regs,
            int(self.state.instret[0]),
            self.mem_size,
        )
        return header + self.read_mem(0, self.mem_size)

    def restore(self, snapshot: bytes) -> None:
        """Rebuild the whole state from a canonical snapshot, whichever vCPU took it."""
        magic, version, pc, *rest = _SNAPSHOT_HEADER.unpack_from(snapshot)
        regs, (instret, mem_len) = rest[:32], rest[32:]
        if magic != SNAPSHOT_MAGIC or version != SNAPSHOT_VERSION:
            raise ValueError("not a shem snapshot of a supported version")
        if mem_len != self.mem_size or len(snapshot) != _SNAPSHOT_HEADER.size + mem_len:
            raise ValueError(f"snapshot has {mem_len} bytes of memory, this vCPU has {self.mem_size}")
        self.reset()
        self.pc = pc
        for i, value in enumerate(regs):
            self.set_reg(i, value)
        self.state.instret[0] = instret
        self.write_mem(0, snapshot[_SNAPSHOT_HEADER.size :])

    def clone(self) -> "NeuralVcpu":
        """A copy for `fork`: same weights, copied state."""
        copy = NeuralVcpu.__new__(NeuralVcpu)
        copy.cpu, copy.dense = self.cpu, self.dense
        copy.state = self.state.clone()
        return copy
