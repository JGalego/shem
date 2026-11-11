"""Measure how the neural vCPU's throughput grows with the batch size.

uv run python bench.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "tests"))
from rv32 import A0, EBREAK, T0, ZERO, add, addi, assemble, bne, li  # noqa: E402

from shem_neural.compile import compile_weights  # noqa: E402
from shem_neural.cpu import NeuralCpu, State  # noqa: E402
from shem_neural.memory import PagePool  # noqa: E402

MEM = 1 << 20
STEPS = 300


def main() -> None:
    torch.set_num_threads(1)
    cpu = NeuralCpu(compile_weights())
    pool = PagePool()
    # Each guest uses a different loop bound.
    print(f"{'guests':>6} {'instr/s total':>14} {'instr/s per guest':>18} {'pages':>6}")
    for batch in (1, 2, 4, 8, 16, 32, 64, 128):
        state = State.zeros(batch, MEM, pool=pool)
        for row in range(batch):
            n = 10_000 + row
            code = assemble(addi(A0, ZERO, 0), li(T0, n), add(A0, A0, T0), addi(T0, T0, -1), bne(T0, ZERO, -8), EBREAK)
            state.mem.set_bytes(row, 0, code)
        cpu.step(state)  # warm up
        started = time.perf_counter()
        for _ in range(STEPS):
            cpu.step(state)
        elapsed = time.perf_counter() - started
        total = batch * STEPS / elapsed
        print(f"{batch:>6} {total:>14,.0f} {total / batch:>18,.0f} {pool.in_use:>6}")
        state.mem.release()


if __name__ == "__main__":
    main()
