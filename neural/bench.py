"""Measure how the neural vCPU's throughput grows with the batch size.

    uv run python bench.py
    uv run python bench.py --check

With `--check`, fail if batching has stopped paying: 64 guests together must
run at least `MIN_SPEEDUP` times as many instructions a second as one. The
speed of one machine says little about another's, so the check is on a ratio,
which a change that costs batching its gains breaks on any of them. The table
goes to the job summary too, when `GITHUB_STEP_SUMMARY` names a file.
"""

import os
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
#: How much 64 guests together must beat one by: batching is meant to give about 30 times
MIN_SPEEDUP = 8


def main() -> None:
    torch.set_num_threads(1)
    cpu = NeuralCpu(compile_weights())
    pool = PagePool()
    # Each guest uses a different loop bound.
    print(f"{'guests':>6} {'instr/s total':>14} {'instr/s per guest':>18} {'pages':>6}")
    totals = {}
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
        totals[batch] = total
        print(f"{batch:>6} {total:>14,.0f} {total / batch:>18,.0f} {pool.in_use:>6}")
        state.mem.release()
    speedup = totals[64] / totals[1]
    print(f"64 guests run {speedup:.1f} times as many instructions a second as one")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as out:
            out.write("### Neural vCPU throughput\n\n| guests | instructions/s |\n|---|---|\n")
            out.writelines(f"| {batch} | {total:,.0f} |\n" for batch, total in totals.items())
            out.write(
                f"\n64 guests run **{speedup:.1f}×** one's instructions a second (at least {MIN_SPEEDUP}× expected).\n"
            )
    if "--check" in sys.argv and speedup < MIN_SPEEDUP:
        sys.exit(f"batching has stopped paying: {speedup:.1f}x, where at least {MIN_SPEEDUP}x is expected")


if __name__ == "__main__":
    main()
