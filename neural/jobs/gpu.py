# /// script
# requires-python = ">=3.12"
# dependencies = ["torch>=2.9", "numpy>=2", "safetensors>=0.4"]
# ///
"""The neural CPU on a GPU: check it against the CPU bit for bit, then measure
its throughput.

    hf jobs uv run --flavor l4x1 --timeout 1h neural/jobs/gpu.py --rev <commit>

The job downloads shem at `--rev`, so it tests what was pushed, not the local
tree. It runs locally too, with `--local`, on whatever device torch finds.

1. Differential test: guests run random RV32IM programs from random register
   files, once with the CPU's paged executor, which the riscv-tests check
   against the Rust reference vCPU, and once per GPU mode. Every register,
   byte of memory, pc, instruction count and exit must match.
2. Throughput: every guest runs the same loop of multiplies, divides, loads,
   stores, shifts and branches, at batch sizes up to the GPU's memory, in each
   mode: eager, compiled, eager in a CUDA graph, and compiled in a CUDA graph.
"""

import argparse
import io
import json
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import torch

MEM = 64 << 10
GRAPH_STEPS = 64
MODES = {
    "eager": dict(compile=False, graph_steps=0),
    "compiled": dict(compile=True, graph_steps=0),
    "eager+graph": dict(compile=False, graph_steps=GRAPH_STEPS),
    "compiled+graph": dict(compile=True, graph_steps=GRAPH_STEPS),
}


def fetch(rev: str) -> Path:
    url = f"https://github.com/JGalego/shem/archive/{rev}.tar.gz"
    root = Path(tempfile.mkdtemp())
    with urllib.request.urlopen(url) as response:
        tarfile.open(fileobj=io.BytesIO(response.read())).extractall(root, filter="data")
    return next(root.iterdir()) / "neural"


def random_programs(isa, rv32, count: int, length: int, generator: torch.Generator) -> list[bytes]:
    """Valid encodings of random instructions with random fields, except that
    branches and jumps go a few instructions either way, so guests loop rather
    than leave at once. Exits and register jumps are rare."""
    weights = torch.tensor([0.02 if ins.exit else 0.1 if ins.next_pc == "jalr" else 1.0 for ins in isa.INSTRUCTIONS])
    programs = []
    for _ in range(count):
        picks = torch.multinomial(weights, length, replacement=True, generator=generator).tolist()
        fields = torch.randint(0, 2**32, (length,), generator=generator, dtype=torch.int64).tolist()
        offsets = (4 * torch.randint(-16, 16, (length,), generator=generator)).tolist()
        words = []
        for pick, field, offset in zip(picks, fields, offsets, strict=True):
            ins = isa.INSTRUCTIONS[pick]
            word = field & ~ins.mask & 0xFFFF_FFFF | ins.match
            if ins.format == "B":
                word = word & ~0xFE000F80 | rv32.b(0, 0, 0, offset) & 0xFE000F80
            elif ins.format == "J":
                word = word & 0xFFF | rv32.jal(0, offset) & 0xFFFFF000
            elif ins.next_pc == "jalr":
                word = word & 0xFFFFF | (offset & 0xFFF) << 20
            words.append(word)
        programs.append(b"".join(w.to_bytes(4, "little") for w in words))
    return programs


def load(state, programs: list[bytes], registers: torch.Tensor) -> None:
    for k, code in enumerate(programs):
        state.mem.set_bytes(k, 0, code)
    bits = (registers[..., None] >> torch.arange(32) & 1).to(torch.float64)
    bits[:, 0] = 0
    state.regs.copy_(bits.to(state.regs.device))


def differential(neural, weights, device: str, guests: int, steps: int) -> dict:
    import rv32

    from shem_neural import isa
    from shem_neural.cpu import NeuralCpu, State
    from shem_neural.gpu import Batch

    generator = torch.Generator().manual_seed(2026)
    programs = random_programs(isa, rv32, guests, 256, generator)
    # Small aligned register values, so loads, stores and jumps often stay in memory
    registers = 4 * torch.randint(0, MEM // 8, (guests, 32), generator=generator, dtype=torch.int64)

    started = time.perf_counter()
    cpu = NeuralCpu(weights)
    expected = State.zeros(guests, MEM)
    load(expected, programs, registers)
    halted = torch.zeros(guests, dtype=torch.bool)
    exits = torch.zeros(guests, 5, dtype=torch.float64)
    info = torch.zeros(guests, dtype=torch.float64)
    for _ in range(steps):
        step_exits, step_info = cpu.step(expected, active=~halted)
        new = (step_exits.sum(-1) > 0) & ~halted
        exits[new], info[new] = step_exits[new], step_info[new]
        halted |= new
    memory = torch.stack(
        [torch.frombuffer(bytearray(expected.mem.get_bytes(k, 0, MEM)), dtype=torch.uint8) for k in range(guests)]
    )
    print(f"reference: {guests} guests x {steps} steps on the CPU in {time.perf_counter() - started:.1f} s", flush=True)
    retired = expected.instret.sum().item()
    report = {
        "guests": guests,
        "steps": steps,
        "instructions": int(retired),
        "exits": {
            name: int(n)
            for name, n in zip(
                ["ecall", "ebreak", "illegal", "fetch fault", "memory fault"], exits.sum(0).tolist(), strict=True
            )
        },
        "still running": int((~halted).sum()),
        "modes": {},
    }

    for mode, options in MODES.items():
        if "graph" in mode and device != "cuda":
            continue
        batch = Batch(weights, guests, MEM, device, **options)
        load(batch.state, programs, registers)
        batch.run(steps)
        got = batch.state
        checks = {
            "pc": torch.equal(got.pc.cpu(), expected.pc),
            "pc_low": torch.equal(got.pc_low.cpu(), expected.pc_low),
            "regs": torch.equal(got.regs.cpu(), expected.regs),
            "instret": torch.equal(got.instret.cpu(), expected.instret),
            "memory": torch.equal(got.mem.bytes[:, :MEM].cpu(), memory),
            "exits": torch.equal(batch.exits.cpu(), exits),
            "info": torch.equal(batch.info.cpu(), info),
        }
        report["modes"][mode] = all(checks.values())
        print(f"{mode}: {'identical' if all(checks.values()) else checks}", flush=True)
    return report


def benchmark_program(rv32) -> bytes:
    S1, S2, T3, T4 = 9, 18, 28, 29
    T0, T1, T2, S0, A0, A1, ZERO = rv32.T0, rv32.T1, rv32.T2, rv32.S0, rv32.A0, rv32.A1, rv32.ZERO
    loop = [
        rv32.m("mul", T0, T0, S1),
        rv32.add(T0, T0, S2),
        rv32.i(0x13, 7, T1, T0, 0x3FC),  # andi: a word in a 1 KiB array
        rv32.add(T1, T1, S0),
        rv32.sw(T0, T1, 0),
        rv32.lw(T2, T1, 0),
        rv32.m("divu", T3, T2, S2),
        rv32.m("remu", T4, T2, S1),
        rv32.r(0, 4, A0, A0, T3),  # xor
        rv32.r(0, 1, A1, A0, T4),  # sll
        rv32.r(0x20, 5, A1, A1, T4),  # sra
    ]
    loop.append(rv32.jal(ZERO, -4 * len(loop)))
    return rv32.assemble(rv32.li(S0, 0x1000), rv32.li(S1, 1103515245), rv32.li(S2, 12345), *loop)


def throughput(neural, weights, device: str, sizes: list[int], seconds: float) -> list[dict]:
    import rv32

    from shem_neural.gpu import Batch

    code = benchmark_program(rv32)
    rows = []
    # The CPU, for comparison, up to the sizes it can manage in reasonable time
    configs = [("cpu", "cpu", dict(compile=False, graph_steps=0), size) for size in sizes if size <= 1024]
    if device == "cuda":
        configs += [(mode, device, options, size) for mode, options in MODES.items() for size in sizes]
    for mode, where, options, size in configs:
        batch = None
        try:
            started = time.perf_counter()
            batch = Batch(weights, size, MEM, where, **options)
            for k in range(size):
                batch.state.mem.set_bytes(k, 0, code)
            batch.run(GRAPH_STEPS + 2)  # compile, capture and warm up
            sync(where)
            setup = time.perf_counter() - started
            steps, elapsed = GRAPH_STEPS, 0.0
            while elapsed < seconds:
                steps *= 2
                before = time.perf_counter()
                batch.run(steps)
                sync(where)
                elapsed = time.perf_counter() - before
            assert not batch.halted.any(), "the benchmark loop never exits"
            rate = size * steps / elapsed
            row = {"mode": mode, "guests": size, "instructions/s": rate, "steps/s": steps / elapsed, "setup s": setup}
            print(
                f"{mode:>15} {size:>7} guests: {rate:>14,.0f} instructions/s ({steps / elapsed:,.0f} steps/s)",
                flush=True,
            )
        except torch.OutOfMemoryError:
            row = {"mode": mode, "guests": size, "error": "out of memory"}
            print(f"{mode:>15} {size:>7} guests: out of memory", flush=True)
        rows.append(row)
        del batch
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rev", default="main", help="the shem commit to test")
    parser.add_argument("--local", action="store_true", help="use the neural package next to this script")
    parser.add_argument("--guests", type=int, default=1024, help="guests in the differential test")
    parser.add_argument("--steps", type=int, default=512, help="steps in the differential test")
    parser.add_argument("--sizes", default="1,64,1024,4096,16384,32768", help="batch sizes to measure")
    parser.add_argument("--seconds", type=float, default=2.0, help="minimum time per measurement")
    args = parser.parse_args()

    neural = Path(__file__).resolve().parent.parent if args.local else fetch(args.rev)
    sys.path[:0] = [str(neural / "src"), str(neural / "tests")]
    from shem_neural.compile import compile_weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    environment = {
        "rev": args.rev,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "cpu threads": torch.get_num_threads(),
    }
    print(json.dumps(environment), flush=True)
    weights = compile_weights()
    report = {
        "environment": environment,
        "differential": differential(neural, weights, device, args.guests, args.steps),
        "throughput": throughput(neural, weights, device, [int(n) for n in args.sizes.split(",")], args.seconds),
    }
    print("RESULTS " + json.dumps(report), flush=True)
    if not all(report["differential"]["modes"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
