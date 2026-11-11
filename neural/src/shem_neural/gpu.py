"""Many guests at once on an accelerator.

`Batch` keeps a batch of guests on a device, each with flat memory, and steps
them with the static executor, so nothing in a step depends on data outside
the kernels. For a network this small the cost of a step is mostly overhead,
the launch of a few hundred small kernels, and that buys two ways to cut it:
`torch.compile` fuses the step into a handful of kernels, and on CUDA a graph
captures many steps so they launch as one. Every buffer is updated in place,
so a captured graph can be replayed over and over.

A guest stops at its first exit and keeps that exit's reason and detail;
`Batch` is for throughput and differential testing, and the server, which
services exits, still steps guests on the CPU.
"""

import torch

from .compile import to
from .cpu import DTYPE, NeuralCpu, State
from .memory import FlatMemory


class Batch:
    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        batch: int,
        mem_size: int,
        device: torch.device | str = "cpu",
        compile: bool = False,
        graph_steps: int = 0,
    ):
        """With `compile`, run the step through `torch.compile`; with
        `graph_steps` on CUDA, capture that many steps in a CUDA graph."""
        self.device = torch.device(device)
        self.cpu = NeuralCpu(to(weights, self.device), static=True)
        self.state = State.zeros(batch, mem_size, device=self.device, flat=True)
        self.halted = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.exits = torch.zeros(batch, 5, dtype=DTYPE, device=self.device)
        self.info = torch.zeros(batch, dtype=DTYPE, device=self.device)
        self._step = torch.compile(self._advance, fullgraph=True, dynamic=False) if compile else self._advance
        self.graph_steps = graph_steps
        self._graph: torch.cuda.CUDAGraph | None = None

    def _buffers(self) -> tuple[torch.Tensor, ...]:
        s = self.state
        return s.pc, s.pc_low, s.regs, s.mem.bytes, s.instret, self.halted, self.exits, self.info

    def _advance(self, pc, pc_low, regs, mem, instret, halted, exits, info) -> None:
        """One step, written back into the buffers it read."""
        s = State(pc=pc, pc_low=pc_low, regs=regs, mem=FlatMemory.of(mem), instret=instret, base=self.state.base)
        step_exits, step_info = self.cpu.step(s, active=~halted)
        new = (step_exits.sum(-1) > 0) & ~halted
        exits.copy_(torch.where(new[:, None], step_exits, exits))
        info.copy_(torch.where(new, step_info, info))
        halted.logical_or_(new)
        pc.copy_(s.pc)
        pc_low.copy_(s.pc_low)
        regs.copy_(s.regs)
        instret.copy_(s.instret)

    def step(self) -> None:
        self._step(*self._buffers())

    def run(self, steps: int) -> None:
        """Step every guest `steps` times; halted guests stay put."""
        if self.graph_steps and self.device.type == "cuda":
            if self._graph is None and steps >= self.graph_steps + 2:
                self._capture()
                steps -= 2  # the warm-up steps before capture are real steps
            while self._graph is not None and steps >= self.graph_steps:
                self._graph.replay()
                steps -= self.graph_steps
        for _ in range(steps):
            self.step()

    def _capture(self) -> None:
        """Warm up on a side stream, as capture requires, then record
        `graph_steps` steps."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.step()
        torch.cuda.current_stream().wait_stream(side)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            for _ in range(self.graph_steps):
                self.step()
