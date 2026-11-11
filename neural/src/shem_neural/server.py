"""Serve neural vCPUs to the shem host over a Unix socket.

The protocol is the `Vcpu` interface as requests, KVM-style: create a vCPU, run
it until it exits, read and write its registers and memory, and fork it by
cloning.

vCPUs share memory pages, by memory size, so forks and guests resumed from the
same snapshot share what they have not written. Runs from different clients
are batched: one executor thread steps every vCPU with a `RUN` in flight in the
same forward pass, and guests join and leave the batch between steps.

Every message is a frame: a little-endian u32 length, then the payload. A
request payload is an op byte and its arguments; a response payload is a
status byte (see `Status`) and the result. `docs/neural.md` lists the ops.

    uv run python -m shem_neural.server --socket /tmp/shem-neural.sock
"""

import argparse
import itertools
import socketserver
import struct
import sys
import threading
from enum import IntEnum
from pathlib import Path

import torch

from .compile import compile_weights
from .cpu import DTYPE, NeuralCpu, State
from .memory import PagePool
from .vcpu import _EXIT_CODES, BUDGET, MemoryFault, NeuralVcpu


class Op(IntEnum):
    CREATE = 1
    DROP = 2
    CLONE = 3
    RUN = 4
    GET_REG = 5
    SET_REG = 6
    GET_PC = 7
    SET_PC = 8
    READ_MEM = 9
    WRITE_MEM = 10
    RESET = 11
    SNAPSHOT = 12
    RESTORE = 13


class Status(IntEnum):
    OK = 0
    ERROR = 1
    MEMORY_FAULT = 2


class Job:
    """A `RUN` in flight: a vCPU, how far it may go, and how far it has got."""

    def __init__(self, vcpu: NeuralVcpu, budget: int):
        self.vcpu = vcpu
        self.budget = budget
        self.steps = 0
        self.result: tuple[int, int, int] | None = None
        self.done = threading.Event()


class Batcher:
    """Steps every vCPU with a `RUN` in flight in one batch: continuous batching.

    The executor thread stacks the running vCPUs' states into one batch and
    steps it. A guest that exits, or runs out of budget, stops committing and
    gets its result; when a new run arrives, the batch is unstacked and rebuilt
    with it. The network computes every guest the same way whatever it runs, so
    a batch of unrelated guests costs about as much per step as one guest.
    """

    def __init__(self, lock: threading.RLock):
        self.lock = lock
        self.wake = threading.Condition(lock)
        self.jobs: list[Job] = []
        self.arrivals = 0
        self.largest = 0  # the biggest batch so far, for tests and benchmarks
        threading.Thread(target=self._loop, daemon=True).start()

    def run(self, vcpu: NeuralVcpu, budget: int) -> tuple[int, int, int]:
        job = Job(vcpu, budget)
        with self.wake:
            self.jobs.append(job)
            self.arrivals += 1
            self.wake.notify()
        job.done.wait()
        assert job.result is not None
        return job.result

    def _loop(self) -> None:
        while True:
            with self.wake:
                while not self.jobs:
                    self.wake.wait()
                first = self.jobs[0].vcpu
                if first.dense:  # the dense network's memory does not batch
                    job = self.jobs.pop(0)
                    job.result = first.run(job.budget)
                    job.done.set()
                    continue
                jobs = [j for j in self.jobs if j.vcpu.mem_size == first.mem_size]
                arrivals = self.arrivals
                batch = State.stack([j.vcpu.state for j in jobs])
                self.largest = max(self.largest, len(jobs))
            self._step(jobs, batch, arrivals)

    def _step(self, jobs: list[Job], batch: State, arrivals: int) -> None:
        cpu = jobs[0].vcpu.cpu
        active = [True] * len(jobs)
        while True:
            with self.lock:
                for row, job in enumerate(jobs):
                    if active[row] and job.steps >= job.budget:
                        job.result, active[row] = (BUDGET, 0, job.steps), False
                if not any(active) or self.arrivals != arrivals:
                    break
                exits, info = cpu.step(batch, torch.tensor(active, dtype=DTYPE))
                exited = (exits.sum(-1) > 0).tolist()
                reasons, details = exits.argmax(-1).tolist(), info.tolist()
                for row, job in enumerate(jobs):
                    if not active[row]:
                        continue
                    if exited[row]:
                        job.result = (_EXIT_CODES[reasons[row]], int(details[row]), job.steps)
                        active[row] = False
                    else:
                        job.steps += 1
        with self.lock:
            batch.unstack([j.vcpu.state for j in jobs])
            for job in jobs:
                if job.result is not None:
                    self.jobs.remove(job)
                    job.done.set()


class Registry:
    def __init__(self, dense: bool):
        weights = compile_weights()
        self.cpu = NeuralCpu(weights, fused=not dense)
        self.dense = dense
        self.vcpus: dict[int, NeuralVcpu] = {}
        self.pools: dict[int, PagePool] = {}  # by memory size
        self.ids = itertools.count(1)
        self.lock = threading.RLock()
        self.batcher = Batcher(self.lock)

    def add(self, vcpu: NeuralVcpu) -> int:
        vcpu_id = next(self.ids)
        self.vcpus[vcpu_id] = vcpu
        return vcpu_id

    def handle(self, payload: bytes) -> bytes:
        op = Op(payload[0])
        if op == Op.RUN:
            vcpu_id, budget = struct.unpack_from("<IQ", payload, 1)
            with self.lock:
                vcpu = self.vcpus[vcpu_id]
            reason, info, steps = self.batcher.run(vcpu, budget)
            with self.lock:
                return struct.pack("<BIIQ", reason, info, vcpu.pc, steps)
        with self.lock:
            return self._handle(op, payload)

    def _handle(self, op: Op, payload: bytes) -> bytes:
        if op == Op.CREATE:
            (mem_size,) = struct.unpack_from("<I", payload, 1)
            (base,) = struct.unpack_from("<I", payload, 5) if len(payload) >= 9 else (0,)
            pool = None if self.dense else self.pools.setdefault(mem_size, PagePool())
            return struct.pack("<I", self.add(NeuralVcpu(mem_size, self.cpu, self.dense, pool, base)))
        (vcpu_id,) = struct.unpack_from("<I", payload, 1)
        args = payload[5:]
        vcpu = self.vcpus[vcpu_id]
        match op:
            case Op.DROP:
                del self.vcpus[vcpu_id]
                vcpu.release()
                return b""
            case Op.CLONE:
                return struct.pack("<I", self.add(vcpu.clone()))
            case Op.GET_REG:
                return struct.pack("<I", vcpu.reg(args[0]))
            case Op.SET_REG:
                index, value = struct.unpack("<BI", args)
                vcpu.set_reg(index, value)
                return b""
            case Op.GET_PC:
                return struct.pack("<I", vcpu.pc)
            case Op.SET_PC:
                (vcpu.pc,) = struct.unpack("<I", args)
                return b""
            case Op.READ_MEM:
                addr, length = struct.unpack("<II", args)
                return vcpu.read_mem(addr, length)
            case Op.WRITE_MEM:
                (addr,) = struct.unpack_from("<I", args)
                vcpu.write_mem(addr, args[4:])
                return b""
            case Op.RESET:
                vcpu.reset()
                return b""
            case Op.SNAPSHOT:
                return vcpu.snapshot()
            case Op.RESTORE:
                vcpu.restore(args)
                return b""
        raise ValueError(f"unknown op {op}")


def read_frame(stream) -> bytes | None:
    header = stream.read(4)
    if len(header) < 4:
        return None
    (length,) = struct.unpack("<I", header)
    return stream.read(length)


def write_frame(stream, payload: bytes) -> None:
    stream.write(struct.pack("<I", len(payload)) + payload)
    stream.flush()


def make_server(socket_path: Path, dense: bool = False) -> tuple[socketserver.BaseServer, Registry]:
    """A server on `socket_path`, not yet serving, and its registry of vCPUs."""
    torch.set_num_threads(1)  # tiny tensors: threads only add overhead
    registry = Registry(dense)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while (payload := read_frame(self.rfile)) is not None:
                try:
                    response = bytes([Status.OK]) + registry.handle(payload)
                except MemoryFault as fault:
                    response = bytes([Status.MEMORY_FAULT]) + str(fault).encode()
                except Exception as error:  # report anything else to the host
                    response = bytes([Status.ERROR]) + f"{type(error).__name__}: {error}".encode()
                write_frame(self.wfile, response)

    socket_path.unlink(missing_ok=True)
    server = socketserver.ThreadingUnixStreamServer(str(socket_path), Handler)
    server.daemon_threads = True
    return server, registry


def serve(socket_path: Path, dense: bool = False) -> None:
    server, _ = make_server(socket_path, dense)
    with server:
        print(f"shem-neural: serving vCPUs on {socket_path}", file=sys.stderr, flush=True)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", type=Path, required=True, help="Unix socket to listen on")
    parser.add_argument("--dense", action="store_true", help="run the network densely, without fused kernels")
    args = parser.parse_args()
    try:
        serve(args.socket, args.dense)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
