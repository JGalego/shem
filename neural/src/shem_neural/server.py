"""Serve neural vCPUs and model sessions over a Unix socket.

vCPUs run on the CPU, sharing pages by memory size and batching runs by CPU
weights and memory size. Model sessions stream replies and cache loaded weights
on the selected model device.

Frames are a little-endian u32 length followed by a payload. Requests start
with an op byte, responses with a `Status` byte. See `docs/neural.md`.

    uv run python -m shem_neural.server --socket /tmp/shem-neural.sock
"""

import argparse
import hashlib
import itertools
import json
import os
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
    GENERATE = 14
    TURN = 15
    SAVE = 16
    CLOSE = 17


class Status(IntEnum):
    OK = 0
    ERROR = 1
    MEMORY_FAULT = 2
    #: Part of a streamed result: more frames follow, then one with OK.
    CHUNK = 3


class Models:
    """Loaded models, by their manifest's contents, so sessions of the same
    model share its weights, however many containers unpack it; and the live
    sessions of running containers, by the ID the shim gives them."""

    def __init__(self, device: str = "auto", socket: Path | None = None):
        self.device = device
        #: This server's socket, where tools' programs run on its vCPUs
        self.socket = socket
        self.loaded: dict[str, tuple] = {}
        self.sessions: dict[str, Live] = {}
        self.lock = threading.Lock()

    def get(self, directory: Path):
        from .model.blocks import Model
        from .model.tokenizer import Tokenizer

        key = hashlib.sha256((directory / "shem.json").read_bytes()).hexdigest()
        with self.lock:
            if key not in self.loaded:
                model = Model(directory, device=self.device)
                self.loaded[key] = (model, Tokenizer(directory, model.manifest.tokenizer))
            return self.loaded[key]

    def generate(self, request: dict, write) -> None:
        """Start a session for `request` and reply to its prompt, writing the
        reply with `write` as it comes. The request has the model's
        directory, the driver's command line (its options and the prompt, as
        a container's arguments), and optionally a saved session to continue,
        stdin (the prompt when the command line has none) and a session ID,
        under which the session stays open for more turns."""
        from .model import generate

        args = generate.options(request.get("argv", []), model=request["model"])
        if request.get("session"):
            args.session = Path(request["session"])
        model, tokenizer = self.get(Path(request["model"]))
        session = generate.start(model, tokenizer, args, err=_Stream(write), vcpu_socket=self.socket)
        prompt = " ".join(args.prompt) or request.get("stdin", "")
        with _all_cores():
            if session is not None and prompt:
                generate.converse(session, tokenizer, prompt, args.max_tokens, out=_Stream(write), err=_Stream(write))
        if session is not None and request.get("session_id"):
            self.sessions[request["session_id"]] = Live(tokenizer, args, session)

    def turn(self, request: dict, write) -> None:
        """Another turn of a live session: the user's text, from the command
        line in `argv` (as `docker exec` gives it) or `text`."""
        from .model import generate

        live = self._live(request)
        args = generate.options(request.get("argv", []), model="-")
        text = " ".join(args.prompt) or request.get("text", "")
        if not text.strip():
            return
        max_tokens = args.max_tokens if "--max-tokens" in request.get("argv", []) else live.args.max_tokens
        with live.lock, _all_cores():
            generate.converse(live.session, live.tokenizer, text, max_tokens, out=_Stream(write), err=_Stream(write))

    def save(self, request: dict) -> bytes:
        """A live session's files, as `docker commit` freezes them into the
        container's /.shem: each as a u32 name length, the name, a u32 data
        length and the data."""
        live = self._live(request)
        with live.lock:
            files = live.session.to_files()
        out = b""
        for name, data in files.items():
            out += struct.pack("<I", len(name)) + name.encode() + struct.pack("<I", len(data)) + data
        return out

    def close(self, request: dict) -> None:
        self.sessions.pop(request["session_id"], None)

    def _live(self, request: dict) -> "Live":
        try:
            return self.sessions[request["session_id"]]
        except KeyError:
            raise ValueError(f"no live session {request.get('session_id')!r}: is the container running?") from None


class Live:
    """A container's session, open for more turns."""

    def __init__(self, tokenizer, args, session):
        self.tokenizer, self.args, self.session = tokenizer, args, session
        self.lock = threading.Lock()


class _Stream:
    """Text written to it goes to the client as it comes."""

    def __init__(self, write):
        self._write = write

    def write(self, text: str) -> None:
        if text:
            self._write(text.encode())

    def flush(self) -> None:
        pass


class _all_cores:
    """Use every core for a model's big matrices; vCPUs' tiny tensors use one."""

    def __enter__(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(os.cpu_count() or 1)

    def __exit__(self, *_):
        torch.set_num_threads(self.threads)


class Job:
    """A `RUN` in flight: a vCPU, how far it may go, and how far it has got."""

    def __init__(self, vcpu: NeuralVcpu, budget: int):
        self.vcpu = vcpu
        self.budget = budget
        self.steps = 0
        self.result: tuple[int, int, int] | None = None
        self.done = threading.Event()


class Batcher:
    """Batch compatible `RUN` requests on one executor thread.

    Exited or budget-exhausted guests stop committing. New arrivals trigger
    unstacking and rebuilding the batch between steps.
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
                jobs = [j for j in self.jobs if j.vcpu.mem_size == first.mem_size and j.vcpu.cpu is first.cpu]
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
        #: CPUs from compiled layers that images carry, by their weights' digest
        self.cpus: dict[str, NeuralCpu] = {hashlib.sha256(_tensors_bytes(weights)).hexdigest(): self.cpu}
        self.dense = dense
        self.vcpus: dict[int, NeuralVcpu] = {}
        self.pools: dict[int, PagePool] = {}  # by memory size
        self.ids = itertools.count(1)
        self.lock = threading.RLock()
        self.batcher = Batcher(self.lock)

    def cpu_for(self, path: Path) -> NeuralCpu:
        """The CPU whose weights are the compiled layer at `path` (its
        `model.safetensors`, or the directory holding it). Images built by the
        same shem carry this server's own weights, and share its CPU."""
        from .compile import load

        if path.is_dir():
            path = path / "model.safetensors"
        weights = load(path)
        if set(weights) != set(self.cpu.w):
            raise ValueError(f"{path}: not the weights of a compiled RV32IM CPU")
        digest = hashlib.sha256(_tensors_bytes(weights)).hexdigest()
        if digest not in self.cpus:
            self.cpus[digest] = NeuralCpu(weights, fused=not self.dense)
        return self.cpus[digest]

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
            weights = payload[9:].decode() if len(payload) > 9 else ""
            cpu = self.cpu_for(Path(weights)) if weights else self.cpu
            pool = None if self.dense else self.pools.setdefault(mem_size, PagePool())
            return struct.pack("<I", self.add(NeuralVcpu(mem_size, cpu, self.dense, pool, base)))
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


def _tensors_bytes(weights: dict[str, torch.Tensor]) -> bytes:
    """The weights' names and values, for a digest."""
    return b"".join(name.encode() + weights[name].contiguous().numpy().tobytes() for name in sorted(weights))


def read_frame(stream) -> bytes | None:
    header = stream.read(4)
    if len(header) < 4:
        return None
    (length,) = struct.unpack("<I", header)
    return stream.read(length)


def write_frame(stream, payload: bytes) -> None:
    stream.write(struct.pack("<I", len(payload)) + payload)
    stream.flush()


def make_server(
    socket_path: Path, dense: bool = False, device: str = "auto", single_thread: bool = True
) -> tuple[socketserver.BaseServer, Registry]:
    """A server on `socket_path`, not yet serving, and its registry of vCPUs."""
    if single_thread:
        torch.set_num_threads(1)  # tiny tensors: threads only add overhead
    registry = Registry(dense)
    models = Models(device, socket_path)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while (payload := read_frame(self.rfile)) is not None:
                try:
                    op, request = payload[0], payload[1:]
                    chunk = lambda data: write_frame(self.wfile, bytes([Status.CHUNK]) + data)  # noqa: E731
                    if op == Op.GENERATE:
                        models.generate(json.loads(request), chunk)
                        response = bytes([Status.OK])
                    elif op == Op.TURN:
                        models.turn(json.loads(request), chunk)
                        response = bytes([Status.OK])
                    elif op == Op.SAVE:
                        response = bytes([Status.OK]) + models.save(json.loads(request))
                    elif op == Op.CLOSE:
                        models.close(json.loads(request))
                        response = bytes([Status.OK])
                    else:
                        response = bytes([Status.OK]) + registry.handle(payload)
                except MemoryFault as fault:
                    response = bytes([Status.MEMORY_FAULT]) + str(fault).encode()
                except Exception as error:  # report anything else to the host
                    response = bytes([Status.ERROR]) + f"{type(error).__name__}: {error}".encode()
                write_frame(self.wfile, response)

    # Listen under a temporary name, then move the socket into place, so that
    # clients waiting for it to appear never find it before it accepts them
    staging = socket_path.with_name(f".{socket_path.name}.{os.getpid()}")
    staging.unlink(missing_ok=True)
    server = socketserver.ThreadingUnixStreamServer(str(staging), Handler)
    server.daemon_threads = True
    # Only this user, and root (as the containerd shim runs), may connect
    os.chmod(staging, 0o600)
    os.replace(staging, socket_path)
    return server, registry


def serve(socket_path: Path, dense: bool = False, device: str = "auto") -> None:
    server, _ = make_server(socket_path, dense, device)
    with server:
        print(f"shem-neural: serving vCPUs on {socket_path}", file=sys.stderr, flush=True)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", type=Path, required=True, help="Unix socket to listen on")
    parser.add_argument("--dense", action="store_true", help="run the network densely, without fused kernels")
    parser.add_argument("--device", default="auto", help="where models run: cpu, cuda, or auto (a GPU if there is one)")
    args = parser.parse_args()
    try:
        serve(args.socket, args.dense, args.device)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
