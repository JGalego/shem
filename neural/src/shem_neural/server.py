"""Serve neural vCPUs and model sessions over a Unix socket.

vCPUs run on the CPU, sharing pages by memory size and batching runs by CPU
weights and memory size. Model sessions stream replies and cache loaded weights
on the selected model device.

Frames are a little-endian u32 length followed by a payload. Requests start
with an op byte, responses with a `Status` byte. See `docs/neural.md`.

    uv run python -m shem_neural.server [--socket PATH]
"""

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import socketserver
import struct
import sys
import threading
import time
import uuid
from enum import IntEnum
from pathlib import Path

import torch

from .compile import compile_weights
from .cpu import DTYPE, NeuralCpu, State
from .memory import PagePool
from .vcpu import _EXIT_CODES, BUDGET, MemoryFault, NeuralVcpu


def default_socket() -> Path:
    """Where the server listens when not told: for root, as the containerd
    shim runs, `/run/shem/neural.sock`; for a user, in their runtime
    directory, or else under `/tmp` by their ID. Not a fixed path in a
    directory everyone can write to, where another user could listen first.
    The same as `shem::neural::default_socket`, which clients try."""
    uid = os.geteuid()
    if uid == 0:
        return Path("/run/shem/neural.sock")
    if runtime := os.environ.get("XDG_RUNTIME_DIR"):
        return Path(runtime) / "shem" / "neural.sock"
    return Path(f"/tmp/shem-neural-{uid}.sock")


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
    """Loaded models, by where they are and the type they compute in, so
    the sessions of a container share its model's weights; and the live
    sessions of running containers, by the ID the shim gives them. A model is
    not shared between containers: two images may hold the same manifest and
    different weights or templates, and a container must get its own. It is
    released when the last container using it closes."""

    def __init__(self, device: str = "auto", socket: Path | None = None, linger: float = 0.0):
        self.device = device
        #: Seconds a model stays loaded after its last user is done, so that
        #: the next run of it does not read its weights again
        self.linger = linger
        #: When each model nobody uses was last used, on the monotonic clock
        self.idle: dict[str, float] = {}
        #: A lock for each model being loaded, so that two containers of one model load it once
        self.loading: dict[str, threading.Lock] = {}
        #: This server's socket, where tools' programs run on its vCPUs
        self.socket = socket
        self.loaded: dict[str, object] = {}
        self.sessions: dict[str, Live] = {}
        #: Containers' APIs, served in their networks, by the ID the shim gives them
        self.served: dict[str, object] = {}
        #: The session IDs (or one-off runs) using each loaded model
        self.users: dict[str, set[str]] = {}
        self.lock = threading.Lock()

    def get(
        self,
        directory: Path,
        dtype: str = "float32",
        user: str = "",
        quantize: str | None = None,
        adapters: list[Path] | None = None,
    ):
        """The model in `directory`, computing in `dtype`, and its tokenizer,
        for `user`, a session ID, until `release(user)`.

        Containers of images with the same weights share one model: it is
        found by the fingerprint its manifest records, which it was checked
        against as it loaded, so an image that claims another's weights
        cannot make a model of its own pass for them: one whose weights are
        not the ones its fingerprint names gets a model that is its alone. Each image's tokenizer
        and chat template are its own. A model with no fingerprint (one
        imported before them) is its directory's alone. Loading takes the
        time it takes, and holds up only the others that want that model."""
        from .model.blocks import FingerprintMismatch, Model
        from .model.generate import dtype as torch_dtype
        from .model.manifest import Manifest
        from .model.tokenizer import Tokenizer

        manifest = Manifest.load(directory)
        shared = bool(manifest.fingerprint and all(spec.file is None for spec in manifest.blocks))

        def load(shared: bool):
            identity = manifest.fingerprint if shared else str(Path(directory).resolve())
            key = f"{identity}\0{dtype}\0{quantize or ''}\0{[str(a) for a in adapters or []]}"
            with self.lock:
                # Claimed first, so that a model being loaded is not released from under its loader
                self.idle.pop(key, None)
                self.users.setdefault(key, set()).add(user)
                loading = self.loading.setdefault(key, threading.Lock())
            try:
                with loading:
                    with self.lock:
                        found = self.loaded.get(key)
                    if found is None:
                        found = Model(
                            directory,
                            dtype=torch_dtype(dtype),
                            device=self.device,
                            quantize=quantize,
                            verify=shared,
                            adapters=adapters,
                        )
                        with self.lock:
                            self.loaded[key] = found
            except BaseException:
                self.release(user)
                raise
            return found

        try:
            model = load(shared)
        except FingerprintMismatch:
            # Not the weights its fingerprint names, or a fingerprint another version made: it runs
            # what it holds, in a model of its own that no other image can be given
            model = load(False)
        return model, Tokenizer(directory, manifest.tokenizer)

    def release(self, user: str) -> None:
        """Forget `user`'s models, and unload those nobody else uses."""
        with self.lock:
            for key in [key for key, users in self.users.items() if user in users]:
                self.users[key].discard(user)
                if not self.users[key]:
                    del self.users[key]
                    if self.linger > 0:
                        self.idle[key] = time.monotonic()
                    else:
                        self.loaded.pop(key, None)
                        self.loading.pop(key, None)

    def evict(self, now: float | None = None) -> list[str]:
        """Unload the models that nobody has used for `linger` seconds, and
        return their keys."""
        now = time.monotonic() if now is None else now
        with self.lock:
            gone = [key for key, since in self.idle.items() if now - since >= self.linger]
            for key in gone:
                del self.idle[key]
                self.loaded.pop(key, None)
                self.loading.pop(key, None)
        return gone

    def generate(self, request: dict, write) -> None:
        """Start a session for `request` and reply to its prompt, writing the
        reply with `write` as it comes. The request has the model's
        directory, the driver's command line (its options and the prompt, as
        a container's arguments), and optionally a saved session to continue,
        stdin (the prompt when the command line has none) and a session ID,
        under which the session stays open for more turns."""
        from .model import generate

        args = generate.resolve(generate.options(request.get("argv", []), model=request["model"]))
        # This server may be root, and a container's command line is the
        # image's: it names no host paths to write or read sessions or receipts at
        if args.warm or args.session or args.receipt or args.replay:
            raise ValueError("a container's command line cannot give --warm, --session, --receipt or --replay")
        if request.get("session"):
            args.session = Path(request["session"])
        user = request.get("session_id") or f"run-{uuid.uuid4()}"
        model, tokenizer = self.get(
            args.model, args.dtype, user, args.quantize, generate.adapter_paths(args.model, args.root, args.adapter)
        )
        kept = False
        try:
            if args.serve:
                kept = True
                return self._serve(request, model, tokenizer, args, write)
            session = generate.start(model, tokenizer, args, err=_Stream(write), vcpu_socket=self.socket)
            prompt = " ".join(args.prompt) or request.get("stdin", "")
            with _all_cores():
                if session is not None and prompt:
                    generate.converse(
                        session, tokenizer, prompt, args.max_tokens, _Stream(write), _Stream(write), args.stop
                    )
            if session is not None and request.get("session_id"):
                self.sessions[request["session_id"]] = Live(tokenizer, args, session)
                kept = True
        finally:
            if not kept and not request.get("session_id"):
                self.release(user)

    def _serve(self, request: dict, model, tokenizer, args, write) -> None:
        """Serve OpenAI's chat completions API at the container's `--serve`
        port, in its network namespace, `netns` in the request, until the
        container ends; its primed session, if it has one, starts replies."""
        from .model import generate, serve

        primed = generate.Session.load(model, args.session, _Stream(write)) if args.session else None
        chat = serve.Chat(model, tokenizer, args, primed, compute=_all_cores)
        host, port = serve.address(args.serve, "0.0.0.0")
        server = serve.start(chat, serve.listen(host, port, request.get("netns")))
        port = server.server_address[1]
        self.served[request.get("session_id") or f"{host}:{port}"] = server
        _Stream(write).write(f"shem: serving {chat.name} at port {port}, as OpenAI's API at /v1\n")

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
        stops = args.stop or live.args.stop
        with live.lock, _all_cores():
            generate.converse(live.session, live.tokenizer, text, max_tokens, _Stream(write), _Stream(write), stops)

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
        if served := self.served.pop(request["session_id"], None):
            served.shutdown()
            served.server_close()
        self.release(request["session_id"])

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
        #: Why the job failed, if it did
        self.error: Exception | None = None
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
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return job.result

    def _loop(self) -> None:
        while True:
            with self.wake:
                while not self.jobs:
                    self.wake.wait()
                head = self.jobs[0]
                first = head.vcpu
                jobs = [head]
                arrivals = self.arrivals
            # A job that fails fails alone: the executor goes on for the others
            try:
                if not first.dense:  # the dense network's memory does not batch
                    size = first.mem_size
                    with self.lock:
                        jobs += [j for j in self.jobs if j is not head and _batches_with(j.vcpu, first, size)]
                        arrivals = self.arrivals
                if first.dense:
                    with self.lock:
                        jobs[0].result = first.run(jobs[0].budget)
                        self.jobs.remove(jobs[0])
                        jobs[0].done.set()
                    continue
                with self.lock:
                    batch = State.stack([j.vcpu.state for j in jobs])
                    self.largest = max(self.largest, len(jobs))
                self._step(jobs, batch, arrivals)
            except Exception as error:
                with self.lock:
                    for job in jobs:
                        if not job.done.is_set():
                            job.error = error
                            if job in self.jobs:
                                self.jobs.remove(job)
                            job.done.set()

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


def _batches_with(vcpu: NeuralVcpu, first: NeuralVcpu, size: int) -> bool:
    """Whether `vcpu` can step in a batch with `first`: a vCPU that cannot
    say is left to fail on its own turn."""
    try:
        return not vcpu.dense and vcpu.mem_size == size and vcpu.cpu is first.cpu
    except Exception:
        return False


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
            if mem_size == 0:
                raise ValueError("a vCPU needs some memory")
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


class Activity:
    """When a server was last used: how many connections it has open, and
    since when it has had none."""

    def __init__(self):
        self.lock = threading.Lock()
        self.open = 0
        self.since = time.monotonic()

    def __enter__(self):
        with self.lock:
            self.open += 1

    def __exit__(self, *_):
        with self.lock:
            self.open -= 1
            self.since = time.monotonic()

    def idle_for(self) -> float:
        """Seconds with no connection open, or 0 while there is one."""
        with self.lock:
            return 0.0 if self.open else time.monotonic() - self.since


def make_server(
    socket_path: Path,
    dense: bool = False,
    device: str = "auto",
    single_thread: bool = True,
    linger: float = 0.0,
) -> tuple[socketserver.BaseServer, Registry]:
    """A server on `socket_path`, not yet serving, and its registry of vCPUs.
    Models stay loaded `linger` seconds after they were last used."""
    if single_thread:
        torch.set_num_threads(1)  # tiny tensors: threads only add overhead
    registry = Registry(dense)
    models = Models(device, socket_path, linger)
    activity = Activity()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            with activity:
                self.serve()

        def serve(self):
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
    if not socket_path.parent.exists():
        socket_path.parent.mkdir(mode=0o755 if os.geteuid() == 0 else 0o700, parents=True, exist_ok=True)
    staging.unlink(missing_ok=True)
    server = socketserver.ThreadingUnixStreamServer(str(staging), Handler)
    server.daemon_threads = True
    # Only this user, and root (as the containerd shim runs), may connect
    os.chmod(staging, 0o600)
    os.replace(staging, socket_path)
    server.models, server.activity = models, activity
    return server, registry


def serve(
    socket_path: Path, dense: bool = False, device: str = "auto", keep_warm: float = 0.0, idle_exit: float = 0.0
) -> None:
    """Serve until interrupted, or, with `idle_exit`, until that many seconds
    have passed with no connection open: a server started on demand by
    `shem run`, which would otherwise stay for good."""
    server, _ = make_server(socket_path, dense, device, linger=keep_warm)
    stopped = threading.Event()

    def watch() -> None:
        while not stopped.wait(min(5.0, max(0.5, keep_warm / 4 or 5.0))):
            if keep_warm:
                server.models.evict()
            if idle_exit and server.activity.idle_for() >= idle_exit:
                server.shutdown()
                return

    with server:
        print(f"shem-neural: serving vCPUs on {socket_path}", file=sys.stderr, flush=True)
        if keep_warm or idle_exit:
            threading.Thread(target=watch, daemon=True).start()
        try:
            server.serve_forever()
        finally:
            stopped.set()
            # Leave no socket behind for a client to find and fail to reach
            with contextlib.suppress(OSError):
                if socket_path.is_socket():
                    socket_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--socket",
        type=Path,
        default=None,
        help="Unix socket to listen on (default: root's /run/shem/neural.sock, else the user's)",
    )
    parser.add_argument("--dense", action="store_true", help="run the network densely, without fused kernels")
    parser.add_argument("--device", default="auto", help="where models run: cpu, cuda, or auto (a GPU if there is one)")
    parser.add_argument(
        "--keep-warm",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="keep a model loaded this long after its last use",
    )
    parser.add_argument(
        "--idle-exit", type=float, default=0.0, metavar="SECONDS", help="exit after this long with no connection open"
    )
    args = parser.parse_args()
    try:
        serve(args.socket or default_socket(), args.dense, args.device, args.keep_warm, args.idle_exit)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
