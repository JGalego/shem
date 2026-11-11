import socket
import struct
import tempfile
import threading
from pathlib import Path

import pytest
from rv32 import A0, EBREAK, T0, ZERO, add, addi, assemble, bne, li

from shem_neural.server import Op, Status, make_server
from shem_neural.vcpu import EXIT_EBREAK

MEM = 64 << 10


class Client:
    def __init__(self, path: Path):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(str(path))
        self.stream = self.sock.makefile("rwb")

    def call(self, op: Op, payload: bytes = b"") -> bytes:
        frame = bytes([op]) + payload
        self.stream.write(struct.pack("<I", len(frame)) + frame)
        self.stream.flush()
        (length,) = struct.unpack("<I", self.stream.read(4))
        reply = self.stream.read(length)
        assert reply[0] == Status.OK, reply[1:].decode()
        return reply[1:]

    def create(self) -> int:
        return struct.unpack("<I", self.call(Op.CREATE, struct.pack("<I", MEM)))[0]


@pytest.fixture
def server():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "neural.sock"
        srv, registry = make_server(path)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        yield path, registry
        srv.shutdown()
        srv.server_close()


def sum_to(n: int) -> bytes:
    """a0 = 1 + 2 + ... + n, in 3n + 3 instructions before its final ebreak."""
    return assemble(addi(A0, ZERO, 0), li(T0, n), add(A0, A0, T0), addi(T0, T0, -1), bne(T0, ZERO, -8), EBREAK)


def test_concurrent_runs_are_batched_and_exact(server):
    path, registry = server
    sizes = [40, 70, 100, 130, 160, 190]
    start = threading.Barrier(len(sizes))
    results = {}

    def client(n: int):
        c = Client(path)
        vcpu = c.create()
        c.call(Op.WRITE_MEM, struct.pack("<II", vcpu, 0) + sum_to(n))
        start.wait()
        reason, _, pc, steps = struct.unpack("<BIIQ", c.call(Op.RUN, struct.pack("<IQ", vcpu, 100_000)))
        a0 = struct.unpack("<I", c.call(Op.GET_REG, struct.pack("<IB", vcpu, A0)))[0]
        results[n] = (reason, steps, a0)

    threads = [threading.Thread(target=client, args=(n,)) for n in sizes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for n in sizes:
        assert results[n] == (EXIT_EBREAK, 3 * n + 3, n * (n + 1) // 2)
    assert registry.batcher.largest > 1, "the runs should have shared a batch"


def test_budgets_end_runs_without_losing_progress(server):
    path, _ = server
    c = Client(path)
    vcpu = c.create()
    c.call(Op.WRITE_MEM, struct.pack("<II", vcpu, 0) + sum_to(50))
    total = 0
    while True:
        reason, _, _, steps = struct.unpack("<BIIQ", c.call(Op.RUN, struct.pack("<IQ", vcpu, 37)))
        total += steps
        if reason == EXIT_EBREAK:
            break
    assert total == 3 * 50 + 3
    assert struct.unpack("<I", c.call(Op.GET_REG, struct.pack("<IB", vcpu, A0)))[0] == 1275


def test_clones_and_restored_snapshots_share_pages(server):
    path, registry = server
    c = Client(path)
    parent = c.create()
    c.call(Op.WRITE_MEM, struct.pack("<II", parent, 0) + bytes(range(256)) * 64)  # 4 pages
    pool = registry.pools[MEM]
    before = pool.in_use
    clones = [struct.unpack("<I", c.call(Op.CLONE, struct.pack("<I", parent)))[0] for _ in range(20)]
    assert pool.in_use == before, "clones copy page tables, not pages"

    snapshot = c.call(Op.SNAPSHOT, struct.pack("<I", parent))
    for _ in range(20):
        fresh = c.create()
        c.call(Op.RESTORE, struct.pack("<I", fresh) + snapshot)
    assert pool.in_use == before, "restored pages are found by content and shared"

    for vcpu in clones:
        c.call(Op.DROP, struct.pack("<I", vcpu))
    assert pool.in_use == before
