"""Serve neural vCPUs to the shem host over a Unix socket.

The protocol is the `Vcpu` interface as requests, KVM-style: create a vCPU, run
it until it exits, read and write its registers and memory, and fork it by
cloning. Every message is a frame: a little-endian u32 length, then the
payload. A request payload is an op byte and its arguments; a response
payload is a status byte (see `Status`) and the result. `docs/neural.md` lists
the ops.

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
from .cpu import NeuralCpu
from .vcpu import MemoryFault, NeuralVcpu


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


class Registry:
    def __init__(self, dense: bool):
        weights = compile_weights()
        self.cpu = NeuralCpu(weights, fused=not dense)
        self.dense = dense
        self.vcpus: dict[int, NeuralVcpu] = {}
        self.ids = itertools.count(1)
        self.lock = threading.Lock()

    def add(self, vcpu: NeuralVcpu) -> int:
        with self.lock:
            vcpu_id = next(self.ids)
            self.vcpus[vcpu_id] = vcpu
            return vcpu_id

    def handle(self, payload: bytes) -> bytes:
        op = Op(payload[0])
        if op == Op.CREATE:
            (mem_size,) = struct.unpack_from("<I", payload, 1)
            return struct.pack("<I", self.add(NeuralVcpu(mem_size, self.cpu, self.dense)))
        (vcpu_id,) = struct.unpack_from("<I", payload, 1)
        args = payload[5:]
        vcpu = self.vcpus[vcpu_id]
        match op:
            case Op.DROP:
                with self.lock:
                    del self.vcpus[vcpu_id]
                return b""
            case Op.CLONE:
                return struct.pack("<I", self.add(vcpu.clone()))
            case Op.RUN:
                (budget,) = struct.unpack("<Q", args)
                reason, info, steps = vcpu.run(budget)
                return struct.pack("<BIIQ", reason, info, vcpu.pc, steps)
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


def serve(socket_path: Path, dense: bool = False) -> None:
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
    with socketserver.ThreadingUnixStreamServer(str(socket_path), Handler) as server:
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
