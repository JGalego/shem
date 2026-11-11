"""A tiny RV32IM assembler for tests."""

ZERO, RA, SP, T0, T1, T2, S0 = 0, 1, 2, 5, 6, 7, 8
A0, A1, A2, A3, A7 = 10, 11, 12, 13, 17
ECALL = 0x0000_0073
EBREAK = 0x0010_0073


def r(f7: int, f3: int, rd: int, rs1: int, rs2: int) -> int:
    return f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | 0x33


def i(op: int, f3: int, rd: int, rs1: int, imm: int) -> int:
    return (imm & 0xFFF) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op


def s(f3: int, rs2: int, rs1: int, imm: int) -> int:
    return (imm >> 5 & 0x7F) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (imm & 0x1F) << 7 | 0x23


def b(f3: int, rs1: int, rs2: int, offset: int) -> int:
    o = offset & 0x1FFF
    return (
        (o >> 12 & 1) << 31 | (o >> 5 & 0x3F) << 25 | rs2 << 20 | rs1 << 15
        | f3 << 12 | (o >> 1 & 0xF) << 8 | (o >> 11 & 1) << 7 | 0x63
    )  # fmt: skip


def addi(rd, rs1, imm):
    return i(0x13, 0, rd, rs1, imm)


def add(rd, rs1, rs2):
    return r(0, 0, rd, rs1, rs2)


def lw(rd, rs1, imm):
    return i(0x03, 2, rd, rs1, imm)


def lb(rd, rs1, imm):
    return i(0x03, 0, rd, rs1, imm)


def lbu(rd, rs1, imm):
    return i(0x03, 4, rd, rs1, imm)


def sw(rs2, rs1, imm):
    return s(2, rs2, rs1, imm)


def sb(rs2, rs1, imm):
    return s(0, rs2, rs1, imm)


def bne(rs1, rs2, offset):
    return b(1, rs1, rs2, offset)


def lui(rd, upper):
    return (upper & 0xFFFFF000) | rd << 7 | 0x37


def jal(rd: int, offset: int) -> int:
    o = offset & 0x1FFFFF
    return (o >> 20 & 1) << 31 | (o >> 1 & 0x3FF) << 21 | (o >> 11 & 1) << 20 | (o >> 12 & 0xFF) << 12 | rd << 7 | 0x6F


def li(rd: int, value: int) -> list[int]:
    value &= 0xFFFF_FFFF
    low = ((value & 0xFFF) ^ 0x800) - 0x800
    return [lui(rd, (value - low) & 0xFFFF_FFFF), addi(rd, rd, low)]


def assemble(*items) -> bytes:
    words: list[int] = []
    for item in items:
        words.extend(item if isinstance(item, list) else [item])
    return b"".join(w.to_bytes(4, "little") for w in words)


M_OPS = {"mul": 0, "mulh": 1, "mulhsu": 2, "mulhu": 3, "div": 4, "divu": 5, "rem": 6, "remu": 7}


def m(op: str, rd: int, rs1: int, rs2: int) -> int:
    return r(1, M_OPS[op], rd, rs1, rs2)
