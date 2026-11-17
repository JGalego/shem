"""The C extension: compressed (16-bit) instructions as the table the neural
vCPU looks them up in.

A 16-bit instruction means one 32-bit instruction of the base set, so the
vCPU needs no units of its own for them: its fetch reads the halfword's row of
this table, a layer of 65,536 rows of 32 bits, and goes on with that as the
instruction. `expand` is the same function as the reference vCPU's
(`expand_compressed` in `crates/shem/src/reference.rs`), and a digest of the
whole table, which both test against, says they are one.
"""

import hashlib
import struct

import torch

_TABLE: torch.Tensor | None = None


def _sext(value: int, bits: int) -> int:
    return value - (1 << bits) if value >> (bits - 1) & 1 else value


def expand(h: int) -> int | None:
    """The 32-bit instruction that the compressed instruction `h` stands for, or
    None if it is illegal, or one of the floating-point and RV64 forms RV32IM has none of."""
    rd_full = h >> 7 & 31
    rs2_full = h >> 2 & 31
    rd3 = 8 + (h >> 2 & 7)
    rs1_3 = 8 + (h >> 7 & 7)
    imm6 = _sext((h >> 7 & 0x20) | (h >> 2 & 0x1F), 6) & 0xFFFFFFFF

    def i_type(imm, rs1, f3, rd, op):
        return (imm & 0xFFF) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op

    def r_type(f7, rs2, rs1, f3, rd):
        return f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | 0x33

    def s_type(imm, rs2, rs1):
        return (imm >> 5 & 0x7F) << 25 | rs2 << 20 | rs1 << 15 | 2 << 12 | (imm & 0x1F) << 7 | 0x23

    def b_type(imm, rs1, f3):
        return (
            (imm >> 12 & 1) << 31
            | (imm >> 5 & 0x3F) << 25
            | rs1 << 15
            | f3 << 12
            | (imm >> 1 & 0xF) << 8
            | (imm >> 11 & 1) << 7
            | 0x63
        )

    def j_type(imm, rd):
        return (
            (imm >> 20 & 1) << 31
            | (imm >> 1 & 0x3FF) << 21
            | (imm >> 11 & 1) << 20
            | (imm >> 12 & 0xFF) << 12
            | rd << 7
            | 0x6F
        )

    j_imm = (
        h >> 1 & 0x800
        | h << 2 & 0x400
        | h >> 1 & 0x300
        | h << 1 & 0x80
        | h >> 1 & 0x40
        | h << 3 & 0x20
        | h >> 7 & 0x10
        | h >> 2 & 0xE
    )
    j_imm = _sext(j_imm, 12) & 0xFFFFFFFF
    b_imm = h >> 4 & 0x100 | h << 1 & 0xC0 | h << 3 & 0x20 | h >> 7 & 0x18 | h >> 2 & 0x6
    b_imm = _sext(b_imm, 9) & 0xFFFFFFFF
    lw_imm = h >> 7 & 0x38 | h >> 4 & 0x4 | h << 1 & 0x40
    shamt = h >> 2 & 0x1F
    shamt_ok = h & 0x1000 == 0  # a shift past 31 is RV64's

    quadrant, funct = h & 3, h >> 13
    if (quadrant, funct) == (0, 0):  # C.ADDI4SPN
        imm = h >> 7 & 0x30 | h >> 1 & 0x3C0 | h >> 4 & 0x4 | h >> 2 & 0x8
        return i_type(imm, 2, 0, rd3, 0x13) if imm else None
    if (quadrant, funct) == (0, 2):  # C.LW
        return i_type(lw_imm, rs1_3, 2, rd3, 0x03)
    if (quadrant, funct) == (0, 6):  # C.SW
        return s_type(lw_imm, rd3, rs1_3)
    if (quadrant, funct) == (1, 0):  # C.ADDI, C.NOP
        return i_type(imm6, rd_full, 0, rd_full, 0x13)
    if (quadrant, funct) == (1, 1):  # C.JAL
        return j_type(j_imm, 1)
    if (quadrant, funct) == (1, 2):  # C.LI
        return i_type(imm6, 0, 0, rd_full, 0x13)
    if (quadrant, funct) == (1, 3):
        if rd_full == 2:  # C.ADDI16SP
            imm = h >> 3 & 0x200 | h >> 2 & 0x10 | h << 1 & 0x40 | h << 4 & 0x180 | h << 3 & 0x20
            imm = _sext(imm, 10) & 0xFFFFFFFF
            return i_type(imm, 2, 0, 2, 0x13) if imm else None
        # C.LUI
        return ((imm6 << 12) & 0xFFFFF000 | rd_full << 7 | 0x37) if imm6 else None
    if (quadrant, funct) == (1, 4):
        rd = rs1_3
        kind = h >> 10 & 3
        if kind == 0:  # C.SRLI
            return i_type(shamt, rd, 5, rd, 0x13) if shamt_ok else None
        if kind == 1:  # C.SRAI
            return i_type(0x400 | shamt, rd, 5, rd, 0x13) if shamt_ok else None
        if kind == 2:  # C.ANDI
            return i_type(imm6, rd, 7, rd, 0x13)
        if h & 0x1000:
            return None
        f7, f3 = [(0x20, 0), (0, 4), (0, 6), (0, 7)][h >> 5 & 3]
        return r_type(f7, rd3, rd, f3, rd)  # C.SUB, C.XOR, C.OR, C.AND
    if (quadrant, funct) == (1, 5):  # C.J
        return j_type(j_imm, 0)
    if (quadrant, funct) == (1, 6):  # C.BEQZ
        return b_type(b_imm, rs1_3, 0)
    if (quadrant, funct) == (1, 7):  # C.BNEZ
        return b_type(b_imm, rs1_3, 1)
    if (quadrant, funct) == (2, 0):  # C.SLLI
        return i_type(shamt, rd_full, 1, rd_full, 0x13) if shamt_ok else None
    if (quadrant, funct) == (2, 2):  # C.LWSP
        imm = h >> 2 & 0x1C | h >> 7 & 0x20 | h << 4 & 0xC0
        return i_type(imm, 2, 2, rd_full, 0x03) if rd_full else None
    if (quadrant, funct) == (2, 4):
        if h & 0x1000 == 0:
            if rs2_full == 0:  # C.JR
                return i_type(0, rd_full, 0, 0, 0x67) if rd_full else None
            return r_type(0, rs2_full, 0, 0, rd_full)  # C.MV
        if rs2_full == 0:
            return 0x00100073 if rd_full == 0 else i_type(0, rd_full, 0, 1, 0x67)  # C.EBREAK, C.JALR
        return r_type(0, rs2_full, rd_full, 0, rd_full)  # C.ADD
    if (quadrant, funct) == (2, 6):  # C.SWSP
        return s_type(h >> 7 & 0x3C | h >> 1 & 0xC0, rs2_full, 2)
    return None


def digest() -> str:
    """A SHA-256 of every halfword's expansion, as four little-endian bytes and 0xFFFFFFFF for none."""
    sha = hashlib.sha256()
    for half in range(1 << 16):
        word = expand(half)
        sha.update(struct.pack("<I", 0xFFFFFFFF if word is None else word))
    return sha.hexdigest()


def table(device: torch.device | str = "cpu") -> torch.Tensor:
    """The expansions as bits, (65536, 32) bytes, an illegal one as zeros (the
    all-zero word, which the base set has no instruction for)."""
    global _TABLE
    if _TABLE is None:
        words = torch.tensor([expand(half) or 0 for half in range(1 << 16)], dtype=torch.int64)
        _TABLE = (words[:, None] >> torch.arange(32) & 1).to(torch.uint8)
    return _TABLE.to(device)
