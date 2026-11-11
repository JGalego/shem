"""RV32IM, described as data for the weight compiler.

Each instruction is a bit pattern, a mask and a match over the 32-bit word, plus
what it does in terms the network understands: where its immediate comes from,
which unit's result goes to rd, how the next pc is chosen, whether it touches
memory, and whether it exits to the host. The set and the patterns match the
reference vCPU (`crates/shem/src/reference.rs`): anything else is illegal.
"""

from dataclasses import dataclass

FORMATS = ("I", "S", "B", "U", "J")
RESULTS = (
    "imm", "auipc", "link", "add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and",
    "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu",
    "lb", "lh", "lw", "lbu", "lhu",
)  # fmt: skip
NEXT_PC = ("seq", "branch", "jal", "jalr")
BRANCHES = ("eq", "ne", "lt", "ge", "ltu", "geu")
EXITS = ("ecall", "ebreak")

OPCODE = 0x0000_007F
FUNCT3 = 0x0000_7000
FUNCT7 = 0xFE00_0000


@dataclass(frozen=True)
class Instruction:
    name: str
    mask: int
    match: int
    format: str | None = None
    result: str | None = None  # the unit whose result goes to rd; None if rd is not written
    uses_imm: bool = False  # the second operand is the immediate rather than rs2
    next_pc: str = "seq"
    branch: str | None = None
    load: int = 0  # bytes loaded
    store: int = 0  # bytes stored
    exit: str | None = None


def _r(name: str, funct3: int, funct7: int, result: str) -> Instruction:
    match = funct7 << 25 | funct3 << 12 | 0x33
    return Instruction(name, OPCODE | FUNCT3 | FUNCT7, match, result=result)


def _i(name: str, opcode: int, funct3: int, **what) -> Instruction:
    match = funct3 << 12 | opcode
    return Instruction(name, OPCODE | FUNCT3, match, format="I", uses_imm=True, **what)


def _shift(name: str, funct3: int, funct7: int, result: str) -> Instruction:
    match = funct7 << 25 | funct3 << 12 | 0x13
    return Instruction(name, OPCODE | FUNCT3 | FUNCT7, match, "I", result, uses_imm=True)


def _branch(name: str, funct3: int, kind: str) -> Instruction:
    match = funct3 << 12 | 0x63
    return Instruction(name, OPCODE | FUNCT3, match, "B", next_pc="branch", branch=kind)


def _store(name: str, funct3: int, size: int) -> Instruction:
    match = funct3 << 12 | 0x23
    return Instruction(name, OPCODE | FUNCT3, match, "S", uses_imm=True, store=size)


INSTRUCTIONS: tuple[Instruction, ...] = (
    Instruction("lui", OPCODE, 0x37, "U", "imm"),
    Instruction("auipc", OPCODE, 0x17, "U", "auipc"),
    Instruction("jal", OPCODE, 0x6F, "J", "link", next_pc="jal"),
    _i("jalr", 0x67, 0, result="link", next_pc="jalr"),
    _branch("beq", 0, "eq"),
    _branch("bne", 1, "ne"),
    _branch("blt", 4, "lt"),
    _branch("bge", 5, "ge"),
    _branch("bltu", 6, "ltu"),
    _branch("bgeu", 7, "geu"),
    _i("lb", 0x03, 0, result="lb", load=1),
    _i("lh", 0x03, 1, result="lh", load=2),
    _i("lw", 0x03, 2, result="lw", load=4),
    _i("lbu", 0x03, 4, result="lbu", load=1),
    _i("lhu", 0x03, 5, result="lhu", load=2),
    _store("sb", 0, 1),
    _store("sh", 1, 2),
    _store("sw", 2, 4),
    _i("addi", 0x13, 0, result="add"),
    _i("slti", 0x13, 2, result="slt"),
    _i("sltiu", 0x13, 3, result="sltu"),
    _i("xori", 0x13, 4, result="xor"),
    _i("ori", 0x13, 6, result="or"),
    _i("andi", 0x13, 7, result="and"),
    _shift("slli", 1, 0x00, "sll"),
    _shift("srli", 5, 0x00, "srl"),
    _shift("srai", 5, 0x20, "sra"),
    _r("add", 0, 0x00, "add"),
    _r("sub", 0, 0x20, "sub"),
    _r("sll", 1, 0x00, "sll"),
    _r("slt", 2, 0x00, "slt"),
    _r("sltu", 3, 0x00, "sltu"),
    _r("xor", 4, 0x00, "xor"),
    _r("srl", 5, 0x00, "srl"),
    _r("sra", 5, 0x20, "sra"),
    _r("or", 6, 0x00, "or"),
    _r("and", 7, 0x00, "and"),
    _r("mul", 0, 0x01, "mul"),
    _r("mulh", 1, 0x01, "mulh"),
    _r("mulhsu", 2, 0x01, "mulhsu"),
    _r("mulhu", 3, 0x01, "mulhu"),
    _r("div", 4, 0x01, "div"),
    _r("divu", 5, 0x01, "divu"),
    _r("rem", 6, 0x01, "rem"),
    _r("remu", 7, 0x01, "remu"),
    Instruction("fence", OPCODE, 0x0F),  # FENCE and FENCE.I: no-ops on one in-order hart
    Instruction("ecall", 0xFFFF_FFFF, 0x0000_0073, exit="ecall"),
    Instruction("ebreak", 0xFFFF_FFFF, 0x0010_0073, exit="ebreak"),
)
