import pytest
import torch
from rv32 import (
    A0,
    A1,
    A2,
    A7,
    EBREAK,
    ECALL,
    S0,
    SP,
    T0,
    T1,
    ZERO,
    add,
    addi,
    assemble,
    bne,
    jal,
    lb,
    lbu,
    li,
    lw,
    m,
    sb,
    sw,
)  # fmt: skip

from shem_neural.compile import compile_weights, isa
from shem_neural.cpu import EBREAK as EBREAK_EXIT
from shem_neural.cpu import MEMORY_FAULT as MEMORY_FAULT_EXIT
from shem_neural.cpu import NeuralCpu, State
from shem_neural.vcpu import (
    BUDGET,
    EXIT_EBREAK,
    EXIT_ECALL,
    EXIT_ILLEGAL,
    EXIT_MEMORY_FAULT,
    NeuralVcpu,
)  # fmt: skip

MEM = 64 << 10
WEIGHTS = compile_weights()
CPU = NeuralCpu(WEIGHTS)
DENSE_CPU = NeuralCpu(WEIGHTS, fused=False)
STATIC_CPU = NeuralCpu(WEIGHTS, static=True)


def load(code: bytes, dense: bool = False) -> NeuralVcpu:
    vcpu = NeuralVcpu(MEM, DENSE_CPU if dense else CPU, dense)
    vcpu.write_mem(0, code)
    return vcpu


def run(*code, dense: bool = False) -> NeuralVcpu:
    """Run `code` followed by an EBREAK, which the tests use as halt."""
    vcpu = load(assemble(*code, EBREAK), dense)
    reason, _, _ = vcpu.run(10_000)
    assert reason == EXIT_EBREAK
    return vcpu


def test_every_instruction_pattern_matches_only_itself():
    for ins in isa.INSTRUCTIONS:
        matches = [o.name for o in isa.INSTRUCTIONS if ins.match & o.mask == o.match]
        assert matches == [ins.name]


def test_arithmetic_and_x0():
    vcpu = run(li(A0, 40), addi(A1, ZERO, 2), add(A2, A0, A1), addi(ZERO, ZERO, 5))
    assert vcpu.reg(A2) == 42
    assert vcpu.reg(ZERO) == 0


def test_loop_counts_instructions():
    vcpu = load(
        assemble(addi(A0, ZERO, 0), addi(T0, ZERO, 10), add(A0, A0, T0), addi(T0, T0, -1), bne(T0, ZERO, -8), EBREAK)
    )
    assert vcpu.run(10_000) == (EXIT_EBREAK, 0, 32)
    assert vcpu.reg(A0) == 55
    assert int(vcpu.state.instret[0]) == 32


def test_loads_sign_and_zero_extend():
    vcpu = run(
        li(T0, 0x1000), addi(T1, ZERO, -1), sb(T1, T0, 0), lb(A0, T0, 0), lbu(A1, T0, 0), sw(T1, T0, -4), lw(A2, T0, -4)
    )
    assert (vcpu.reg(A0), vcpu.reg(A1), vcpu.reg(A2)) == (0xFFFF_FFFF, 0xFF, 0xFFFF_FFFF)


MIN = 0x8000_0000
M_CASES = [
    ("mul", 7, -3, -21),
    ("mulh", -1, -1, 0),
    ("mulh", MIN, MIN, 0x4000_0000),
    ("mulhu", 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFE),
    ("mulhsu", -1, 0xFFFF_FFFF, 0xFFFF_FFFF),
    ("div", -7, 2, -3),
    ("rem", -7, 2, -1),
    ("divu", 7, 2, 3),
    ("remu", 7, 2, 1),
    ("div", 5, 0, -1),
    ("divu", 5, 0, 0xFFFF_FFFF),
    ("rem", 5, 0, 5),
    ("remu", 5, 0, 5),
    ("div", MIN, -1, MIN),
    ("rem", MIN, -1, 0),
]


@pytest.mark.parametrize(("op", "x", "y", "expected"), M_CASES)
def test_m_extension_including_edge_cases(op, x, y, expected):
    vcpu = run(li(A0, x), li(A1, y), m(op, A2, A0, A1))
    assert vcpu.reg(A2) == expected & 0xFFFF_FFFF


def test_jumps_link_and_land():
    vcpu = load(assemble(jal(1, 8), EBREAK, addi(A0, ZERO, 42), 0x0000_8067))  # ret
    assert vcpu.run(100)[0] == EXIT_EBREAK
    assert (vcpu.pc, vcpu.reg(A0), vcpu.reg(1)) == (4, 42, 4)


def test_exits_report_the_reference_details():
    vcpu = load(assemble(addi(A7, ZERO, 93), ECALL))
    assert vcpu.run(100) == (EXIT_ECALL, 0, 1)
    assert vcpu.pc == 4
    assert load(assemble(0xC000_2573)).run(10) == (EXIT_ILLEGAL, 0xC000_2573, 0)
    vcpu = load(assemble(li(T0, MEM), addi(A0, ZERO, 1), lw(A0, T0, 0)))
    assert vcpu.run(10) == (EXIT_MEMORY_FAULT, MEM, 3)
    assert vcpu.reg(A0) == 1, "a faulting load changes nothing"
    vcpu = load(assemble(EBREAK))
    vcpu.pc = 2
    assert vcpu.run(10) == (EXIT_MEMORY_FAULT, 2, 0), "misaligned fetch"


def test_budget_exits_are_resumable():
    vcpu = load(
        assemble(addi(A0, ZERO, 0), addi(T0, ZERO, 10), add(A0, A0, T0), addi(T0, T0, -1), bne(T0, ZERO, -8), EBREAK)
    )
    assert vcpu.run(5) == (BUDGET, 0, 5)
    assert vcpu.run(1000)[0] == EXIT_EBREAK
    assert vcpu.reg(A0) == 55


def test_snapshot_restores_onto_a_fresh_vcpu():
    vcpu = load(assemble(addi(T0, ZERO, 10), addi(T0, T0, -1), bne(T0, ZERO, -4), EBREAK))
    vcpu.run(7)
    snapshot = vcpu.snapshot()
    other = NeuralVcpu(MEM, CPU)
    other.restore(snapshot)
    assert other.snapshot() == snapshot
    assert vcpu.run(1000) == other.run(1000)
    assert vcpu.snapshot() == other.snapshot()


PROGRAMS = [
    (li(T0, 0x2000), addi(T1, ZERO, -5), sw(T1, T0, 0), lb(A0, T0, 1), lw(A1, T0, 0), sb(A0, SP, 0)),
    *[(li(A0, x), li(A1, y), m(op, A2, A0, A1)) for op, x, y, _ in M_CASES],
]


@pytest.mark.parametrize("program", PROGRAMS)
def test_dense_network_and_fused_executor_give_identical_runs(program):
    fused, dense = run(*program), run(*program, dense=True)
    assert fused.snapshot() == dense.snapshot()


def test_fused_units_match_the_network_on_random_values():
    from shem_neural.cpu import decompose

    generator = torch.Generator().manual_seed(7)
    values = torch.randint(0, 2**35 - 1, (4096,), generator=generator).to(torch.float64)
    assert torch.equal(CPU.decompose(values, 35), decompose(values, 35))
    data = torch.randint(0, 256, (4096,), generator=generator).to(torch.float64)
    assert torch.equal(CPU.byte_bits(data), DENSE_CPU.byte_bits(data))
    fields = CPU.decompose(torch.randint(0, 32, (4096,), generator=generator).to(torch.float64), 5)
    assert torch.equal(CPU.index(fields), DENSE_CPU.index(fields))
    a, o = (torch.randint(0, 2**32, (512,), generator=generator).to(torch.float64) for _ in "ao")
    o[:64] = 0  # division by zero
    o[64:128] = 2**32 - 1  # by -1
    a[64:96] = 2**31  # INT_MIN / -1
    a_bits, o_bits = CPU.decompose(a, 32), CPU.decompose(o, 32)
    fused = torch.stack(CPU.divide(a_bits, o_bits, a, o))
    dense = torch.stack(DENSE_CPU.divide(a_bits, o_bits, a, o))
    assert torch.equal(fused, dense)


BATCH = [
    assemble(li(A0, 6), li(A1, 7), m("mul", A2, A0, A1), EBREAK),
    assemble(li(A0, 100), li(A1, 7), m("divu", A2, A0, A1), sw(A2, ZERO, 0x800), EBREAK),
    assemble(addi(S0, ZERO, 3), addi(S0, S0, -1), bne(S0, ZERO, -4), EBREAK),
    assemble(li(T0, 0x2000), addi(T1, ZERO, -5), sw(T1, T0, 0), lb(A0, T0, 1), lbu(A1, T0, 3), EBREAK),
]


def run_batch(cpu: NeuralCpu, state: State, steps: int) -> None:
    """Step every guest `steps` times, holding each one once it has exited,
    as the server does."""
    halted = torch.zeros(state.pc.shape[0], dtype=torch.bool, device=state.pc.device)
    for _ in range(steps):
        exits, _ = cpu.step(state, active=~halted)
        halted |= exits.sum(-1) > 0


def test_flat_memory_and_static_executor_match_the_paged_one():
    paged, flat = State.zeros(len(BATCH), MEM), State.zeros(len(BATCH), MEM, flat=True)
    for k, code in enumerate(BATCH):
        paged.mem.set_bytes(k, 0, code)
        flat.mem.set_bytes(k, 0, code)
    run_batch(CPU, paged, 50)
    run_batch(STATIC_CPU, flat, 50)
    for name in ("pc", "pc_low", "regs", "instret"):
        assert torch.equal(getattr(paged, name), getattr(flat, name)), name
    for k in range(len(BATCH)):
        assert paged.mem.get_bytes(k, 0, MEM) == flat.mem.get_bytes(k, 0, MEM)


def test_gpu_batch_steps_in_place_and_keeps_each_first_exit():
    from shem_neural.gpu import Batch

    paged, batch = State.zeros(len(BATCH), MEM), Batch(WEIGHTS, len(BATCH), MEM)
    for k, code in enumerate(BATCH):
        paged.mem.set_bytes(k, 0, code)
        batch.state.mem.set_bytes(k, 0, code)
    run_batch(CPU, paged, 50)
    batch.run(50)
    assert torch.equal(paged.regs, batch.state.regs)
    assert torch.equal(paged.instret, batch.state.instret)
    assert batch.halted.all()
    assert batch.exits.argmax(-1).tolist() == [EBREAK_EXIT, MEMORY_FAULT_EXIT, EBREAK_EXIT, EBREAK_EXIT]
    assert batch.info[1] == 2**32 - 2048  # the store to -2048


def test_a_batch_runs_different_programs_like_separate_vcpus():
    programs = [
        assemble(li(A0, 6), li(A1, 7), m("mul", A2, A0, A1), EBREAK),
        assemble(li(A0, 100), li(A1, 7), m("divu", A2, A0, A1), sw(A2, ZERO, 0x800), EBREAK),
        assemble(addi(S0, ZERO, 3), addi(S0, S0, -1), bne(S0, ZERO, -4), EBREAK),
    ]
    alone = []
    for code in programs:
        vcpu = load(code)
        vcpu.run(100)
        alone.append(vcpu)

    batch = State.zeros(len(programs), MEM)
    for k, code in enumerate(programs):
        batch.mem.set_bytes(k, 0, code)
    halted = torch.zeros(len(programs), dtype=torch.bool)
    for _ in range(100):
        exits, _ = CPU.step(batch)
        halted |= exits.sum(-1) > 0
        if halted.all():
            break
    for k, vcpu in enumerate(alone):
        assert torch.equal(batch.regs[k], vcpu.state.regs[0])
        assert batch.mem.get_bytes(k, 0, MEM) == vcpu.read_mem(0, MEM)
        assert batch.pc[k] == vcpu.state.pc[0]


@pytest.mark.parametrize("dense", [False, True])
def test_memory_can_start_at_a_base_address(dense):
    from shem_neural.vcpu import MemoryFault

    base = 0x8000_0000
    vcpu = NeuralVcpu(MEM, DENSE_CPU if dense else CPU, dense, base=base)
    data = base + 0x2000
    code = assemble(li(T0, data), addi(T1, ZERO, 42), sw(T1, T0, 0), lw(A0, T0, 0), lw(A1, ZERO, 0), EBREAK)
    vcpu.write_mem(base, code)
    vcpu.pc = base
    assert vcpu.run(100) == (EXIT_MEMORY_FAULT, 0, 5)  # address 0 is below memory
    assert vcpu.reg(A0) == 42
    assert vcpu.read_mem(data, 4) == (42).to_bytes(4, "little")
    with pytest.raises(MemoryFault):
        vcpu.read_mem(0, 4)
    with pytest.raises(MemoryFault):
        vcpu.read_mem(base + MEM - 2, 4)
    vcpu.pc = base + MEM  # just past the end
    assert vcpu.run(10) == (EXIT_MEMORY_FAULT, base + MEM, 0)
    other = NeuralVcpu(MEM, CPU, base=base)
    other.restore(vcpu.snapshot())
    assert other.snapshot() == vcpu.snapshot()
