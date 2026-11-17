"""The C extension on the neural vCPU: the table of expansions it fetches from, and guests that run compressed code."""

from pathlib import Path

import torch

from shem_neural import rvc
from shem_neural.compile import compile_weights
from shem_neural.cpu import NeuralCpu
from shem_neural.vcpu import EXIT_EBREAK, EXIT_ILLEGAL, EXIT_MEMORY_FAULT, NeuralVcpu

TABLE = Path(__file__).parents[2] / "crates/shem/tests/data/compressed.txt"
MEM = 1 << 16


def test_every_form_the_assembler_encodes_expands_to_the_instruction_it_writes_in_full():
    forms = [line.split()[:2] for line in TABLE.read_text().splitlines() if not line.startswith("#")]
    assert len(forms) > 3000
    for half, word in forms:
        assert rvc.expand(int(half, 16)) == int(word, 16), (half, word)


def test_the_whole_table_is_the_one_the_reference_vcpu_expands_by():
    # `the_whole_table_of_expansions_is_the_one_the_neural_vcpu_looks_up`, in Rust, holds the same digest
    assert rvc.digest() == "c0b1867d9cc6c495c3799b9e41bffc9a55d1a0654d6b28b823cd9b875f214270"


def test_the_table_is_bytes_of_bits_with_zeros_for_what_is_illegal():
    table = rvc.table()
    assert table.shape == (1 << 16, 32) and table.dtype == torch.uint8
    assert int(table[0x0000].sum()) == 0, "the all-zero halfword is illegal"
    assert [int(bit) for bit in table[0x4515]] == [0x00500513 >> i & 1 for i in range(32)]  # c.li a0, 5


def vcpu(code: list[int], compressed: bool = True) -> NeuralVcpu:
    cpu = NeuralCpu(compile_weights(), fused=True)
    machine = NeuralVcpu(MEM, cpu)
    machine.compressed = compressed
    machine.write_mem(0, b"".join(half.to_bytes(2, "little") for half in code))
    return machine


def test_a_guest_runs_compressed_and_ordinary_instructions_at_any_halfword():
    addi_a1 = 0x00150593  # addi a1, a0, 1, on a halfword boundary that is not a word's
    machine = vcpu([0x4515, 0x050D, addi_a1 & 0xFFFF, addi_a1 >> 16, 0x0001, 0x9002])
    reason, _, steps = machine.run(100)
    assert (reason, steps) == (EXIT_EBREAK, 4)
    assert (machine.reg(10), machine.reg(11), machine.pc) == (8, 9, 10)


def test_links_are_by_the_length_of_the_instruction():
    # c.jal +4 (ra = 2), a c.ebreak that is jumped over, c.mv a1, ra; c.jalr a1 (to 2, ra = 8)
    machine = vcpu([0x2011, 0x9002, 0x8586, 0x9582])
    assert machine.run(100)[0] == EXIT_EBREAK
    assert (machine.reg(11), machine.reg(1), machine.pc) == (2, 8, 2)


def test_an_illegal_halfword_says_its_16_bits_and_an_odd_pc_is_misaligned():
    machine = vcpu([0x0001, 0x0000])
    assert machine.run(10) == (EXIT_ILLEGAL, 0, 1)
    machine = vcpu([0x0001, 0x9002])
    machine.pc = 1
    assert machine.run(10)[0] == EXIT_MEMORY_FAULT
    # An instruction cut by the end of memory is a fault, and a halfword that fits is not
    machine = vcpu([])
    machine.write_mem(MEM - 2, (0x9002).to_bytes(2, "little"))
    machine.pc = MEM - 2
    assert machine.run(10)[0] == EXIT_EBREAK
    machine.write_mem(MEM - 2, (0x0003).to_bytes(2, "little"))  # the start of a 32-bit one
    assert machine.run(10)[0] == EXIT_MEMORY_FAULT


def test_a_guest_without_the_extension_runs_as_before():
    machine = vcpu([0x0001, 0x0000], compressed=False)
    assert machine.run(10) == (EXIT_ILLEGAL, 1, 0), "the word, which is all a plain guest can say"
    machine = vcpu([0x4515], compressed=False)
    machine.pc = 2
    assert machine.run(10)[0] == EXIT_MEMORY_FAULT, "a halfword boundary is a fault without it"


def test_snapshots_carry_the_extension_and_old_ones_are_as_they_were():
    machine = vcpu([0x4515, 0x9002])
    machine.run(10)
    snapshot = machine.snapshot()
    assert int.from_bytes(snapshot[4:8], "little") == 2
    other = vcpu([], compressed=False)
    other.restore(snapshot)
    assert other.compressed and other.reg(10) == 5
    plain = vcpu([0x0073], compressed=False).snapshot()
    assert int.from_bytes(plain[4:8], "little") == 1, "without the extension a snapshot is version 1, as it was"
    other.restore(plain)
    assert not other.compressed
    machine.reset()
    assert not machine.compressed
