import torch

from shem_neural.memory import PAGE_SIZE, ZERO_PAGE, DeltaMemory, FlatMemory, PagedMemory, PagePool

SIZE = 16 * PAGE_SIZE


def test_delta_paged_and_flat_memory_agree_on_random_traffic():
    generator = torch.Generator().manual_seed(3)
    memories = DeltaMemory(2, SIZE), PagedMemory(2, SIZE), FlatMemory(2, SIZE)
    for _ in range(200):
        addr = torch.randint(0, SIZE - 4, (2,), generator=generator).to(torch.float64)
        values = torch.randint(0, 256, (2, 4), generator=generator).to(torch.float64)
        beta = torch.randint(0, 2, (2, 4), generator=generator).to(torch.float64)
        for memory in memories:
            memory.write(addr, values, beta)
        probe = torch.randint(0, SIZE - 4, (2,), generator=generator).to(torch.float64)
        reads = [memory.read(probe, 4) for memory in memories]
        assert all(torch.equal(reads[0], read) for read in reads)
    for row in range(2):
        contents = {memory.get_bytes(row, 0, SIZE) for memory in memories}
        assert len(contents) == 1


def test_fresh_memory_costs_nothing_until_written():
    pool = PagePool()
    memory = PagedMemory(1, SIZE, pool)
    assert pool.in_use == 1, "only the zero page"
    memory.write(torch.tensor([5.0 * PAGE_SIZE]), torch.tensor([[7.0]]), torch.tensor([[1.0]]))
    assert pool.in_use == 2
    assert memory.get_bytes(0, 5 * PAGE_SIZE, 1) == b"\x07"


def test_clones_share_pages_until_written():
    pool = PagePool()
    parent = PagedMemory(1, SIZE, pool)
    parent.set_bytes(0, 100, b"shared")
    child = parent.clone()
    assert pool.in_use == 2, "the written page is shared, not copied"
    child.set_bytes(0, 100, b"CHILD!")
    assert pool.in_use == 3, "the child's write copied the page"
    assert parent.get_bytes(0, 100, 6) == b"shared"
    assert child.get_bytes(0, 100, 6) == b"CHILD!"
    child.release()
    parent.release()
    assert pool.in_use == 1, "every page went back to the pool"


def test_restored_pages_are_interned_by_content():
    pool = PagePool()
    image = bytes(range(256)) * (SIZE // 256)
    guests = [PagedMemory(1, SIZE, pool) for _ in range(50)]
    for guest in guests:
        guest.set_bytes(0, 0, image)
    # 16 pages of the same 4 KiB pattern: one physical page for all 800.
    assert pool.in_use == 2
    assert all(g.get_bytes(0, 0, SIZE) == image for g in guests)
    guests[0].write(torch.tensor([0.0]), torch.tensor([[9.0]]), torch.tensor([[1.0]]))
    assert guests[1].get_bytes(0, 0, 1) == b"\x00", "writing one guest leaves the others alone"
    assert guests[0].get_bytes(0, 0, 1) == b"\x09"


def test_pages_written_in_place_stop_being_found_by_content():
    pool = PagePool()
    first = PagedMemory(1, SIZE, pool)
    page = b"\x01" * PAGE_SIZE
    first.set_bytes(0, 0, page)
    first.set_bytes(0, 0, b"\x02")  # sole holder: changed in place
    second = PagedMemory(1, SIZE, pool)
    second.set_bytes(0, 0, page)
    assert second.get_bytes(0, 0, 2) == b"\x01\x01"
    assert first.get_bytes(0, 0, 2) == b"\x02\x01"
    assert int(first.table[0, 0]) != int(second.table[0, 0]) != ZERO_PAGE


def test_the_pool_grows_as_pages_are_needed():
    pool = PagePool(capacity=4)
    memory = PagedMemory(1, SIZE, pool)
    for page in range(16):
        memory.set_bytes(0, page * PAGE_SIZE, bytes([page + 1]))
    assert pool.in_use == 17
    assert all(memory.get_bytes(0, p * PAGE_SIZE, 1) == bytes([p + 1]) for p in range(16))
