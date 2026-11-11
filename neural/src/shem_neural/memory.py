"""Guest RAM with dense, paged and flat backing.

`DeltaMemory` holds float64 bytes in delta-rule state:
`S <- S + beta (v - S k) k^T`. One-hot keys select a byte; beta 1 replaces it
and beta 0 leaves it unchanged.

`PagedMemory` uses shared 4 KiB pages with copy-on-write. Fresh memory maps the
zero page; whole pages written by the host are interned by content, allowing
restored snapshots to share pages.

`FlatMemory` holds a byte array per guest on the device, without page sharing
or host bookkeeping, for GPU compilation and graph capture.
"""

import hashlib

import torch
import torch.nn.functional as F

DTYPE = torch.float64
PAGE_SIZE = 4096
ZERO_PAGE = 0


class DeltaMemory:
    def __init__(self, batch: int, size: int):
        self.size = size
        self.state = torch.zeros(batch, size, dtype=DTYPE)

    def _key(self, addr: torch.Tensor) -> torch.Tensor:
        # Out-of-range addresses only occur on accesses the network gates off,
        # so any in-range row will do.
        return addr.clamp(0, self.size - 1).long()

    def read(self, addr: torch.Tensor, count: int) -> torch.Tensor:
        """The bytes at `addr .. addr + count`, as a (batch, count) tensor: one
        one-hot query per byte."""
        keys = self._key(addr[:, None] + torch.arange(count, device=addr.device))
        queries = F.one_hot(keys, self.size).to(DTYPE)
        return torch.einsum("bm,bkm->bk", self.state, queries)

    def write(self, addr: torch.Tensor, values: torch.Tensor, beta: torch.Tensor) -> None:
        """Write bytes at `addr ..` with the delta rule, one gate per byte:
        `values` and `beta` are (batch, count)."""
        keys = self._key(addr[:, None] + torch.arange(values.shape[1], device=addr.device))
        for k in range(values.shape[1]):
            onehot = F.one_hot(keys[:, k], self.size).to(DTYPE)
            old = (self.state * onehot).sum(-1)
            self.state = self.state + (beta[:, k] * (values[:, k] - old))[:, None] * onehot

    def get_bytes(self, row: int, addr: int, length: int) -> bytes:
        return bytes(self.state[row, addr : addr + length].to(torch.uint8).tolist())

    def set_bytes(self, row: int, addr: int, data: bytes) -> None:
        self.state[row, addr : addr + len(data)] = torch.tensor(list(data), dtype=DTYPE)

    def clone(self) -> "DeltaMemory":
        copy = DeltaMemory(self.state.shape[0], self.size)
        copy.state = self.state.clone()
        return copy

    def release(self) -> None:
        pass


class FlatMemory:
    """Each guest's memory as its own row of bytes on the device. A write with
    beta 0 lands on a scratch byte past the end, so the scatter never has two
    writers for a real byte, and nothing depends on data outside the kernels."""

    def __init__(self, batch: int, size: int, device: torch.device | str = "cpu"):
        self.size = size
        self.bytes = torch.zeros(batch, size + 1, dtype=torch.uint8, device=device)

    @classmethod
    def of(cls, data: torch.Tensor) -> "FlatMemory":
        """A view of existing (B, size + 1) bytes, which writes update in place."""
        memory = cls.__new__(cls)
        memory.size, memory.bytes = data.shape[1] - 1, data
        return memory

    def _keys(self, addr: torch.Tensor, count: int) -> torch.Tensor:
        return (addr[:, None] + torch.arange(count, device=addr.device)).clamp(0, self.size - 1).long()

    def read(self, addr: torch.Tensor, count: int) -> torch.Tensor:
        return self.bytes.gather(1, self._keys(addr, count)).to(DTYPE)

    def write(self, addr: torch.Tensor, values: torch.Tensor, beta: torch.Tensor) -> None:
        keys = torch.where(beta > 0, self._keys(addr, values.shape[1]), self.size)
        self.bytes.scatter_(1, keys, values.round().to(torch.uint8))

    def get_bytes(self, row: int, addr: int, length: int) -> bytes:
        return bytes(self.bytes[row, addr : addr + length].cpu().tolist())

    def set_bytes(self, row: int, addr: int, data: bytes) -> None:
        values = torch.tensor(list(data), dtype=torch.uint8, device=self.bytes.device)
        self.bytes[row, addr : addr + len(data)] = values

    def clone(self) -> "FlatMemory":
        return FlatMemory.of(self.bytes.clone())

    def release(self) -> None:
        pass


class PagePool:
    """Physical pages shared by guests, with reference counts.

    Page 0 is the zero page, which never changes and is never freed. A page
    with more than one holder is copied before it is written. Interned pages
    are found by content, so equal pages are stored once.
    """

    def __init__(self, capacity: int = 64):
        self.pages = torch.zeros(capacity, PAGE_SIZE, dtype=torch.uint8)
        self.refs = [0] * capacity
        self.refs[ZERO_PAGE] = 1 << 62
        self.free = list(range(capacity - 1, 0, -1))
        self.interned: dict[bytes, int] = {}
        self.digest_of: dict[int, bytes] = {}

    @property
    def in_use(self) -> int:
        """Pages holding data, the zero page included."""
        return len(self.refs) - len(self.free)

    def alloc(self) -> int:
        if not self.free:
            capacity = len(self.refs)
            self.pages = torch.cat([self.pages, torch.zeros_like(self.pages)])
            self.refs.extend([0] * capacity)
            self.free = list(range(2 * capacity - 1, capacity - 1, -1))
        page = self.free.pop()
        self.refs[page] = 1
        return page

    def incref(self, page: int) -> None:
        self.refs[page] += 1

    def decref(self, page: int) -> None:
        if page == ZERO_PAGE:
            return
        self.refs[page] -= 1
        if self.refs[page] == 0:
            self.free.append(page)
            if (digest := self.digest_of.pop(page, None)) is not None:
                del self.interned[digest]

    def intern(self, data: bytes) -> int:
        """A page holding `data`, shared with any page that already does."""
        if not any(data):
            return ZERO_PAGE
        digest = hashlib.sha256(data).digest()
        if (page := self.interned.get(digest)) is not None:
            self.incref(page)
            return page
        page = self.alloc()
        self.pages[page] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self.interned[digest] = page
        self.digest_of[page] = digest
        return page

    def private(self, page: int) -> int:
        """A page its holder may write: `page` itself if nobody else holds it,
        otherwise a copy, which takes over the holder's reference."""
        if page != ZERO_PAGE and self.refs[page] == 1:
            if (digest := self.digest_of.pop(page, None)) is not None:
                del self.interned[digest]  # about to change, so no longer findable
            return page
        copy = self.alloc()
        self.pages[copy] = self.pages[page]
        self.decref(page)
        return copy


class PagedMemory:
    def __init__(self, batch: int, size: int, pool: PagePool | None = None):
        if size % PAGE_SIZE:
            raise ValueError(f"memory size must be a multiple of {PAGE_SIZE} bytes")
        self.size = size
        self.pool = pool or PagePool()
        self.table = torch.full((batch, size // PAGE_SIZE), ZERO_PAGE, dtype=torch.long)

    def read(self, addr: torch.Tensor, count: int) -> torch.Tensor:
        """The bytes at `addr .. addr + count`, as a (batch, count) tensor."""
        offsets = (addr[:, None] + torch.arange(count)).clamp(0, self.size - 1).long()
        pages = self.table.gather(1, offsets // PAGE_SIZE)
        return self.pool.pages[pages, offsets % PAGE_SIZE].to(DTYPE)

    def write(self, addr: torch.Tensor, values: torch.Tensor, beta: torch.Tensor) -> None:
        """The delta rule's writes, for gates of 0 and 1: store where beta is 1.
        A store happens only for an instruction that commits, so its address is
        in range."""
        rows, cols = (beta > 0).nonzero(as_tuple=True)
        if rows.numel() == 0:
            return
        offsets = (addr[rows] + cols).long()
        guest_pages = offsets // PAGE_SIZE
        for row, guest_page in set(zip(rows.tolist(), guest_pages.tolist(), strict=True)):
            self.table[row, guest_page] = self.pool.private(int(self.table[row, guest_page]))
        pages = self.table[rows, guest_pages]
        old = self.pool.pages[pages, offsets % PAGE_SIZE].to(DTYPE)
        new = old + beta[rows, cols] * (values[rows, cols] - old)
        self.pool.pages[pages, offsets % PAGE_SIZE] = new.round().to(torch.uint8)

    def get_bytes(self, row: int, addr: int, length: int) -> bytes:
        out = bytearray()
        while length > 0:
            page, offset = divmod(addr, PAGE_SIZE)
            take = min(length, PAGE_SIZE - offset)
            physical = int(self.table[row, page])
            out += bytes(self.pool.pages[physical, offset : offset + take].tolist())
            addr, length = addr + take, length - take
        return bytes(out)

    def set_bytes(self, row: int, addr: int, data: bytes) -> None:
        """Host writes: whole pages are interned, partial ones written in place."""
        at = 0
        while at < len(data):
            page, offset = divmod(addr + at, PAGE_SIZE)
            take = min(len(data) - at, PAGE_SIZE - offset)
            chunk = data[at : at + take]
            old = int(self.table[row, page])
            if take == PAGE_SIZE:
                self.table[row, page] = self.pool.intern(chunk)
                self.pool.decref(old)
            else:
                physical = self.pool.private(old)
                self.table[row, page] = physical
                written = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
                self.pool.pages[physical, offset : offset + take] = written
            at += take

    def clone(self) -> "PagedMemory":
        """A copy that shares every page until one side writes it."""
        copy = PagedMemory.__new__(PagedMemory)
        copy.size, copy.pool = self.size, self.pool
        copy.table = self.table.clone()
        for page in self.table.flatten().tolist():
            self.pool.incref(page)
        return copy

    def release(self) -> None:
        """Give back every page, as when the guest goes away."""
        for page in self.table.flatten().tolist():
            self.pool.decref(page)
        self.table.fill_(ZERO_PAGE)

    @staticmethod
    def stack(memories: list["PagedMemory"]) -> "PagedMemory":
        """One memory for a batch, taking over the guests' page tables."""
        batch = PagedMemory.__new__(PagedMemory)
        first = memories[0]
        batch.size, batch.pool = first.size, first.pool
        batch.table = torch.cat([m.table for m in memories])
        return batch
