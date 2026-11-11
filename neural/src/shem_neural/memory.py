"""Guest RAM as a delta-rule state matrix.

Each byte of guest memory is one row of the state. Writing value `v` with a
one-hot key `k` and gate `beta` is the delta rule, `S <- S + beta (v - S k) k^T`,
which replaces the keyed row exactly when beta is 1 and leaves it alone when
beta is 0. Reading with a one-hot query `q` is `S q`, which returns the row.
This is how Gated DeltaNet layers hold memory in Qwenputer.

There are two executors for the same arithmetic. The dense one builds the
one-hot vectors and does the matrix products, as a model would. The fused one
uses gather and scatter, which compute exactly the same thing in O(1) instead
of O(memory) per access; tests check that they agree.
"""

import torch
import torch.nn.functional as F

DTYPE = torch.float64


class DeltaMemory:
    def __init__(self, batch: int, size: int, dense: bool = False):
        self.size = size
        self.dense = dense
        self.state = torch.zeros(batch, size, dtype=DTYPE)

    def _key(self, addr: torch.Tensor) -> torch.Tensor:
        # Out-of-range addresses only occur on accesses the network gates off,
        # so any in-range row will do.
        return addr.clamp(0, self.size - 1).long()

    def read(self, addr: torch.Tensor, count: int) -> torch.Tensor:
        """The bytes at `addr .. addr + count`, as a (batch, count) tensor: one
        one-hot query per byte."""
        keys = self._key(addr[:, None] + torch.arange(count))
        if self.dense:
            queries = F.one_hot(keys, self.size).to(DTYPE)
            return torch.einsum("bm,bkm->bk", self.state, queries)
        return self.state.gather(1, keys)

    def write(self, addr: torch.Tensor, values: torch.Tensor, beta: torch.Tensor) -> None:
        """Write bytes at `addr ..` with the delta rule, one gate per byte:
        `values` and `beta` are (batch, count)."""
        keys = self._key(addr[:, None] + torch.arange(values.shape[1]))
        if self.dense:
            for k in range(values.shape[1]):
                onehot = F.one_hot(keys[:, k], self.size).to(DTYPE)
                old = (self.state * onehot).sum(-1)
                self.state = self.state + (beta[:, k] * (values[:, k] - old))[:, None] * onehot
        else:
            # Accumulate the deltas, so that gated-off bytes whose keys were
            # clamped onto the same row as a real write add nothing.
            old = self.state.gather(1, keys)
            self.state.scatter_add_(1, keys, beta * (values - old))

    def clone(self) -> "DeltaMemory":
        copy = DeltaMemory(self.state.shape[0], self.size, self.dense)
        copy.state = self.state.clone()
        return copy
