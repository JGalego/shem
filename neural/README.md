# shem-neural

The neural vCPU for [shem](https://github.com/JGalego/shem): an RV32IM CPU compiled into the weights of a network that runs one instruction per forward pass.

![An RV32 program runs one instruction per forward pass. On ecall, the host handles the write system call, updates the return value and program counter, then resumes the guest.](https://raw.githubusercontent.com/JGalego/shem/main/docs/assets/execution.svg)

From this directory, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv run pytest
uv run python -m shem_neural.server --socket /tmp/shem.sock
```

Then, in another terminal:

```sh
shem run --vcpu neural --neural-socket /tmp/shem.sock /path/to/rv32-program
```

See the [neural vCPU guide](https://github.com/JGalego/shem/blob/main/docs/neural.md) for the network, protocol and benchmarks.
