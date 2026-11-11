# shem-neural

The neural vCPU for [shem](../README.md): an RV32IM CPU compiled into the weights of a network that runs one instruction per forward pass.

```sh
uv run pytest                                            # unit tests
uv run python -m shem_neural.server --socket /tmp/shem.sock   # serve vCPUs to the host
```

See [docs/neural.md](../docs/neural.md) for how the network works and how the host talks to it.
