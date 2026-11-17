# Roadmap

Planned work, in no fixed order. See [architecture](docs/architecture.md) for the design; open an issue to discuss changes.

- **One network.** Compiled CPU layers that read and write a language model's residual stream, so that a model's own forward pass can execute a guest ([what is in the weights](docs/architecture.md#what-is-in-the-weights-and-what-is-not)). Today the CPU and the models are separate networks.
- **Neural execution.** Narrower types for bits, fewer tensor operations per step, and GPU execution in the vCPU server.
- **Builds and filesystems.** A layer-digest build cache, overlayfs instead of copied filesystems, and coherence between processes sharing writable file mappings.
- **Guest coverage.** `docker exec` into program containers, `/proc/PID/fd`, datagram sockets and a network beyond the container, and VM block and network devices.
- **ISA validation.** Cross-check the reference interpreter against Spike.

Released features are listed in the [changelog](CHANGELOG.md).
