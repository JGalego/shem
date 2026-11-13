# Roadmap

Planned work, in no fixed order. See [architecture](docs/architecture.md) for the design; open an issue to discuss changes.

- **Neural execution.** Narrower types for bits, fewer tensor operations per step, and GPU execution in the vCPU server.
- **Builds and filesystems.** A layer-digest build cache, overlayfs instead of copied filesystems, and coherence between processes sharing writable file mappings.
- **Guest coverage.** Terminals under `docker run -t` for program containers, `/proc/PID/fd`, sockets, and VM block and network devices.
- **ISA validation.** Cross-check the reference interpreter against Spike.

Released features are listed in the [changelog](CHANGELOG.md).
