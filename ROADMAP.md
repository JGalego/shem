# Roadmap

shem is experimental, and this list is what the maintainers plan to work on next, in no fixed order. Open an issue to propose or discuss an item. The design behind it is in [docs/architecture.md](docs/architecture.md).

- **Neural execution.** Narrower types for bits, fewer tensor operations per step, and GPU execution in the vCPU server.
- **Builds and filesystems.** A layer-digest build cache, overlayfs instead of copied filesystems, and coherence between processes sharing writable file mappings.
- **Guest coverage.** Terminals under `docker run -t` for program containers, `/proc/PID/fd`, sockets, and VM block and network devices.
- **ISA validation.** Cross-check the reference interpreter against Spike.

Released features are listed in the [changelog](CHANGELOG.md).
