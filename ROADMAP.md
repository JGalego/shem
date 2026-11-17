# Roadmap

Planned work, in no fixed order. See [architecture](docs/architecture.md) for the design; open an issue to discuss changes.

- **One network.** Compiled CPU layers that read and write a language model's residual stream, so that a model's own forward pass can execute a guest ([what is in the weights](docs/architecture.md#what-is-in-the-weights-and-what-is-not)). Today the CPU and the models are separate networks.
- **Platforms.** A native Windows build, which needs the Unix-specific parts (descriptors, sockets, signals, terminals) behind a layer; today shem runs on Linux and macOS, and on Windows in WSL 2.
- **Neural execution.** Narrower types for bits, fewer tensor operations per step, and GPU execution in the vCPU server.
- **Builds and filesystems.** Overlayfs instead of copied filesystems, and coherence between processes sharing writable file mappings.
- **Guest coverage.** `/proc/PID/fd`, datagram sockets and a network beyond the container, and VM block and network devices.
- **Guest reach.** Rust and other languages' targets for RV32 Linux (the C extension, which Rust's RV32 Linux targets need, runs on both vCPUs); more interpreters and databases as examples, as Lua is.
- **Multimodal models.** Vision encoders as a block type, so that the images of a chat request reach the model.
- **ISA validation.** Cross-check the reference interpreter against Spike too: random programs already run against QEMU's user-mode emulator in CI.

Released features are listed in the [changelog](CHANGELOG.md).
