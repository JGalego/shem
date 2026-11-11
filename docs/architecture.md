# Architecture

shem treats the host as a hypervisor and a model as the machine it runs. The model supplies a CPU, compiled into its weights, and RAM, which is its recurrent state. The host supplies everything else: system calls, devices, scheduling, paging and storage.

The starting point is [Qwenputer](https://divagr.com/blog/qwenputer), which compiles an RV32IM CPU into Qwen3.5-4B and uses the model's Gated DeltaNet state as addressable memory. shem's neural vCPU is a standalone compiled network, not part of a pretrained language model. Language-model images can call programs on it as tools ([models.md](models.md#compiled-layers-and-tools)). The host uses the same [vCPU interface](#the-vcpu-interface) for the neural executor and the reference interpreter.

![OCI registries store the images. Docker reaches the host through the containerd shim; the shem CLI drives it directly. The host provides container syscalls or VM devices and drives either the neural or reference vCPU.](assets/architecture.svg)

## How virtualization maps onto a model

| Virtualization | shem |
|---|---|
| vCPU | Compiled CPU layers |
| Guest RAM | DeltaNet state. A one-hot key writes one row and a one-hot query reads it back. |
| `KVM_RUN` loop | `Vcpu::run` |
| VM exit | The vCPU stops on `ecall`, `ebreak`, an illegal instruction or a memory fault |
| DMA, virtio rings | The host reads and writes state rows directly, without decoding any tokens |
| Timer interrupt | The host stops decoding after a step budget |
| Snapshot | Save canonical machine state and the host-side process or device state |
| Multi-tenancy | The batch dimension |

## Two modes, one runtime

| | Container (`shem run`) | VM (`shem boot`) |
|---|---|---|
| Guest runs | Programs, with the host as their kernel | A kernel and its programs |
| Guest exits on | System calls | Device access, traps, timer ticks |
| Host provides | Linux system calls ([`linux/`](../crates/shem/src/linux/)) | The machine: CSRs, traps, a timer, a UART and a syscon ([vm.md](vm.md)) |
| ISA needed | RV32IM; the host emulates the A extension | RV32IM too; the host emulates A, the CSRs and traps |
| Start | Load an ELF into memory, or thaw a warm image | Boot a kernel, or thaw a booted VM |

## The vCPU interface

[`vcpu.rs`](../crates/shem/src/vcpu.rs) defines the `Vcpu` trait that both the reference interpreter and a neural vCPU implement:

- `run(budget)` executes at most `budget` instructions and returns an `Exit`.
- `reg`, `set_reg`, `pc`, `set_pc`, `read_mem`, `write_mem` give the host direct access to the machine, the way a hypervisor reads guest registers and memory.
- `snapshot` and `restore` use `MachineState`, a canonical, vCPU-independent format holding the pc, registers, instruction count and memory.

Exit semantics: except for `Budget`, the exit's `pc` points at the instruction that caused it, and that instruction has not retired. For a system call the host writes the result to `a0` and resumes at `pc + 4`.

Two requirements follow for a neural vCPU:

1. **Trace equivalence.** It must produce the same registers, memory and exits as [`ReferenceVcpu`](../crates/shem/src/reference.rs), step for step. That is how it gets tested: run the same program on both and compare.
2. **Restorable from canonical state.** At any instruction boundary, its whole internal state must be rebuildable from a `MachineState`. A snapshot taken on one vCPU can then resume on another. In particular, slow work such as booting a kernel can run on the fast reference vCPU, and the result can be loaded onto the model.

## Design rules that come from running on a model

1. **The host handles unsupported operations.** The vCPU exits for system calls, atomics and, in VM mode, privileged instructions and device accesses. [`emulate.rs`](../crates/shem/src/emulate.rs) handles the A extension because musl's riscv32 port needs LR/SC.
2. **Batching supports unrelated guests.** The network routes each guest's decoded instruction through tensor operations rather than a separate host instruction handler. Guests can run different programs in one batch. Throughput and per-guest latency still depend on batch size, workload and device ([benchmarks](neural.md#speed)).
3. **Instruction budgets provide preemption.** The host drives execution and stops the vCPU after a bounded number of retired instructions.
4. **`fork()` clones vCPU state.** The reference interpreter copies memory; the neural server shares pages until a write. Threads share their process's vCPU memory and save their registers between turns ([`container.rs`](../crates/shem/src/container.rs)).
5. **The host manages memory backing.** The fused CPU executor uses copy-on-write pages; the dense executor uses delta-rule state, and the GPU harness uses flat byte arrays. This is not a guest MMU.
6. **Build-time execution is reproducible with fixed inputs.** The clock is virtual, `getrandom` uses a seeded PRNG, and directory listings are sorted. Builds use empty stdin and normalize layer metadata ([shemfile.md](shemfile.md#reproducibility)). Replaying an interactive run would also require its external inputs, not just its exits.

## Isolation

- The only way out of a guest is an exit, so the attack surface is the exit handler. The [system-call layer](../crates/shem/src/linux/) treats every guest pointer and length as hostile: transfers are capped, and bad pointers return `EFAULT`.
- Guest paths resolve inside the root directory with chroot semantics ([`rootfs.rs`](../crates/shem/src/linux/rootfs.rs)): `..` stops at the root, and symlinks, absolute ones included, resolve relative to it. Resolution and the final open are separate host calls, so a concurrent writer to the host directory could race them; a single guest cannot. `openat2` with `RESOLVE_IN_ROOT` would close that gap on Linux hosts.
- Tool calls run on a separate vCPU; the language model does not share its residual stream or state with the compiled CPU.
- Tool output is untrusted model context. Executing a tool on a vCPU does not prevent prompt injection through its output.

## Tech stack

| Part | Implementation |
|---|---|
| Host runtime, CLI, reference vCPU and image store | Rust, in [`crates/shem`](../crates/shem/) |
| Neural vCPU and language-model runtime | Python and PyTorch, in [`neural/`](../neural/). The vCPU server runs on the CPU; the GPU vCPU harness is separate. Model sessions support CPU and GPU execution. |
| Host to neural server | [Framed requests over a Unix socket](neural.md#the-protocol) |
| Images | OCI layouts and registries; ORAS for push and pull |
| Docker, containerd and Kubernetes | [containerd shim](containerd.md) and [RuntimeClass](kubernetes.md) |
| Builds | `shem build` and a Go [BuildKit frontend](shemfile.md#with-docker-build) |
| Guest toolchain | Zig targeting `riscv32-linux-musl`, in [`toolchain/Dockerfile`](../toolchain/Dockerfile) |
| Tests | `cargo test --workspace`, `just guest-tests`, `just neural-tests`, and the integration jobs in [CI](../.github/workflows/ci.yml) |

## Roadmap

- **Neural execution.** Narrower types for bits, fewer tensor operations per step, and GPU execution in the vCPU server.
- **Builds and filesystems.** A layer-digest build cache, overlayfs instead of copied filesystems, and coherence between processes sharing writable file mappings.
- **Guest coverage.** Terminals under `docker run -t` for program containers, `/proc/PID/fd`, sockets, and VM block and network devices.
- **ISA validation.** Cross-check the reference interpreter against Spike.
