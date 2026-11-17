# Architecture

shem's host supplies system calls, devices, scheduling and storage. Programs run on either a reference RV32IM interpreter or a neural vCPU whose weights implement the CPU and whose state holds guest memory.

The neural vCPU is inspired by [Qwenputer](https://divagr.com/blog/qwenputer), but is a standalone compiled network, not part of a pretrained language model. Language-model sessions run separately and can [call RV32 programs as tools](models.md#compiled-layers-and-tools).

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

## What is in the weights, and what is not

The tagline says containers run inside a model's weights. Precisely:

| Part | Where it runs | In a language model's weights? |
|---|---|---|
| The neural vCPU | A standalone network of compiled weights: fetch, decode, units, writeback ([neural.md](neural.md)) | No. It is its own network, published as [`jgalego/shem-rv32im`](https://huggingface.co/jgalego/shem-rv32im) |
| Its registers and memory | Delta-rule state of that network, or pages the fused executor keeps | No: the network's own, not a model's recurrent state |
| A language model | PyTorch, from the checkpoint's own tensors, block by block ([models.md](models.md)) | Yes: it is the model |
| A model's session | The model's KV caches and recurrent state | Yes, but it holds a conversation, not guest memory |
| A tool call | The host runs the program on the compiled CPU and returns the output as text | The two meet only in text: the CPU does not share the model's residual stream or state |

[Qwenputer](https://divagr.com/blog/qwenputer) goes further than shem does: its CPU layers sit inside a pretrained model, so one forward pass is both. Joining the two here would take layers that read and write the residual stream directly, with a token's embedding carrying an instruction and a Gated DeltaNet block's state as the guest's memory, a design the compiled CPU's delta-rule state suggests but nothing has tested. Nothing of that is built. The pieces that exist are the part of the CPU that does not depend on it, the [`Vcpu` interface](#the-vcpu-interface), and the receipts that would let anyone check such a run. Until a model's own forward pass executes the guest, "inside a model" means the model and the machine are both neural networks, driven by one host.

## Two modes, one runtime

| | Container (`shem run`) | VM (`shem boot`) |
|---|---|---|
| Guest runs | Programs, with the host as their kernel | A kernel and its programs |
| Guest exits on | System calls | Device access, traps, timer ticks |
| Host provides | Linux system calls ([`linux/`](../crates/shem/src/linux/)) | The machine: CSRs, traps, a timer, a UART and a syscon ([vm.md](vm.md)) |
| ISA needed | RV32IM (and C, for a program built with it: the ELF header's RVC flag turns it on); the host emulates the A extension | RV32IM too; the host emulates A, the CSRs and traps |
| Start | Load an ELF into memory, or thaw a warm image | Boot a kernel, or thaw a booted VM |

## The vCPU interface

[`vcpu.rs`](../crates/shem/src/vcpu.rs) defines the `Vcpu` trait that both the reference interpreter and a neural vCPU implement:

- `run(budget)` executes at most `budget` instructions and returns an `Exit`.
- `reg`, `set_reg`, `pc`, `set_pc`, `read_mem`, `write_mem` access registers and memory.
- `snapshot` and `restore` use `MachineState`, a canonical, vCPU-independent format holding the pc, registers, instruction count and memory.

Exit semantics: except for `Budget`, the exit's `pc` points at the instruction that caused it, and that instruction has not retired. For a system call the host writes the result to `a0` and resumes at `pc + 4`.

Both vCPUs must satisfy:

1. **Trace equivalence.** Registers, memory and exits match [`ReferenceVcpu`](../crates/shem/src/reference.rs) instruction for instruction.
2. **Canonical snapshots.** All state can be restored from `MachineState` at any instruction boundary. A guest can boot on the reference vCPU and resume on the neural one.

## Design rules that come from running on a model

1. **The host handles unsupported operations.** The vCPU exits for system calls, atomics and, in VM mode, privileged instructions and device accesses. [`emulate.rs`](../crates/shem/src/emulate.rs) handles the A extension because musl's riscv32 port needs LR/SC.
2. **Batching supports unrelated guests.** Tensor routing lets one batch run different programs. Throughput and per-guest latency depend on batch size, workload and device ([benchmarks](neural.md#speed)).
3. **Instruction budgets provide preemption.** The host drives execution and stops the vCPU after a bounded number of retired instructions.
4. **`fork()` clones vCPU state.** The reference interpreter copies memory; the neural server shares pages until a write. Threads share their process's vCPU memory and save their registers between turns ([`container.rs`](../crates/shem/src/container.rs)).
5. **The host manages memory backing.** The fused CPU executor uses copy-on-write pages; the dense executor uses delta-rule state, and the GPU harness uses flat byte arrays. This is not a guest MMU.
6. **Builds are reproducible with fixed inputs.** Time is virtual, randomness is seeded, directory listings are sorted, and layer metadata is normalized ([reproducibility](shemfile.md#reproducibility)). Interactive replay also needs the external inputs.

## Isolation

- The only way out of a guest is an exit, so the attack surface is the exit handler. The [system-call layer](../crates/shem/src/linux/) treats every guest pointer and length as hostile: transfers are capped, and bad pointers return `EFAULT`.
- Guest paths resolve inside the root directory with chroot semantics ([`rootfs.rs`](../crates/shem/src/linux/rootfs.rs)): `..` stops at the root, and symlinks, absolute ones included, resolve relative to it. Resolution and the final open are separate host calls, so a concurrent writer to the host directory could race them; a single guest cannot. On Linux 5.6 and later the open is made with `openat2`, beneath the root, refusing any symlink on the way ([`RootFs::open_beneath`](../crates/shem/src/linux/rootfs.rs)), so a link swapped in after the path was resolved leads nowhere: opening a file, the call that would give a guest a host file's contents, no longer races. `mkdir`, `unlink`, `symlink` and `rename` act on the directory that holds their path, opened beneath the root the same way, so they cannot be sent through a swapped link either. `stat`, `readlink`, `utimensat` and directory listings are read the same way, from a descriptor opened beneath the root. Only a host without `openat2` (Linux before 5.6, or another system) does everything by path as before.
- A container has a network of its own and no other: its processes can listen on `AF_UNIX` paths and loopback ports, connect to them and talk over stream sockets, and `socketpair` works, but no socket leaves the container, so an untrusted guest can serve and call its siblings and reach no host or network ([`linux/net.rs`](../crates/shem/src/linux/net.rs)). A connection is two pipes, so it blocks, polls and closes as a pipe does. Not yet: datagrams, `AF_INET6`, addresses beyond loopback, `sendmsg` and `recvmsg`, descriptor passing, and a file for a Unix socket's path.
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
| ISA | The reference vCPU passes the riscv-tests (rv32uc, the C extension, included), and agrees with QEMU's user-mode emulator on random RV32IM programs, and on programs with compressed instructions among them ([`tests/external.rs`](../crates/shem/tests/external.rs), in CI) |
| Tests | `cargo test --workspace`, `just guest-tests`, `just neural-tests`, and the integration jobs in [CI](../.github/workflows/ci.yml) |

## Roadmap

See [ROADMAP.md](../ROADMAP.md) for planned work.
