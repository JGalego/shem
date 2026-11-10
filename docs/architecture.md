# Architecture

shem treats the host as a hypervisor and a model as the machine it runs. The model supplies a CPU, compiled into its weights, and RAM, which is its recurrent state. The host supplies everything else: system calls, devices, scheduling, paging and storage.

The starting point is [Qwenputer](https://divagr.com/blog/qwenputer), which compiles an RV32IM CPU into Qwen3.5-4B and uses the model's Gated DeltaNet state as addressable memory. shem is not tied to that model: any model that exposes the [vCPU interface](#the-vcpu-interface) can run guests.

## How virtualization maps onto a model

| Virtualization | shem |
|---|---|
| vCPU | Compiled CPU layers |
| Guest RAM | DeltaNet state. A one-hot key writes one row and a one-hot query reads it back. |
| `KVM_RUN` loop | Decode loop |
| VM exit | The vCPU stops on `ecall`, `ebreak`, an illegal instruction or a memory fault |
| DMA, virtio rings | The host reads and writes state rows directly, without decoding any tokens |
| Timer interrupt | The host stops decoding after a step budget |
| Snapshot, migration | Save the state tensors |
| Multi-tenancy | The batch dimension |

## Two modes, one runtime

| | Container (`shem run`) | VM (`shem boot`, planned) |
|---|---|---|
| Guest runs | One user-mode program | A kernel and its programs |
| Guest exits on | System calls | Device access, traps, timer ticks |
| Host provides | Linux system calls ([`linux.rs`](../crates/shem/src/linux.rs)) | Virtual devices: console, block, timer |
| ISA needed | RV32IM | RV32IMA plus Zicsr, M-mode traps, a timer |
| Start cost | Load an ELF into memory | Boot a kernel, or resume a booted snapshot |

Container mode comes first because user-mode RV32IM is exactly what a Qwenputer already implements, and there is no kernel to boot.

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

1. **Exits are relatively cheap.** On real hardware a VM exit costs about a thousand instructions, so hypervisors avoid them. Here each guest instruction is a forward pass, about 1 ms at Qwenputer's published speed, and an exit costs a GPU sync plus a few tensor reads: by our estimate, about as much as a few instructions. So anything awkward goes to the host: system calls, atomics, page faults, possibly floating point.
2. **Batching doesn't slow guests down when they diverge.** Every instruction runs the same fixed matrix multiplications, so many guests can decode in one batch for little more than the cost of one, until the GPU becomes compute-bound. Each guest is slow, but one GPU can run many of them. The batch scheduler is the hypervisor's scheduler: a guest that exits sits out while the others keep decoding.
3. **Preemption is free.** The host drives every step, so a step budget replaces a timer interrupt.
4. **`fork()` is a state copy.** Each guest process could be its own sequence, with the host acting as the kernel: fork clones the state, pipes are host buffers.
5. **The host can be the MMU.** Keep the guest's working set in the state and the rest in host memory. A miss exits and the host swaps rows in.
6. **Determinism gives exact replay.** Execution is deterministic, so logging the exits is enough to replay a run bit for bit. It also makes build-time snapshots reproducible (see [the Shemfile spec](shemfile.md)).

## Isolation

- The only way out of a guest is an exit, so the attack surface is the exit handler. [`linux.rs`](../crates/shem/src/linux.rs) treats every guest pointer and length as hostile: transfers are capped, and bad pointers return `EFAULT`.
- In a hybrid model, the language layers share activations with the CPU layers. Whether they can write into the CPU's state heads is a property of the weights. If the compiled heads are disjoint (zero blocks in the relevant projections), isolation can be checked by inspecting the matrices.
- Guest output that flows back into a language model's context is untrusted input, and a possible prompt-injection channel. The host should filter it like any other device output.

## Tech stack

| Part | Choice | Why |
|---|---|---|
| Runtime, CLI, system-call layer, reference vCPU, image store | Rust | The system-call layer is the security boundary; Firecracker, crosvm, youki and runwasi are Rust for the same reason. The reference vCPU has to be fast enough to run build-time snapshots. Ships as one static binary. |
| Neural vCPU: weight compiler and GPU executor | Python, PyTorch, Triton (planned, in `neural/`) | The GPU ecosystem. Gated DeltaNet kernels already exist in Triton ([flash-linear-attention](https://github.com/fla-org/flash-linear-attention)). |
| Host to neural vCPU | A small KVM-style RPC over a Unix socket (planned) | One GPU executor serves many containers. An RPC per exit is small next to about 1 ms per guest instruction. |
| Images | OCI image spec and registries | Digests, layer sharing, signing and existing registries for free |
| Docker integration | A containerd shim, as runwasi does for Wasm (planned) | `docker run --runtime io.containerd.shem.v1` through Docker's containerd image store |
| Builds | Own builder first, then a BuildKit frontend in Go (planned) | BuildKit frontends are written in Go |
| Guest toolchain | Clang or GCC for rv32im/ilp32, in a Docker build stage | |
| Testing | `cargo test`; the official riscv-tests; Spike as a golden model; differential traces between neural and reference vCPUs | |

## Roadmap

1. **Reference vCPU and container runtime.** Done: RV32IM, static ELF loading, the Linux stack layout, and a minimal system-call layer (`read`, `write`, `writev`, `close`, `brk`, `exit`).
2. **ISA compliance.** Run the official riscv-tests against the reference vCPU, and cross-check it against Spike.
3. **Real programs.** Filesystem system calls backed by a host root directory (`openat`, `fstat`, `lseek`, `getdents64`), anonymous `mmap`, a fixed clock. Goal: a static BusyBox.
4. **Neural vCPU.** A weight compiler and Triton executor behind the KVM-style RPC, with differential trace tests against the reference vCPU.
5. **Images.** A Shemfile builder that writes OCI layouts, plus `WARM` snapshots. See [shemfile.md](shemfile.md).
6. **Density.** Many guests batched on one GPU, and paged state with copy-on-write.
7. **VM mode.** `shem boot`, virtual devices, a no-MMU Linux kernel.
8. **Ecosystem.** A containerd shim and a BuildKit frontend.
