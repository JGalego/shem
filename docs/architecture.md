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

| | Container (`shem run`) | VM (`shem boot`) |
|---|---|---|
| Guest runs | Programs, with the host as their kernel | A kernel and its programs |
| Guest exits on | System calls | Device access, traps, timer ticks |
| Host provides | Linux system calls ([`linux/`](../crates/shem/src/linux/)) | The machine: CSRs, traps, a timer, a UART and a syscon ([vm.md](vm.md)) |
| ISA needed | RV32IM; the host emulates the A extension | RV32IM too; the host emulates A, the CSRs and traps |
| Start cost | Load an ELF into memory, or thaw a warm image | Boot a kernel: about two seconds on the reference vCPU |

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

1. **Exits are relatively cheap.** On real hardware a VM exit costs about a thousand instructions, so hypervisors avoid them. Here each guest instruction is a forward pass, about 1 ms at Qwenputer's published speed, and an exit costs a GPU sync plus a few tensor reads: by our estimate, about as much as a few instructions. So anything awkward goes to the host: system calls, atomics, page faults, possibly floating point. Atomics already work this way: the vCPU exits on an A-extension instruction and [`emulate.rs`](../crates/shem/src/emulate.rs) carries it out, since musl's riscv32 port cannot be built without LR/SC.
2. **Batching doesn't slow guests down when they diverge.** Every instruction runs the same fixed matrix multiplications, so many guests can decode in one batch for little more than the cost of one, until the GPU becomes compute-bound. Each guest is slow, but one GPU can run many of them. The batch scheduler is the hypervisor's scheduler: a guest that exits sits out while the others keep decoding.
3. **Preemption is free.** The host drives every step, so a step budget replaces a timer interrupt.
4. **`fork()` is a state copy.** Each guest process is its own vCPU, with the host acting as the kernel: fork clones the vCPU, and pipes are host buffers. This is how [`container.rs`](../crates/shem/src/container.rs) works today.
5. **The host can be the MMU.** Keep the guest's working set in the state and the rest in host memory. A miss exits and the host swaps rows in.
6. **Determinism gives exact replay.** Execution is deterministic, so logging the exits is enough to replay a run bit for bit. It also makes build-time snapshots reproducible (see [the Shemfile spec](shemfile.md)). The host keeps its side deterministic too: the clock is virtual (one nanosecond per retired instruction, starting at 2026-01-01, and sleeping advances it instantly), `getrandom` is a seeded PRNG, and directory listings are sorted. File metadata does not leak the host either: files the guest creates, writes, renames or removes, and their directories, are stamped with the virtual time, inode numbers count up in the order files are first seen, and the device and block counts are fixed. Running the same thing twice prints the same `ls -li`.

## Isolation

- The only way out of a guest is an exit, so the attack surface is the exit handler. The [system-call layer](../crates/shem/src/linux/) treats every guest pointer and length as hostile: transfers are capped, and bad pointers return `EFAULT`.
- Guest paths resolve inside the root directory with chroot semantics ([`rootfs.rs`](../crates/shem/src/linux/rootfs.rs)): `..` stops at the root, and symlinks, absolute ones included, resolve relative to it. Resolution and the final open are separate host calls, so a concurrent writer to the host directory could race them; a single guest cannot. `openat2` with `RESOLVE_IN_ROOT` would close that gap on Linux hosts.
- In a hybrid model, the language layers share activations with the CPU layers. Whether they can write into the CPU's state heads is a property of the weights. If the compiled heads are disjoint (zero blocks in the relevant projections), isolation can be checked by inspecting the matrices.
- Guest output that flows back into a language model's context is untrusted input, and a possible prompt-injection channel. The host should filter it like any other device output.

## Tech stack

| Part | Choice | Why |
|---|---|---|
| Runtime, CLI, system-call layer, reference vCPU, image store | Rust | The system-call layer is the security boundary; Firecracker, crosvm, youki and runwasi are Rust for the same reason. The reference vCPU has to be fast enough to run build-time snapshots. Ships as one static binary. |
| Neural vCPU: weight compiler and executor | Python and PyTorch, in [`neural/`](../neural/); Triton kernels for GPUs later | The GPU ecosystem. The executor runs on CPUs today; Gated DeltaNet kernels already exist in Triton ([flash-linear-attention](https://github.com/fla-org/flash-linear-attention)). |
| Host to neural vCPU | A small KVM-style protocol over a Unix socket ([neural.md](neural.md#the-protocol)) | One executor serves many containers. A request per exit is small next to about 1 ms per guest instruction. |
| Images | OCI image spec and registries | Digests, layer sharing, signing and existing registries for free |
| containerd integration | A containerd shim, as runwasi does for Wasm ([containerd.md](containerd.md)) | `ctr run --runtime io.containerd.shem.v1`; Docker and Kubernetes reach runtimes through containerd |
| Builds | Own builder first, then a BuildKit frontend in Go (planned) | BuildKit frontends are written in Go |
| Guest toolchain | Zig (`zig cc`) targeting `riscv32-linux-musl`, in [`toolchain/Dockerfile`](../toolchain/Dockerfile) | One hermetic binary that builds musl and compiler-rt for the exact target CPU. Works the same on x86-64 and arm64 hosts. |
| Testing | `cargo test`; the official riscv-tests (`just guest-tests`); differential tests of the neural vCPU against the reference (`just neural-tests`); Spike as a golden model (planned) | |

## Roadmap

1. **Reference vCPU and container runtime.** Done: RV32IM, static ELF loading, the Linux stack layout, and a minimal system-call layer. The reference vCPU runs about 175 million instructions per second (measured on a simple loop).
2. **ISA compliance.** Done: all 60 tests in the official rv32ui, rv32um and rv32ua suites pass, the last through host emulation. Still to do: cross-check against Spike.
3. **Real programs.** Done for single processes. C programs linked against musl run, and so does a static BusyBox with a host directory as its root filesystem (`shem run --root`): text tools, `sha256sum`, `awk`, `find`, file and directory operations, `date` and `sleep` on the virtual clock, and `sh` scripts that use only builtins. Not yet: `df` and anything else that reads `/proc`.
4. **Processes.** Done. `fork` (and `vfork`, as a copy), `execve` with `#!` scripts, `wait4` and `waitid`, pipes, process groups, and signals delivered to real handlers: the host pushes a frame onto the guest stack and returns through an `rt_sigreturn` trampoline. A round-robin scheduler gives each process a slice of instructions in turn, and parks a process whose system call cannot finish yet, retrying the call on its next turn. BusyBox `sh` runs pipelines, command substitution, background jobs and `trap` handlers. Files can be mapped, as private copies. Not yet: threads, `/proc` and terminals.
5. **Neural vCPU.** Done. An RV32IM CPU compiled into network weights, one instruction per forward pass, with memory and registers in delta-rule state, served to the host over a Unix socket ([neural.md](neural.md)). It matches the reference vCPU exactly on all 60 riscv-tests, on random programs, on resuming a reference snapshot, and on forking containers. It runs about 900 instructions per second on one CPU core.
6. **Images.** Done. `shem build` turns a Shemfile into an OCI image layout with reproducible layers, building ordinary Docker stages for `COPY --from`. `WARM` freezes a running container into a state layer on the reference vCPU, and `shem run` thaws it, on either vCPU ([shemfile.md](shemfile.md)). Images live in a local store, and `shem push` and `shem pull` move them through OCI registries. Layers are compressed with zstd. Not yet: a build cache, overlayfs.
7. **Density.** Done for the neural executor: memory in shared pages with copy-on-write and content deduplication, so forks and containers resumed from one warm image share memory, and continuous batching of every running vCPU into one forward pass, which multiplies throughput about 32 times at 64 guests on one CPU core ([neural.md](neural.md#density)). On GPUs, a static executor with flat memory runs compiled and in CUDA graphs, bit-exact against the CPU, at up to 12,859 instructions per second for one guest and 7.4 million for 32,768 guests on an A100 ([neural.md](neural.md#on-gpus)).
8. **VM mode.** Done on the reference vCPU. `shem boot` boots a no-MMU Linux 6.8 to a shell in about 350 million instructions, two seconds, with the host emulating the privileged architecture, the timer and the devices ([vm.md](vm.md)). Machines freeze and thaw, and a VM image can ship a kernel that booted at build time, so `shem run` starts at its prompt. VM mode works on the neural vCPU too: Linux, booted on the reference vCPU, runs a shell command on the neural one and ends in the same state. Not yet: block and network devices.
9. **Ecosystem.** Partly done. A containerd shim, `containerd-shim-shem-v1`, runs RV32 executables from ordinary OCI images on shem through `ctr run --runtime io.containerd.shem.v1` ([containerd.md](containerd.md)), and CI runs it. Registries work through `shem push` and `shem pull`. Not yet: a BuildKit frontend for Shemfiles, and Kubernetes.

### Next

- **A BuildKit frontend** for Shemfiles, so `docker buildx build` parses, caches and pushes them.
- **Faster neural execution.** Bits in narrow float types and only values in float64, fewer and larger operations per forward pass, and GPU execution in the server.
- **Builds and filesystems.** A build cache keyed on layer digests, overlayfs instead of copied layers, and cross-checking the reference vCPU against Spike.
- **Guest coverage.** `/proc`, terminals, threads, and shared writable file mappings.
