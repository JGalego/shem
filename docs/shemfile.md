# Shemfile (draft)

A Shemfile describes a model container image, the way a Dockerfile describes a container image. It uses Dockerfile syntax, keeps Docker's instructions where their meaning carries over, and adds two new ones. Nothing here is implemented yet; this is the design that the builder will follow.

## Example

```dockerfile
# Stage 1: an ordinary Docker stage that cross-compiles the guest program
FROM shem-toolchain AS build
COPY hello.c .
RUN rv32-cc -O2 -o hello hello.c

# Stage 2: the model container
FROM ghcr.io/shem/qwen3.5-4b:rv32im      # machine image: model weights + compiled CPU
COPY --from=build /hello /bin/hello      # guest filesystem (lives on the host)
WORKDIR /root
ENTRYPOINT ["/bin/hello"]
WARM --until=syscall:read                # run at build time, save the state as a layer
```

`shem-toolchain` is the image built from [`toolchain/Dockerfile`](../toolchain/Dockerfile) (`just toolchain`); `ghcr.io/shem/qwen3.5-4b:rv32im` is illustrative.

## Stages

A stage whose `FROM` is a machine image or another shem image is a **shem stage**. Any other stage is an **ordinary Docker stage**: it is built by Docker, and shem stages can `COPY --from` it.

## Instructions

| Instruction | Meaning in a shem stage |
|---|---|
| `FROM` | A machine image (model weights plus compiled CPU) or a shem image |
| `COPY`, `ENV`, `WORKDIR`, `ENTRYPOINT`, `CMD`, `ARG`, `LABEL` | Same as in Docker. In container mode the guest filesystem lives on the host, because the host implements the system calls. |
| `RUN` | Not allowed. At about 1,000 instructions per second on a neural vCPU, a compiler would take days. Build in an ordinary stage and `COPY --from`. |
| `WARM` | New. See below. |
| `CPU` | New, optional. Compiles CPU layers into a plain base model, e.g. `FROM qwen3.5-4b` followed by `CPU rv32im`. Most images start from a prebuilt machine image instead. |

Resource limits and the system-call policy are `shem run` flags (`--mem`, `--max-steps`, a planned `--syscalls`), not Shemfile instructions, as in Docker.

### WARM

```dockerfile
WARM --until=<condition> [--max-steps=<n>]
```

Runs the entrypoint at build time until the condition holds, then saves the machine as a layer. `shem run` resumes from that point instead of starting cold. Docker cannot do this: it has no way to put a running process in an image, short of its experimental CRIU checkpointing. In shem a running process is a tensor, so it can be a layer.

Conditions:

- `syscall:<name>`: stop just before the guest's first `<name>` system call is serviced, so at run time that call sees the real input.
- `steps:<n>`: stop after `n` instructions.
- `ebreak`: stop when the guest executes `ebreak`. A program can use this to mark itself ready.

`--max-steps` fails the build if the condition is not reached in time. A stage may contain several `WARM` steps; each one resumes from the previous snapshot, and each is cached separately.

A `WARM` layer holds two things:

1. The vCPU's machine state in the canonical format (`MachineState` in [`vcpu.rs`](../crates/shem/src/vcpu.rs)): pc, registers, instruction count, and the memory rows that changed since the parent snapshot.
2. The host-side process state, because in container mode the kernel lives on the host: open files and offsets, the program break, the working directory.

Filesystem writes made during `WARM` become an ordinary filesystem layer.

**WARM can run on the reference vCPU.** Snapshots are canonical, and a neural vCPU must be trace-equivalent to the reference one. So the builder can run `WARM` on the fast reference vCPU, at about 175 million instructions per second instead of about a thousand, and the snapshot then resumes on the model. This is what makes VM mode practical: boot the kernel at build time on the CPU, and ship the booted state.

## Image format

A shem image is an OCI image manifest. Its config and layers are:

| Media type | Contents |
|---|---|
| `application/vnd.shem.config.v1+json` | Entrypoint, cmd, env, workdir, and the machine it needs: ISA, state geometry, base model digest |
| `application/vnd.shem.weights.v1+safetensors` | Base model weights, or a patch holding the compiled CPU layers. Stored once and shared by digest across every image built on them. |
| `application/vnd.oci.image.layer.v1.tar` (also `+gzip`, `+zstd`) | Guest filesystem layers, in the standard format |
| `application/vnd.shem.state.v1+safetensors` | A `WARM` snapshot: changed memory rows plus host process state |

Images push to and pull from any OCI registry (Docker Hub, GHCR) with standard tooling such as ORAS. Docker Model Runner and KitOps already ship models as OCI artifacts.

## Caching and reproducibility

Each instruction's cache key is its parent layer's digest, the instruction itself, and the digests of its inputs, as in BuildKit. Execution is deterministic, so a `WARM` layer depends only on that key. Caching it is always safe, and anyone can rebuild it and get the same digest.

That holds only if the guest's inputs are fixed. During builds the host provides a fixed clock, fixed `AT_RANDOM` bytes (the runtime already does this), empty stdin and no network.

## Copy-on-write

- **Filesystem.** overlayfs on the host, exactly as Docker uses it.
- **Guest memory**, from least to most effort:
  1. **Fork by copying.** The state is a few MB, and a GPU copies that in microseconds. Full copies cost memory, not speed.
  2. **Paged state.** Group state rows into pages, give each container a page table, count references, and copy a page only on the first write to a shared one. This is Linux `fork()` and vLLM's block sharing. A thousand containers resumed from one snapshot then share almost all their memory. It needs a custom executor, because the stock DeltaNet kernel updates the whole state matrix every step. Each step reads one row and writes one row per head, so routing those accesses through a page table fits naturally.
  3. **Diff-only layers.** Every write goes to a known row, so a `WARM` layer stores only the rows that changed since its parent.

## Build plan

1. A small builder of our own that writes OCI image layouts, so registries and digests work from day one. Ordinary stages are built with `docker buildx build --target <stage> --output type=local`.
2. Later, a BuildKit frontend (`# syntax=ghcr.io/shem/shemfile:1`), so `docker buildx build` does the parsing, caching and pushing. `WARM` steps on a neural vCPU need a GPU inside the build. BuildKit has added device support for that; check that it is mature before relying on it. `WARM` steps on the reference vCPU need no GPU.

## Open questions

- Which OCI platform should shem images declare, so Docker and containerd pick the shem shim? runwasi uses `wasi/wasm`.
- Should a machine image pin its exact state geometry (heads, rows, width)? Diff-only state layers depend on the row layout. Canonical snapshots do not.
- How much host process state should a `WARM` layer carry before it becomes CRIU by another name?
