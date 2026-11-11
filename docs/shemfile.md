# Shemfile

A Shemfile describes a shem image, the way a Dockerfile describes a container image. It uses Dockerfile syntax, keeps Docker's instructions where their meaning carries over, and adds `WARM`, which freezes a running guest into the image.

```sh
shem build -t hello examples/hello       # build into the local image store
shem run hello                           # run it
shem images                              # list the store
shem build -o hello.oci examples/hello   # or write an OCI image layout directory
```

## Examples

[`examples/hello`](../examples/hello/Shemfile) compiles a C program in an ordinary Docker stage and ships it:

```dockerfile
FROM shem-toolchain AS build
WORKDIR /src
COPY hello.c .
RUN rv32-cc -O2 -o hello hello.c

FROM scratch
COPY --from=build /src/hello /bin/hello
ENTRYPOINT ["/bin/hello"]
CMD ["from", "a", "shem", "image"]
```

[`examples/warm-shell`](../examples/warm-shell/Shemfile) starts a BusyBox shell at build time and freezes it just before it reads its script:

```dockerfile
FROM shem-busybox AS busybox

FROM scratch
COPY --from=busybox / /
ENV PATH=/bin:/sbin:/usr/bin:/usr/sbin
ENTRYPOINT ["/bin/sh"]
WARM --until=syscall:read
```

```sh
$ printf 'echo hello from a warm shell\nuname -m\n' | shem run warm-shell
hello from a warm shell
riscv32
```

`shem-toolchain` and `shem-busybox` are Docker images built from [`toolchain/Dockerfile`](../toolchain/Dockerfile) by `just toolchain` and `just busybox-image`.

## Stages

The last stage is the image. It must be a **shem stage**: `FROM scratch`, or `FROM` the directory of a shem image's OCI layout, relative to the build context. Every other stage is an **ordinary Docker stage**, built by Docker only when a `COPY --from` needs it. The builder adds a small `FROM scratch` export stage that copies just the requested paths, so only those files leave Docker.

## Instructions

| Instruction | In a shem stage |
|---|---|
| `FROM` | `scratch`, or a shem image's layout directory, whose layers, config and frozen state the new image builds on |
| `COPY [--from=stage] src... dest` | As in Docker: a directory's contents are copied into `dest`; a file goes to `dest`, or into it when `dest` ends in `/` or there are several sources. Each `COPY` is one layer. Sources must stay inside the context. |
| `ENV`, `WORKDIR`, `ENTRYPOINT`, `CMD`, `ARG`, `LABEL` | As in Docker, including shell and exec forms, and `$NAME` and `${NAME}` substitution from `ARG` and `ENV` |
| `WARM` | New; see below |
| `RUN` | Not allowed: at about a thousand instructions per second on a neural vCPU, a compiler would take days. Build in a Docker stage and `COPY --from` it. |

Resource limits are `shem run` flags (`--mem`, `--max-steps`), not Shemfile instructions, as in Docker.

### WARM

```dockerfile
WARM --until=<condition> [--max-steps=<n>]
```

Runs the entrypoint at build time until the condition holds, then freezes the container into a state layer. `shem run` thaws it and carries on from that point instead of starting cold. Docker cannot do this: it has no way to put a running process in an image, short of experimental CRIU checkpointing. In shem a running process is a vCPU state, so it can be a layer.

Conditions:

- `syscall:<name>` (or `syscall:<number>`): stop just before the guest's first such system call is serviced, so at run time the call sees the real input. Names include `read`, `write`, `openat`, `execve`, `wait4` and `clock_nanosleep`.
- `steps:<n>`: stop once `n` instructions have retired.
- `ebreak`: stop when the guest executes `ebreak`, and resume after it. A program can use this to mark itself ready.

`--max-steps` fails the build if the condition is not reached in time (default: 4 billion instructions). A later `WARM` resumes from the previous one's state.

The guest runs with an empty stdin; its output goes to the build's stderr. The frozen container must be a single process with only stdin, stdout and stderr open: other open files live on the host and are not part of the snapshot yet.

A state layer holds:

1. `machine.state`: the vCPU's machine state in the canonical format (`MachineState` in [`vcpu.rs`](../crates/shem/src/vcpu.rs)): pc, registers, instructions retired, and memory.
2. `container.json`: the host's side of the process, since the kernel lives on the host: working directory, umask, memory map, signal dispositions and mask, the LR/SC reservation, the virtual clock and the PRNG state.

Files the guest wrote during `WARM` become an ordinary filesystem layer, with whiteouts for deletions.

**WARM runs on the reference vCPU.** Snapshots are canonical, and the neural vCPU is trace-equivalent to the reference one, so the builder runs `WARM` at about 175 million instructions per second instead of about a thousand, and the snapshot resumes on the model. A test builds a warm image this way and runs it on the neural vCPU. This is also what will make VM mode practical: boot a kernel at build time on the CPU, and ship the booted state.

A warm image takes no arguments or environment at run time, and runs with the memory it was frozen with.

## Image format

A shem image is an [OCI image layout](https://github.com/opencontainers/image-spec/blob/main/image-layout.md) whose manifest has:

| Media type | Contents |
|---|---|
| `application/vnd.shem.config.v1+json` | Entrypoint, cmd, env, workdir, labels, the machine (`rv32im`), and a warm image's memory size |
| `application/vnd.oci.image.layer.v1.tar` | Guest filesystem layers, in the standard format |
| `application/vnd.shem.state.v1.tar` | A `WARM` snapshot, always the last layer |

Images in the local store live under `$SHEM_HOME/images/<name>/<tag>`, by default `~/.local/share/shem/images`. Each run unpacks the image's filesystem into a fresh directory and removes it afterwards, like `docker run --rm`.

Planned: a `application/vnd.shem.weights.v1+safetensors` layer for machine images that carry model weights, pushing and pulling through OCI registries, and compressed layers.

## Reproducibility

Layers have sorted entries, timestamps of 0 and root ownership, and execution is deterministic: the clock is virtual, the PRNG is seeded and stdin is empty during builds. So building the same Shemfile twice gives the same manifest digest, warm images included. Tests check both. A build cache keyed on those digests, as in BuildKit, is planned.

## Copy-on-write

- **Filesystem.** Today each run gets a full copy of the layers. Planned: overlayfs on the host, as Docker uses it.
- **Guest memory**, from least to most effort:
  1. **Fork by copying.** What the container does today: `fork` clones the vCPU's state. Full copies cost memory, not speed.
  2. **Paged state.** Group state rows into pages, give each container a page table, count references, and copy a page only on the first write to a shared one. This is Linux `fork()` and vLLM's block sharing. A thousand containers resumed from one snapshot then share almost all their memory. Each step reads and writes one row per head, so routing those accesses through a page table fits naturally.
  3. **Diff-only layers.** Every write goes to a known row, so a state layer could store only the rows that changed since its parent.

## Later

- A BuildKit frontend (`# syntax=ghcr.io/shem/shemfile:1`), so `docker buildx build` does the parsing, caching and pushing.
- Which OCI platform should shem images declare, so Docker and containerd pick a shem shim? runwasi uses `wasi/wasm`.
- How much host process state should a state layer carry before it becomes CRIU by another name?
