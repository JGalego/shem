# Shemfile

A Shemfile describes a shem image, the way a Dockerfile describes a container image. It uses Dockerfile syntax, keeps Docker's instructions where their meaning carries over, and adds `WARM`, which freezes a running guest into the image.

```sh
shem build -t hello examples/hello       # build into the local image store
shem run hello                           # run it
shem images                              # list the store
shem build -o hello.oci examples/hello   # or write an OCI image layout directory
shem push hello ghcr.io/me/hello:1.0     # push to an OCI registry
shem pull ghcr.io/me/hello:1.0           # pull into the store, and run it by that name
```

## With docker build

`docker build` builds Shemfiles too, through shem's BuildKit frontend ([`frontend/`](../frontend/)). Make the first line name it:

```dockerfile
# syntax=ghcr.io/jgalego/shem/dockerfile:latest
FROM hf.co/Qwen/Qwen3.5-0.8B
ENTRYPOINT ["shem:generate", "--system", "You are Shem, a golem made of clay."]
CMD ["Who are you?"]
WARM --until=prompt
```

```sh
docker build --platform linux/riscv32 -t golem .
```

BuildKit pulls and runs the frontend, which turns the Shemfile into a Dockerfile and has BuildKit's own Dockerfile frontend build it:

- `FROM hf.co/ORG/NAME[:REVISION]` runs the importer in the `shem-neural` image, then copies each weights file into a layer of its own, as `shem build` does. A gated model's token comes from the build's secret `HF_TOKEN` (`docker build --secret id=HF_TOKEN,env=HF_TOKEN`), and downloads are cached between builds.
- `WARM --until=prompt` runs the model's driver over the prompt in the `shem-neural` image and copies the primed session into `/.shem`.
- `KERNEL` becomes the image's `dev.shem.kernel` and `dev.shem.bootargs` labels.
- Other stages are Docker stages, built on the build platform, and `RUN` is refused in shem stages, as with `shem build`.

The image is for `linux/riscv32` (pass `--platform linux/riscv32` to keep Docker from warning that it is not the host's). `docker push`, `docker save` and the rest work as for any image, and `shem run` runs what `docker save` writes. Not yet: `WARM` for programs and VMs, which `shem build` does; the build argument `SHEM_NEURAL_IMAGE` picks another runtime image.

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
| `KERNEL path [command line]` | New: makes a VM image; see [VM images](#vm-images) |
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

**WARM runs on the reference vCPU.** Snapshots are canonical, and the neural vCPU is trace-equivalent to the reference one, so the builder runs `WARM` at about 175 million instructions per second instead of about a thousand, and the snapshot resumes on the model. A test builds a warm image this way and runs it on the neural vCPU. VM images use the same idea: the kernel boots at build time on the CPU, and the image ships the booted state.

A warm image takes no arguments or environment at run time, and runs with the memory it was frozen with.

### VM images

```dockerfile
KERNEL <path> [command line]
```

Makes the image a VM image: `shem run` boots the kernel at `path` in the image's filesystem in [VM mode](vm.md), with the console on stdio, rather than starting a program. The rest of the line is the kernel command line (default: `earlycon=uart8250,mmio,0x10000000,1000000 console=ttyS0`). Memory is `--mem`, 64 MiB by default.

In a VM image, `WARM` boots the kernel at build time and freezes the machine. Its conditions are:

- `output:TEXT`: stop once the console's output ends with `TEXT`, such as a shell's prompt. Quote it to include spaces; in double quotes, JSON escapes such as `\n` work.
- `steps:<n>`: stop once `n` instructions have retired.

The state layer holds `machine.state` and `vm.json`, the host's half of the hardware (see [vm.md](vm.md#booted-snapshots)).

## Image format

A shem image is a standard [OCI image](https://github.com/opencontainers/image-spec) for the platform `linux/riscv32`, stored as an [OCI image layout](https://github.com/opencontainers/image-spec/blob/main/image-layout.md), so registries, Docker and containerd handle it like any other image: `docker pull --platform linux/riscv32` pulls what `shem push` pushed. shem's own settings are labels: `dev.shem.machine`, and where they apply `dev.shem.memory` (a warm image's memory), `dev.shem.kernel` and `dev.shem.bootargs` (a VM image's) and `dev.shem.model` (a model image's). A warm image's state is its last layer, whose files are all under `/.shem`.

Layouts in shem's earlier format, with these media types, can still be read:

| Media type | Contents |
|---|---|
| `application/vnd.shem.config.v1+json` | Entrypoint, cmd, env, workdir, labels, the machine (`rv32im`), a warm image's memory size, and a VM image's kernel and command line |
| `application/vnd.oci.image.layer.v1.tar+zstd` | Guest filesystem layers, in the standard format |
| `application/vnd.shem.state.v1.tar+zstd` | A `WARM` snapshot, always the last layer: a container's (`machine.state`, `container.json`) or a VM's (`machine.state`, `vm.json`) |

Layers are tars compressed with zstd, which matters most for state layers: they hold all of a guest's memory, mostly zeros. The booted Linux image is 2 MB, where its RAM alone is 64 MiB. Uncompressed layers, which earlier versions wrote, are still read.

Images in the local store live under `$SHEM_HOME/images/<name>/<tag>`, by default `~/.local/share/shem/images`. Each run unpacks the image's filesystem into a fresh directory and removes it afterwards, like `docker run --rm`.

`shem push` and `shem pull` copy images to and from any OCI registry with [ORAS](https://oras.land), which handles any media types, so registries need nothing special; `--plain-http` reaches a registry without TLS, such as a local one. Pushes go by manifest digest, and an image keeps its digest through the round trip. A test pushes a warm image to a registry, pulls it back, and runs it.

Planned: a `application/vnd.shem.weights.v1+safetensors` layer for machine images that carry model weights.

## Reproducibility

Layers have sorted entries, timestamps of 0 and root ownership, compression is deterministic, and execution is deterministic too: the clock is virtual, the PRNG is seeded and stdin is empty during builds. So building the same Shemfile twice gives the same manifest digest, warm images included. Tests check both. A build cache keyed on those digests, as in BuildKit, is planned.

## Copy-on-write

- **Filesystem.** Today each run gets a full copy of the layers. Planned: overlayfs on the host, as Docker uses it.
- **Guest memory**, from least to most effort:
  1. **Fork by copying.** What the container does today: `fork` clones the vCPU's state. Full copies cost memory, not speed.
  2. **Paged state.** Group state rows into pages, give each container a page table, count references, and copy a page only on the first write to a shared one. This is Linux `fork()` and vLLM's block sharing. A thousand containers resumed from one snapshot then share almost all their memory. Each step reads and writes one row per head, so routing those accesses through a page table fits naturally.
  3. **Diff-only layers.** Every write goes to a known row, so a state layer could store only the rows that changed since its parent.

## Later

- A BuildKit frontend (`# syntax=ghcr.io/shem/shemfile:1`), so `docker buildx build` does the parsing, caching and pushing.
- How much host process state should a state layer carry before it becomes CRIU by another name?
