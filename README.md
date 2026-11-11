# shem

**Run containers and VMs inside a model's weights.**

[![CI](https://github.com/JGalego/shem/actions/workflows/ci.yml/badge.svg)](https://github.com/JGalego/shem/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-blue)](Cargo.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange)](docs/architecture.md#roadmap)

shem treats the host as a hypervisor and a model as the machine. The model supplies a CPU, compiled into its weights, and RAM, which is its recurrent state, as in [Qwenputer](https://divagr.com/blog/qwenputer). The host supplies everything else: system calls, devices, scheduling and storage. Images are built from a [Shemfile](docs/shemfile.md) as OCI images, like Docker images, and can carry a running guest, frozen at build time.

The name comes from the golem legend: a clay figure comes to life when a *shem*, a slip of paper with words on it, is placed in its mouth. Here the weights are the clay and the program is the shem.

## What works

Early, but every part of the design exists, from the reference vCPU to a model running guests.

- **Containers** (`shem run`). The host is the guest's kernel: static RV32 Linux programs run with files from a host directory as their root, processes (`fork`, `execve`, `wait`, pipes, signal handlers), a virtual clock, and atomics emulated so the vCPU stays RV32IM. C programs linked against musl and a static BusyBox run, `sh` pipelines and job control included. `--trace` prints each system call.
- **Images** (`shem build`). A [Shemfile](docs/shemfile.md) becomes a reproducible OCI image. `WARM` freezes a running guest into the image at build time, so `shem run` resumes it where the build left off. `shem push` and `shem pull` move images through OCI registries.
- **VM mode** (`shem boot`). The host is the machine around a guest kernel, emulating the privileged architecture, a timer, a UART and a syscon: a no-MMU Linux 6.8 boots to a shell in two seconds ([docs/vm.md](docs/vm.md)).
- **containerd**. `ctr run --runtime io.containerd.shem.v1` runs RV32 programs from ordinary OCI images on shem ([docs/containerd.md](docs/containerd.md)).
- **The neural vCPU**. An RV32IM CPU compiled into network weights, one instruction per forward pass, with memory in delta-rule state. It matches the reference vCPU exactly on the official riscv-tests and random programs, resumes snapshots the reference took, batches every running guest into one forward pass, and shares memory pages between them ([docs/neural.md](docs/neural.md)).
- **The reference vCPU**. An RV32IM interpreter that defines correct behaviour for the neural one, passes the official riscv-tests, and runs about 175 million instructions per second.

See the [roadmap](docs/architecture.md#roadmap) for what is next.

## Quickstart

You need Rust 1.88 or later, [just](https://github.com/casey/just) and [pre-commit](https://pre-commit.com). Building guest programs also needs Docker, and the neural vCPU needs [uv](https://docs.astral.sh/uv/).

```sh
just build                                  # build target/release/shem (put it on PATH as shem)
just test                                   # run the unit and integration tests
just hooks                                  # install the git hooks

just cc -O2 -o target/hello guests/hello.c  # compile a C guest with the toolchain image
just run target/hello one two               # run it as a container
just guest-tests                            # build and run riscv-tests, the C guests and BusyBox
just neural-tests                           # test the neural vCPU, alone and against the reference
just vm-tests                               # boot Linux, run commands in its shell, power off
just containerd-test                        # run a container through containerd (needs sudo)

# after `just guest-tests`, BusyBox and its rootfs are in target/busybox
just run --root target/busybox target/busybox/bin/busybox ls -l /bin
just run --root target/busybox target/busybox/bin/busybox sh -c 'ls /bin | wc -l'

# build images from Shemfiles, and run them
just busybox-image
shem build -t hello examples/hello
shem build -t warm-shell examples/warm-shell
shem run hello
echo 'echo hello from a warm shell' | shem run warm-shell
shem push warm-shell ghcr.io/me/warm-shell:1.0   # needs oras (https://oras.land)

# boot Linux in VM mode
just linux-image
shem boot target/linux/Image

# run a guest on the neural vCPU
just neural-server &
just run --vcpu neural --neural-socket /tmp/shem-neural.sock --mem 1M target/riscv-tests/rv32ui-add
```

Guest programs are static RV32 Linux executables. `just cc` builds them with Zig against musl, for RV32IMA with soft float; see [`toolchain/`](toolchain/).

## Layout

```
crates/shem/src/
  vcpu.rs        the vCPU interface and the canonical snapshot format
  reference.rs   the reference RV32IM interpreter
  emulate.rs     host emulation of instructions the vCPU lacks (atomics)
  elf.rs         ELF loading and the initial process stack
  linux/         Linux system calls, implemented by the host
    mod.rs       dispatch, per-process state, time and identity
    files.rs     file descriptors, files and directories
    rootfs.rs    path resolution inside the guest's root directory
    signal.rs    signal dispositions, masks and delivery to handlers
    abi.rs       system call numbers, errno values, flags and structs
  mm.rs          the guest address space: heap and anonymous mappings
  container.rs   container mode: processes, the scheduler, fork, exec and wait
  neural.rs      the neural vCPU, as the host reaches it over a socket
  image/         images: Shemfile parser, builder, OCI layouts, the local store
  vm.rs          VM mode: the machine around a guest kernel
  fdt.rs         device tree writer
  main.rs        the shem CLI
crates/shem/tests/  unit, integration and guest-program tests
crates/containerd-shim-shem/  the containerd shim
neural/         the neural vCPU: ISA table, weight compiler, network, executor, server
toolchain/      the guest toolchain image and the riscv-tests environment
guests/         C guest programs
examples/       example Shemfiles
docs/
  architecture.md  design, tech stack and roadmap
  neural.md        how the neural vCPU works, and its protocol
  vm.md            VM mode
  containerd.md    the containerd shim
  shemfile.md      the Shemfile and the image format
```

## License

[MIT](LICENSE)
