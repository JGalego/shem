# shem

**Run containers and VMs inside a model's weights.**

[![CI](https://github.com/JGalego/shem/actions/workflows/ci.yml/badge.svg)](https://github.com/JGalego/shem/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-blue)](Cargo.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange)](docs/architecture.md#roadmap)

shem treats the host as a hypervisor and a model as the machine. The model supplies a CPU, compiled into its weights, and RAM, which is its recurrent state, as in [Qwenputer](https://divagr.com/blog/qwenputer). The host supplies everything else: system calls, devices, scheduling and storage. Images are built from a [Shemfile](docs/shemfile.md) and stored in OCI registries, like Docker images.

The name comes from the golem legend: a clay figure comes to life when a *shem*, a slip of paper with words on it, is placed in its mouth. Here the weights are the clay and the program is the shem.

## Status

Early. What works today is the host side, running on a reference vCPU:

- an RV32IM interpreter that defines correct behaviour for any neural vCPU, and passes the official riscv-tests suites
- host emulation of the A extension (atomics), so the vCPU itself stays RV32IM
- a loader for static RV32 Linux executables, with the standard Linux stack layout
- a Linux system-call layer: files and directories in a host directory used as the guest's root, `brk` and anonymous `mmap`, and a virtual clock, enough for C programs linked against musl and for a static BusyBox
- processes: `fork`, `execve`, `wait`, pipes, process groups and signal handlers, with the host as the kernel and a scheduler that gives each process a turn
- `shem run`, which runs an executable as a container (`--trace` prints each system call)

There is no neural vCPU yet. See the [roadmap](docs/architecture.md#roadmap).

## Quickstart

You need Rust 1.88 or later, [just](https://github.com/casey/just) and [pre-commit](https://pre-commit.com). Building guest programs also needs Docker.

```sh
just build                                  # build target/release/shem
just test                                   # run the unit and integration tests
just hooks                                  # install the git hooks

just cc -O2 -o target/hello guests/hello.c  # compile a C guest with the toolchain image
just run target/hello one two               # run it as a container
just guest-tests                            # build and run riscv-tests, the C guests and BusyBox

# after `just guest-tests`, BusyBox and its rootfs are in target/busybox
just run --root target/busybox target/busybox/bin/busybox ls -l /bin
just run --root target/busybox target/busybox/bin/busybox sh -c 'ls /bin | wc -l'
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
  main.rs        the shem CLI
crates/shem/tests/  unit, integration and guest-program tests
toolchain/      the guest toolchain image and the riscv-tests environment
guests/         C guest programs
docs/
  architecture.md  design, tech stack and roadmap
  shemfile.md      the Shemfile and image format (draft)
```

## License

[MIT](LICENSE)
