# shem

Run containers and VMs inside a model's weights.

shem treats the host as a hypervisor and a model as the machine. The model supplies a CPU, compiled into its weights, and RAM, which is its recurrent state, as in [Qwenputer](https://divagr.com/blog/qwenputer). The host supplies everything else: system calls, devices, scheduling and storage. Images are built from a [Shemfile](docs/shemfile.md) and stored in OCI registries, like Docker images.

The name comes from the golem legend: a clay figure comes to life when a *shem*, a slip of paper with words on it, is placed in its mouth. Here the weights are the clay and the program is the shem.

## Status

Early. What works today is the host side, running on a reference vCPU:

- an RV32IM interpreter that defines correct behaviour for any neural vCPU
- a loader for static RV32 Linux executables, with the standard Linux stack layout
- a minimal Linux system-call layer: `read`, `write`, `writev`, `close`, `brk`, `exit`
- `shem run`, which runs an executable as a container

There is no neural vCPU yet. See the [roadmap](docs/architecture.md#roadmap).

## Quickstart

You need Rust 1.88 or later, [just](https://github.com/casey/just) and [pre-commit](https://pre-commit.com).

```sh
just build              # build target/release/shem
just test               # run the tests
just run hello.elf a b  # run a guest program with arguments
just hooks              # install the git hooks
```

Guest programs must be static, non-PIE RV32 Linux executables built with `-march=rv32im -mabi=ilp32`, linked against a libc that was built the same way.

## Layout

```
crates/shem/src/
  vcpu.rs        the vCPU interface and the canonical snapshot format
  reference.rs   the reference RV32IM interpreter
  elf.rs         ELF loading and the initial process stack
  linux.rs       Linux system calls, implemented by the host
  container.rs   the hypervisor loop for container mode
  main.rs        the shem CLI
docs/
  architecture.md  design, tech stack and roadmap
  shemfile.md      the Shemfile and image format (draft)
```
