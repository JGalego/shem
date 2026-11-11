# shem

**Run containers and VMs inside a model's weights.**

[![CI](https://github.com/JGalego/shem/actions/workflows/ci.yml/badge.svg)](https://github.com/JGalego/shem/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-blue)](Cargo.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange)](docs/architecture.md#roadmap)

shem treats the host as a hypervisor and a model as the machine. The model supplies a CPU, compiled into its weights, and RAM, which is its recurrent state, as in [Qwenputer](https://divagr.com/blog/qwenputer). The host supplies everything else: system calls, devices, scheduling and storage. Images are built from a [Shemfile](docs/shemfile.md) as OCI images, like Docker images, and can carry a running guest, frozen at build time.

> 📜 **Why shem?** The name comes from the golem legend: a clay figure comes to life when a *shem*, a slip of paper with words on it, is placed in its mouth. Here the weights are the clay and the program is the shem.

## What works

Early, but every part of the design exists, from the reference vCPU to a model running guests.

- **Containers** (`shem run`). The host is the guest's kernel: static RV32 Linux programs run with files from a host directory as their root, processes (`fork`, `execve`, `wait`, pipes, signal handlers), a virtual clock, and atomics emulated so the vCPU stays RV32IM. C programs linked against musl and a static [🧰 BusyBox](https://busybox.net) run, `sh` pipelines and job control included, and so do `ps`, `df` and `free`, on a `/proc` the host makes up. `--trace` prints each system call.
- **Images** (`shem build`). A [Shemfile](docs/shemfile.md) becomes a reproducible OCI image. `WARM` freezes a running guest into the image at build time, so `shem run` resumes it where the build left off. `shem push` and `shem pull` move images through OCI registries.
- **VM mode** (`shem boot`). The host is the machine around a guest kernel, emulating the privileged architecture, a timer, a UART and a syscon: a no-MMU [🐧 Linux](https://kernel.org) 6.8 boots to a shell in two seconds. A VM image can ship Linux already booted, so `shem run` starts at its prompt, on either vCPU ([docs/vm.md](docs/vm.md)).
- **[🐳 Docker](https://www.docker.com) and [📦 containerd](https://containerd.io)**. `docker run --runtime io.containerd.shem.v1` and `ctr run` run RV32 programs from ordinary OCI images, built with `docker build --platform linux/riscv32`, on shem ([docs/containerd.md](docs/containerd.md)).
- **The neural vCPU**. An RV32IM CPU compiled into network weights and run with [🔥 PyTorch](https://pytorch.org), one instruction per forward pass, with memory in delta-rule state. It matches the reference vCPU exactly on the official riscv-tests, random programs and a Linux kernel running a shell command, resumes snapshots the reference took, batches every running guest into one forward pass, and shares memory pages between them. On GPUs it is bit-exact against the CPU and runs up to 7.4 million instructions per second across 32,768 guests on an A100, measured with [🤗 Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) ([docs/neural.md](docs/neural.md)).
- **The reference vCPU**. An RV32IM interpreter in [🦀 Rust](https://www.rust-lang.org) that defines correct behaviour for the neural one, passes the official riscv-tests, and runs about 175 million instructions per second.

See the [roadmap](docs/architecture.md#roadmap) for what is next.

## Getting started

You need [🦀 Rust](https://www.rust-lang.org) 1.88 or later and [🐳 Docker](https://www.docker.com). The neural vCPU also needs [⚡ uv](https://docs.astral.sh/uv/). Pushing images needs [🗄️ ORAS](https://oras.land). Guest programs are static RV32 Linux executables built with [🦎 Zig](https://ziglang.org) against musl in a [toolchain image](toolchain/).

Build shem and put it on your `PATH`.

```sh
cargo build --release
export PATH="$PWD/target/release:$PATH"
```

Build the guest toolchain image.

```sh
docker build --target toolchain -t shem-toolchain toolchain
```

Compile a C program for RV32.

```sh
mkdir -p demo
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -e ZIG_GLOBAL_CACHE_DIR=/tmp/zig shem-toolchain rv32-cc -O2 -o demo/hello guests/hello.c
```

Run it as a container.

```sh
shem run demo/hello one two
```

Build a BusyBox root filesystem.

```sh
docker build --target busybox --output type=local,dest=demo/busybox toolchain
```

Run a BusyBox shell in it.

```sh
shem run --root demo/busybox demo/busybox/bin/busybox sh -c 'ls /bin | wc -l'
```

Build the BusyBox image that Shemfiles copy from.

```sh
docker build --target busybox -t shem-busybox toolchain
```

Build two images from their Shemfiles.

```sh
shem build -t hello examples/hello
shem build -t warm-shell examples/warm-shell
```

List the images.

```sh
shem images
```

Run the hello image.

```sh
shem run hello
```

Resume the shell that the build froze.

```sh
echo 'echo hello from a warm shell' | shem run warm-shell
```

Push an image to a registry.

```sh
shem push warm-shell ghcr.io/me/warm-shell:1.0
```

Pull it back.

```sh
shem pull ghcr.io/me/warm-shell:1.0
```

Install the containerd shim for Docker.

```sh
cargo build --release -p containerd-shim-shem
sudo install target/release/containerd-shim-shem-v1 /usr/local/bin/
```

Package the C program as an RV32 Docker image.

```sh
printf 'FROM scratch\nCOPY hello /hello\nENTRYPOINT ["/hello"]\n' | docker build --platform linux/riscv32 -t shem-hello -f - demo
```

Run it with Docker on shem.

```sh
docker run --rm --runtime io.containerd.shem.v1 --platform linux/riscv32 shem-hello one two
```

Download a prebuilt no-MMU Linux.

```sh
curl -fsSL --create-dirs -o demo/linux.zip https://raw.githubusercontent.com/cnlohr/mini-rv32ima-images/7da8c6e486b7d510426b5c2283cd338abcf04eb7/images/linux-6.8-rc1-rv32nommu-cnl-1.zip
```

Unpack the kernel.

```sh
python3 -m zipfile -e demo/linux.zip demo/linux
```

Boot it to a shell and type `poweroff` to leave.

```sh
shem boot demo/linux/Image
```

Build Linux into an image that has already booted.

```sh
shem build -t linux examples/linux
```

Run a command at its prompt.

```sh
echo 'uname -a; poweroff' | shem run linux
```

Start the neural vCPU server in the background.

```sh
uv run --project neural python -m shem_neural.server --socket /tmp/shem-neural.sock &
```

Run the C program on the neural vCPU.

```sh
shem run --vcpu neural --neural-socket /tmp/shem-neural.sock --mem 1M demo/hello one two
```

Run the tests.

```sh
cargo test --workspace
```

### Cleaning up

Stop the neural vCPU server.

```sh
kill %1
```

Remove the shem images.

```sh
rm -rf ~/.local/share/shem
```

Remove the Docker images.

```sh
docker rmi shem-hello shem-busybox shem-toolchain
```

Uninstall the containerd shim.

```sh
sudo rm /usr/local/bin/containerd-shim-shem-v1
```

Remove the demo files.

```sh
rm -rf demo
```

Remove the build outputs.

```sh
cargo clean
```

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
  vm.rs          VM mode: the machine around a guest kernel, and its snapshots
  fdt.rs         device tree writer
  main.rs        the shem CLI
crates/shem/tests/  unit, integration and guest-program tests
crates/containerd-shim-shem/  the containerd shim
neural/         the neural vCPU: ISA table, weight compiler, network, executors, server
  jobs/gpu.py    GPU differential test and benchmark, for Hugging Face Jobs
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

## Development

### Stack

- [🦀 Rust](https://www.rust-lang.org) for the host: the reference vCPU, the Linux system calls, images, VM mode, the CLI and the containerd shim ([`crates/`](crates/)).
- [🐍 Python](https://www.python.org) and [🔥 PyTorch](https://pytorch.org) for the neural vCPU: the weight compiler, the network and its server ([`neural/`](neural/)).
- [⚡ uv](https://docs.astral.sh/uv/) for the Python environment and the GPU jobs.
- [🐳 Docker](https://www.docker.com) for the guest toolchain image and the ordinary stages of a Shemfile ([`toolchain/`](toolchain/)).
- [🦎 Zig](https://ziglang.org) to compile guest programs for RV32 against musl.
- [📦 containerd](https://containerd.io) to run shem containers under Docker and containerd.

### Set up

Install the git hooks with [pre-commit](https://pre-commit.com). They format and lint on every commit and run the tests before every push.

```sh
pre-commit install --hook-type pre-commit --hook-type pre-push
```

### Check your changes

Format the Rust code.

```sh
cargo fmt --all
```

Lint it.

```sh
cargo clippy --workspace --all-targets -- -D warnings
```

Run the Rust tests.

```sh
cargo test --workspace
```

Run the neural vCPU's tests.

```sh
uv run --project neural pytest neural/tests
```

The slower suites are ignored tests that need inputs built first: guest programs, BusyBox, the riscv-tests, a Linux kernel, a neural vCPU server, a registry or containerd. CI builds the inputs and runs every suite on each push, and [`.github/workflows/ci.yml`](.github/workflows/ci.yml) shows how. Run one locally with `cargo test --release --test <suite> -- --ignored` once its inputs exist.

### Extend shem

| To add | Change | Test in |
|---|---|---|
| A system call | Its number in [`linux/abi.rs`](crates/shem/src/linux/abi.rs), its dispatch in `Linux::handle` in [`linux/mod.rs`](crates/shem/src/linux/mod.rs), and its implementation on `Process` | [`tests/linux.rs`](crates/shem/tests/linux.rs) |
| A Shemfile instruction | The parser in [`image/shemfile.rs`](crates/shem/src/image/shemfile.rs) and the builder in [`image/build.rs`](crates/shem/src/image/build.rs), then [`docs/shemfile.md`](docs/shemfile.md) | [`tests/image.rs`](crates/shem/tests/image.rs) |
| A device in VM mode | `device_read` and `device_write` in [`vm.rs`](crates/shem/src/vm.rs), and its node in the device tree | [`tests/vm.rs`](crates/shem/tests/vm.rs) |
| An instruction | The reference vCPU in [`reference.rs`](crates/shem/src/reference.rs), the ISA table in [`isa.py`](neural/src/shem_neural/isa.py), and the network in [`cpu.py`](neural/src/shem_neural/cpu.py) | [`tests/reference.rs`](crates/shem/tests/reference.rs), [`neural/tests`](neural/tests/) and [`tests/neural.rs`](crates/shem/tests/neural.rs) |
| A vCPU | The `Vcpu` trait in [`vcpu.rs`](crates/shem/src/vcpu.rs), with snapshots in the canonical format | The differential tests in [`tests/neural.rs`](crates/shem/tests/neural.rs) show how |

[docs/architecture.md](docs/architecture.md) explains the design, and its [roadmap](docs/architecture.md#roadmap) lists what is next.

### Ground rules

- **The reference vCPU defines correct.** The neural vCPU must match it instruction for instruction, and the dense network and the fused executor must match each other. Tests check both.
- **Runs are deterministic.** Time comes from the virtual clock and randomness from the seeded PRNG. Nothing the guest sees may depend on the host, such as real time, inode numbers or directory order.
- **Docs have no hard wraps.** Each paragraph or list item is one line.
- **Commit messages are one line**, in the form `<type>: <message>`, with a type such as `feat`, `fix`, `docs`, `test`, `ci`, `perf` or `refactor`.

### Contribute

Fork the repository, make your change on a branch with its tests and docs, and open a pull request once CI is green. For a change to the design, open an issue first. Contributions are under the [MIT license](LICENSE).

## License

[MIT](LICENSE)
