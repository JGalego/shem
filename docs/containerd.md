# containerd

`containerd-shim-shem-v1` ([`crates/containerd-shim-shem`](../crates/containerd-shim-shem/)) runs static RV32 Linux executables from OCI images on a shem vCPU. Docker and Kubernetes reach it through containerd. Model images use the same shim with a host-side session driver ([models.md](models.md#sessions-under-docker)).

## With Docker

![A Dockerfile imports a Hugging Face model into an OCI image. docker push and pull exchange it with a registry; docker run starts a model session through the shem containerd shim.](assets/docker-workflow.svg)

```sh
cargo build --release -p containerd-shim-shem
sudo install target/release/containerd-shim-shem-v1 /usr/local/bin/

docker build --platform linux/riscv32 -t shem-hello examples/hello
docker run --rm --runtime io.containerd.shem.v1 --platform linux/riscv32 shem-hello one two
```

Docker 23+ accepts the shim name directly as `--runtime`; no daemon configuration is needed. The hello program exits with its argument count: 3 here.

To use the neural vCPU, start the [neural server](neural.md#running-it) and add:

```sh
docker run --rm --runtime io.containerd.shem.v1 --platform linux/riscv32 \
  --annotation dev.shem.vcpu=neural \
  --annotation dev.shem.neural-socket=/tmp/shem-neural.sock shem-hello one two
```

With `-t`, a program gets a terminal of its own, which echoes what is typed and lets the program set it up, as a shell does to edit lines. `Ctrl-C`, `Ctrl-\` and `Ctrl-Z` send their signals to the program's foreground processes, init included, and the window takes the size of the terminal `docker run -it` runs in.

`docker commit` freezes a running program into the image it makes, and a container of that image goes on where the program was, its memory and registers as they were. With an image of [`tally.c`](../guests/tally.c), which prints a running total of the numbers it reads:

```console
$ docker run -i --name tally --runtime io.containerd.shem.v1 --platform linux/riscv32 tally
1
total 1
2
total 3
```

In another terminal, while it waits for more:

```console
$ docker commit tally tally:at-3
$ echo 4 | docker run -i --rm --runtime io.containerd.shem.v1 --platform linux/riscv32 tally:at-3
total 7
```

A program can be frozen when it is a single process with only its standard streams open; otherwise the commit holds its files alone.

To load an image built with `shem build`, including its `WARM` snapshot:

```sh
shem save IMAGE | docker load
```

Keep both `--runtime io.containerd.shem.v1` and `--platform linux/riscv32`. Without the runtime, Docker tries the host CPU and fails with `exec format error`.

### Docker Desktop

Docker Desktop, on macOS, Windows and Linux, runs containerd in a VM of its own, which cannot see a shim installed on the host. Build images with Docker there, and run them with shem:

```sh
docker build --platform linux/riscv32 -t shem-hello examples/hello
docker save shem-hello | shem load
shem run shem-hello one two
```

`shem load` stores each image of the archive under the name `docker save` gave it, or under `--tag`, and refuses images for other platforms. Model images run the same way: `shem run` starts their session driver itself, without the model server.

## With containerd

```sh
docker build --platform linux/riscv32 -t shem-hello examples/hello
docker save shem-hello | sudo ctr images import -

sudo ctr run --rm --runtime io.containerd.shem.v1 \
  docker.io/library/shem-hello:latest demo /bin/hello one two
```

Install the shim as above, on containerd's `PATH`. The command after the container ID (`demo`) replaces the image entrypoint, so include `/bin/hello`.

## How it works

containerd calls the shim's task service over ttrpc:

| Request | Action |
|---|---|
| `create` | Mount the root filesystem prepared by the snapshotter |
| `start` | Read arguments, environment, working directory and memory limit from `config.json`; run with containerd's FIFOs as stdio. Default guest memory: 64 MiB. |
| `wait`, `state` | Report init's exit status; publish lifecycle events so Docker observes it |
| `kill` | Stop at the next instruction slice, or in a wait for input, with the signal's shell exit status |
| `resize_pty` | Set the window size of a program's terminal, and send the program SIGWINCH |
| `pause`, `resume` | Freeze a program into `/.shem`, where `docker commit` takes it, until it goes on; save a model session there |
| `delete` | Unmount the root filesystem |

Implementation: [`bundle.rs`](../crates/containerd-shim-shem/src/bundle.rs) loads bundles; [`service.rs`](../crates/containerd-shim-shem/src/service.rs) handles the protocol. Run `just docker-test` or `just containerd-test` for integration tests.

## Not yet

- Program containers: `docker exec`.
- Checkpoints and resource statistics.

Model containers already support `exec`, `commit` and terminal input; see [sessions under Docker](models.md#sessions-under-docker).
