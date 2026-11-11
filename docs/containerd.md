# containerd

`containerd-shim-shem-v1` ([`crates/containerd-shim-shem`](../crates/containerd-shim-shem/)) is a containerd runtime shim. With it, containerd runs containers on shem: the image is an ordinary OCI image, and its entrypoint, a static RV32 Linux executable, runs on a shem vCPU instead of the host CPU. This is the pattern runwasi uses to run WebAssembly.

```sh
cargo build --release -p containerd-shim-shem
sudo install target/release/containerd-shim-shem-v1 /usr/local/bin/

# an ordinary OCI image holding an RV32 program
just cc -O2 -o hello guests/hello.c
printf 'FROM scratch\nCOPY hello /hello\nENTRYPOINT ["/hello"]\n' | docker build -t shem-hello -f - .
docker save shem-hello | sudo ctr images import -

sudo ctr run --rm --runtime io.containerd.shem.v1 docker.io/library/shem-hello:latest demo one two
```

containerd finds the shim by name on its `PATH`, so installing the binary is the only setup. `just containerd-test` does all of the above and checks the output, and CI runs it on every push.

## How it works

containerd starts the shim and calls its task service over ttrpc. The shim:

1. **create**: mounts the image's root filesystem, which containerd's snapshotter prepared, into the bundle;
2. **start**: reads the bundle's `config.json` (the OCI runtime spec) for the process's arguments, environment and working directory, and the memory limit, which becomes the guest's memory (64 MiB if none is set), then runs the process as a shem container on the reference vCPU, with its standard streams on containerd's FIFOs;
3. **wait**, **state**: report the exit status when the container's init exits;
4. **kill**: stops the guest at the next slice of instructions, with the status a shell reports for the signal;
5. **delete**: unmounts the root filesystem.

The part that turns a bundle into a container ([`bundle.rs`](../crates/containerd-shim-shem/src/bundle.rs)) is plain Rust with its own tests. The protocol around it ([`service.rs`](../crates/containerd-shim-shem/src/service.rs)) is Linux-only, as containerd is.

## Not yet

- Shem images with warm state: containerd unpacks only standard layers, so for now the shim starts the entrypoint cold.
- The neural vCPU, which needs a server; the shim uses the reference vCPU.
- `ctr exec`, pausing, checkpoints and resource statistics.
- A Kubernetes `RuntimeClass` example.
