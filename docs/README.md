# shem

**Run containers and VMs inside a model's weights.**

shem treats the host as a hypervisor and a model as the machine. The model supplies a CPU, compiled into its weights, and RAM, which is its recurrent state. The host supplies everything else: system calls, devices, scheduling and storage. Images are standard OCI images, built from Shemfiles with `shem build` or `docker build`, and they can carry a running guest, a booted kernel or a primed language model session.

## Quick start

```sh
curl -fsSL https://jgalego.github.io/shem/install.sh | sh
shem run ghcr.io/jgalego/shem/hello:latest
```

From a repository checkout, with [uv](https://docs.astral.sh/uv/) installed:

```sh
shem build -t qwen3 examples/qwen3
shem run qwen3 "What is a golem?"
```

Programs use the reference interpreter by default. `--vcpu neural` selects a standalone RV32IM network with compiled weights; it needs a [neural server](neural.md#running-it). Language-model sessions use PyTorch, not the vCPU.

## Guides

| Task | Guide |
|---|---|
| Import a model, save a session or add tools | [Model images](models.md) |
| Build, push and pull images | [Shemfile](shemfile.md) |
| Run with Docker or containerd | [containerd and Docker](containerd.md) |
| Configure nodes or mount model images | [Kubernetes](kubernetes.md) |
| Boot Linux or save a booted VM | [VM mode](vm.md) |
| Run and benchmark the compiled CPU | [Neural vCPU](neural.md) |
| Understand the runtime and its limits | [Architecture](architecture.md) |

The source is at [github.com/JGalego/shem](https://github.com/JGalego/shem).
