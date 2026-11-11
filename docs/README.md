# shem

**Run containers and VMs inside a model's weights.**

shem treats the host as a hypervisor and a model as the machine. The model supplies a CPU, compiled into its weights, and RAM, which is its recurrent state. The host supplies everything else: system calls, devices, scheduling and storage. Images are standard OCI images, built from Shemfiles with `shem build` or `docker build`, and they can carry a running guest, a booted kernel or a primed language model session.

- [Architecture](architecture.md): the design, the stack and the roadmap
- [Model images](models.md): language models as images, from Hugging Face
- [The Shemfile](shemfile.md): building images, with `shem build` or `docker build`, and the image format
- [The neural vCPU](neural.md): an RV32IM CPU compiled into network weights
- [VM mode](vm.md): booting Linux, and shipping it booted
- [containerd and Docker](containerd.md): running shem images with `docker run`

The source is at [github.com/JGalego/shem](https://github.com/JGalego/shem).
