# Changelog


## 0.3.0 (2026-11-13)

### Features

- `shem load` stores the images of a `docker save` archive, to run what Docker built where its containerd cannot see the shim, as on Docker Desktop
- `--dtype bfloat16` runs a model session in half the memory, and the model server keeps a model loaded in each type its images ask for
- `docker commit` freezes a running program, and its image goes on where the program paused; a kill ends a program that waits for input
- `docker run -t` gives a program a terminal of its own, with echo, line editing, Ctrl-C and the window's size
- `shem serve` and `docker run --serve` serve a model image as OpenAI's chat completions API, in the container's own network under Docker
- A landing page and an install script on the website, with the docs under /docs/

### Fixes

- An image's layers and session files can no longer delete or write files outside the root they unpack into, by paths from /, paths that climb out, or links that lead out
- A signal typed at the terminal goes to its foreground process group, init's included, and no longer returns from a system call init never made
- Counts a frozen container brings, of instructions and time, wrap or stop at their limits instead of overflowing
- Leave out the default ENTRYPOINT when a model stage sets its own, which made docker build warn about a line of the generated Dockerfile

### Documentation

- Start the model server from a release, as a systemd service or with sudo and full paths, as well as from a checkout
- Add CNCF governance files and a DCO check

### Tests

- Fuzz what shem reads from images, archives for `shem load`, programs, frozen containers, layers and Shemfiles, every day, and try their seeds in CI

### Other

- Relicense from MIT to Apache 2.0

## 0.2.0 (2026-11-12)

### Features

- Llama, Mistral and Qwen2 imports, and model directories that load as Hugging Face checkpoints
- Gemma, Phi-3, OLMo and SmolLM3 imports, and chat templates that see the tokenizer's special tokens
- Run ModelPack artifacts as image twins that share their blobs
- Publish model images as ModelPack artifacts with shem push --modelpack
- Publish sessions as artifacts linked to their models, and resume them with shem run --state
- Keep block files at the top of the model directory, where vLLM looks for weights
- Run ModelPack layout directories as twins, and export uncompressed artifacts with their files at the root
- Mixtures of experts, Granite, Mamba and GGUF imports, docker build on ModelPack artifacts, and model images as Kubernetes image volumes
- Shemfile syntax highlighting and file icon for VS Code
- --system-file adds a file of the image to the system prompt, and --stop ends a reply at a text, as base models need
- Examples for SmolLM2, Granite MoE, Mamba, a Llama GGUF, Gemma, a handbook primed at build time, a ModelPack round trip and an image volume, tested in CI and on Hugging Face Jobs
- Version session state by block type and precision

### Fixes

- Leave out the second copy of the weights that Mistral's repos carry in their own format
- Keep a layout's blob that is already there, as a read-only link oras left, when an image built on a ModelPack twin exports uncompressed
- Commit new formulas to the Homebrew tap and skip unchanged ones

### Documentation

- A blog post on LLMs as containers
- Name ModelPack in the blog post's prior art
- Compare ModelPack and shem in the README
- The large checkpoints compared in bfloat16
- The checkpoints compared on a Hugging Face Job, up to Qwen3-30B-A3B and Mixtral-8x7B, the driver's new options and the examples

### Tests

- Compare same-weight logits within float rounding, as math libraries order a mixture of experts' sums by memory layout
- Compare checkpoints of up to 47 billion parameters with transformers on a Hugging Face Job with 256 GB, with just models-job

## 0.1.0 (2026-11-11)

### Features

- Reference RV32IM vCPU and container runtime
- Run musl C programs and pass the official riscv-tests
- Run BusyBox with a filesystem, a virtual clock and more system calls
- Processes with fork, exec, wait, pipes and signals
- Neural vCPU compiled into network weights
- Shemfile builds, OCI layouts and WARM
- Paged memory with copy-on-write and continuous batching
- Boot a no-MMU Linux kernel in VM mode
- Containerd shim to run RV32 containers through ctr
- Shem push and shem pull
- Neural CPU on GPUs with flat memory and compiled steps
- Booted VM snapshots and VM images
- VM mode on the neural vCPU
- Compress image layers with zstd
- File mappings and file metadata on the virtual clock
- /proc, sysinfo and statfs
- Docker run on shem
- Run containers inside the model with docker run, and shem save for docker load
- Model images that build, run, push and pull language models
- Model images with primed sessions frozen into state layers
- Shem images are standard OCI images that docker pulls and loads
- Run model images in containers on the model server
- A BuildKit frontend that lets docker build Shemfiles
- Publish the compiled CPU's weights with a model card on Hugging Face
- A Kubernetes RuntimeClass for shem, tested on kind
- Live model sessions under docker run -i, docker exec and docker commit, and Kubernetes sandboxes
- Programs as compiled layers, run exactly when a model calls them as tools
- Shared writable file mappings, host terminals, and ppoll
- List images under node platforms for Kubernetes on released containerd
- Threads on any vCPU, futex, and a socket that appears only when it listens

### Fixes

- Open the shim's stdin FIFO without waiting for a writer
- Report a failed start on the container's stderr
- Give docker a network namespace to configure, and end tasks killed before they start
- Multi-turn sessions on every chat template, model sessions on GPUs, and the kubelet feature gate for riscv32 images

### Performance

- Integer shifter and register rows by gather and scatter

### Documentation

- What works by mode and what is next
- File mappings and deterministic file metadata
- Getting started without just and with cleanup
- Link the software shem is built with
- Move the built-with links to their own section
- Set the name's origin apart from the intro
- Development guide in the README
- Getting started runs a container inside the model with docker
- One Contribute section, with the install note under Release

### Tests

- Report ctr's status and containerd's log on failure
- Give ctr run the program as well as its arguments
- Report Docker's state and logs when docker run fails
- Trace the shim's requests and time out the docker test's diagnostics

### CI

- Add CI workflow, README badges and tagline
- Pin setup-uv to v10.3.3
- Use actions/cache v6
- Release pipeline for crates.io, PyPI, binaries and the toolchain image
- Publish the runtime image and the example images on each release
- Cargo deny for licences and advisories, and Dependabot updates
- Cargo binstall metadata, a Homebrew formula and a docs site

### Other

- Add MIT license
- Format the shim's model test and the neural project lookup
- Let the model card keep one line per paragraph
- Licence notices for model images, SECURITY.md, a changelog, and a private model server socket
