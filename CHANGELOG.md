# Changelog


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

### Fixes

- Open the shim's stdin FIFO without waiting for a writer
- Report a failed start on the container's stderr
- Give docker a network namespace to configure, and end tasks killed before they start

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
