# Security

## Reporting a vulnerability

Please report vulnerabilities privately, through GitHub's [Report a vulnerability](https://github.com/JGalego/shem/security/advisories/new) form, not in a public issue. Include what you found, how to reproduce it, and what it affects. You will get a reply within a week, and a fix or a plan for one as soon as the problem is understood.

## What shem trusts

- **Guests are untrusted.** A guest program or kernel runs on a vCPU, so it cannot execute host instructions. Its system calls are serviced by the host, and its paths are resolved inside its root filesystem (`crates/shem/src/linux/rootfs.rs`). Any way for a guest to reach host files outside its root, or to crash or hang the host, is a vulnerability.
- **Images are untrusted input.** Layers are unpacked with whiteouts and paths checked, and blobs are verified against their digests. A layer or config that escapes the unpack directory is a vulnerability.
- **The model server is local.** `shem-neural-server` listens on a Unix socket that only its user (and root, as the containerd shim runs) can open. Anyone who can connect can run models and vCPUs on it.
- **Models run as data.** Model images hold weights in safetensors and a JSON manifest; nothing in them is executed as code.

## Supported versions

Fixes go into the latest release.
