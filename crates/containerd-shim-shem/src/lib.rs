//! A containerd shim for shem: `ctr run --runtime io.containerd.shem.v1 IMAGE ID`
//! runs the image's entrypoint, a static RV32 Linux executable, on a shem
//! vCPU instead of the host CPU, as runwasi does for WebAssembly.
//!
//! [`bundle`] turns an OCI runtime bundle into a running container, and is
//! plain Rust; the shim protocol around it is Linux-only, like containerd.

pub mod bundle;
#[cfg(target_os = "linux")]
pub mod service;
