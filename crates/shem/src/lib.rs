//! shem runs containers and VMs inside a model's weights.
//!
//! The host acts as the hypervisor and kernel; a vCPU, either the reference
//! interpreter or a model with a CPU compiled into its weights, runs the guest
//! and exits to the host for everything else. See `docs/architecture.md`.

pub mod container;
pub mod elf;
pub mod emulate;
pub mod linux;
pub mod mm;
pub mod reference;
pub mod vcpu;
