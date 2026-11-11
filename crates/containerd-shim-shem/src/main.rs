fn main() {
    #[cfg(target_os = "linux")]
    containerd_shim::run::<containerd_shim_shem::service::Service>("io.containerd.shem.v1", None);
    #[cfg(not(target_os = "linux"))]
    eprintln!("containerd-shim-shem-v1 runs only on Linux, like containerd");
}
