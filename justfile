set positional-arguments

toolchain_image := "shem-toolchain"

# List recipes
default:
    @just --list

# Build the shem binary
build:
    cargo build --release

# Run all tests
test:
    cargo test --workspace

# Check formatting and lints
lint:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings

# Format the code
fmt:
    cargo fmt --all

# Run a guest executable, e.g. `just run hello arg1`
[no-cd]
run *args: build
    "{{ justfile_directory() }}/target/release/shem" run "$@"

# Everything the hooks check
check: lint test

# Install the git hooks
hooks:
    pre-commit install --hook-type pre-commit --hook-type pre-push

# Build the guest toolchain image
toolchain:
    docker build -q --target toolchain -t {{ toolchain_image }} toolchain

# Compile C into a static shem guest, e.g. `just cc -O2 -o hello hello.c`
cc *args: toolchain
    mkdir -p "{{ justfile_directory() }}/target/zig-cache"
    docker run --rm -u "$(id -u):$(id -g)" \
        -v "{{ invocation_directory() }}:/src" \
        -v "{{ justfile_directory() }}/target/zig-cache:/zig-cache" \
        -e ZIG_GLOBAL_CACHE_DIR=/zig-cache \
        {{ toolchain_image }} rv32-cc "$@"

# Build the guest programs (riscv-tests, guests/, BusyBox, examples/) and run them; needs Docker
guest-tests: toolchain busybox-image
    docker build -q --target riscv-tests --output type=local,dest=target/riscv-tests toolchain
    docker build -q --target busybox --output type=local,dest=target/busybox toolchain
    mkdir -p target/guests
    just cc -O2 -o target/guests/hello guests/hello.c
    cargo test --release --test guests -- --ignored

# Test the neural vCPU: its own unit tests, then against the reference vCPU; needs Docker and uv
neural-tests: toolchain
    docker build -q --target riscv-tests --output type=local,dest=target/riscv-tests toolchain
    uv run --project neural pytest -q neural/tests
    cargo test --release --test neural -- --ignored

# Serve neural vCPUs on a Unix socket, for `shem run --vcpu neural`
neural-server socket="/tmp/shem-neural.sock":
    uv run --project neural python -m shem_neural.server --socket {{ socket }}

# Build the BusyBox rootfs as a Docker image, for Shemfiles to COPY --from
busybox-image:
    docker build -q --target busybox -t shem-busybox toolchain

# Download a prebuilt no-MMU Linux for VM mode (from cnlohr/mini-rv32ima-images, GPL)
linux-image:
    #!/bin/sh
    set -e
    [ -f target/linux/Image ] && exit 0
    mkdir -p target/linux
    commit=7da8c6e486b7d510426b5c2283cd338abcf04eb7
    curl -fsSL "https://raw.githubusercontent.com/cnlohr/mini-rv32ima-images/$commit/images/linux-6.8-rc1-rv32nommu-cnl-1.zip" -o target/linux/linux.zip
    echo "f6fa72e17ee810dfa42daec6d3836e427cadac3693069daa9a52d00055e4c056  target/linux/linux.zip" | sha256sum -c -
    python3 -m zipfile -e target/linux/linux.zip target/linux

# Boot Linux in VM mode, run commands in its shell, and power it off
vm-tests: linux-image
    cargo test --release --test vm -- --ignored

# Run an RV32 container through containerd with the shem shim; needs Docker, containerd and sudo
containerd-test: toolchain
    cargo build --release -p containerd-shim-shem
    sudo install -m 0755 target/release/containerd-shim-shem-v1 /usr/local/bin/
    mkdir -p target/oci-hello
    just cc -O2 -o target/oci-hello/hello guests/hello.c
    printf 'FROM scratch\nCOPY hello /hello\nENTRYPOINT ["/hello"]\n' | docker build -q -t shem-hello:latest -f - target/oci-hello
    docker save shem-hello:latest | sudo ctr images import -
    timeout 120 sudo ctr run --rm --runtime io.containerd.shem.v1 docker.io/library/shem-hello:latest shem-test one two | tee target/oci-hello/out.txt
    grep -q "sum of squares below 100000" target/oci-hello/out.txt
