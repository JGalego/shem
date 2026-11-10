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

# Build the guest programs (riscv-tests and guests/) and run them; needs Docker
guest-tests: toolchain
    docker build -q --target riscv-tests --output type=local,dest=target/riscv-tests toolchain
    mkdir -p target/guests
    just cc -O2 -o target/guests/hello guests/hello.c
    cargo test --release --test guests -- --ignored
