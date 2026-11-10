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

# Run a guest executable, e.g. `just run hello.elf arg1`
run *args:
    cargo run --release -- run {{ args }}

# Everything the hooks check
check: lint test

# Install the git hooks
hooks:
    pre-commit install --hook-type pre-commit --hook-type pre-push
