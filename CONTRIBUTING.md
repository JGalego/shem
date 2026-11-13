# Contributing to shem

Thanks for helping. shem is experimental, so small, focused changes with tests are the easiest to review.

## Before you start

- For a bug or a small fix, open a pull request directly.
- For a change to the design, a new Shemfile instruction or a new device, open an issue first so we can agree on the approach.
- Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).
- Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Build and test

Use Rust 1.88+, [uv](https://docs.astral.sh/uv/), Go for the BuildKit frontend, and Docker for guest builds. The [toolchain image](toolchain/Dockerfile) cross-compiles RV32 programs with Zig and musl.

Install [pre-commit](https://pre-commit.com) hooks:

```sh
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Run the checks for the component you changed:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uv run --project neural pytest neural/tests
(cd frontend && go vet ./... && go test ./...)
cargo deny check                              # requires cargo-deny
```

The hooks also lint Dockerfiles and Shemfiles with [hadolint](https://github.com/hadolint/hadolint), in its Docker image, through [`scripts/hadolint`](scripts/hadolint), which turns shem's own instructions into comments first. Run it alone with `pre-commit run hadolint --all-files`.

Integration suites need built guest programs, servers or external tools. The [justfile](justfile) supplies recipes; [CI](.github/workflows/ci.yml) shows the full setup.

| Suite | Command | Requires |
|---|---|---|
| RV32 guests, BusyBox, riscv-tests | `just guest-tests` | Docker |
| Neural vCPU differential tests | `just neural-tests` | Docker, uv |
| Linux boot and snapshots | `just vm-tests` | Prebuilt kernel downloaded by the recipe |
| containerd / Docker | `just containerd-test` / `just docker-test` | Docker, containerd, sudo |
| Kubernetes | `just kubernetes-test` | Docker, kind, kubectl, static Rust target |
| Registry and ModelPack | `just registry-test` | Docker, ORAS, uv |

Once a suite's inputs exist, run its ignored tests with `cargo test --release --test <suite> -- --ignored`.

## Extend shem

| To add | Change | Test in |
|---|---|---|
| A system call | Number in [`linux/abi.rs`](crates/shem/src/linux/abi.rs), dispatch in `Linux::handle` in [`linux/mod.rs`](crates/shem/src/linux/mod.rs), implementation on `Process` | [`tests/linux.rs`](crates/shem/tests/linux.rs) |
| A Shemfile instruction | [`image/shemfile.rs`](crates/shem/src/image/shemfile.rs), [`image/build.rs`](crates/shem/src/image/build.rs), then [the reference](docs/shemfile.md) | [`tests/image.rs`](crates/shem/tests/image.rs) |
| A VM device | `device_read` and `device_write` in [`vm.rs`](crates/shem/src/vm.rs), plus its device-tree node | [`tests/vm.rs`](crates/shem/tests/vm.rs) |
| An instruction | [`reference.rs`](crates/shem/src/reference.rs), [`isa.py`](neural/src/shem_neural/isa.py), [`cpu.py`](neural/src/shem_neural/cpu.py) | [`tests/reference.rs`](crates/shem/tests/reference.rs), [`neural/tests`](neural/tests/), [`tests/neural.rs`](crates/shem/tests/neural.rs) |
| A vCPU | `Vcpu` in [`vcpu.rs`](crates/shem/src/vcpu.rs), with canonical snapshots | [`tests/neural.rs`](crates/shem/tests/neural.rs) |

## Ground rules

- The neural vCPU must match the reference interpreter instruction for instruction. Dense and fused executors must agree.
- Guest time, randomness, inode numbers and directory order must not depend on the host. Use the virtual clock and seeded PRNG.
- Keep each documentation paragraph or list item on one line; do not hard-wrap prose.

## Pull requests

1. Fork the repository and make your change on a branch, with its tests and docs.
2. Write one-line commit messages in the form `<type>: <message>`, with a type such as `feat`, `fix`, `docs`, `test`, `ci`, `perf` or `refactor`.
3. Sign off every commit (see below).
4. Open a pull request once CI is green.

## Sign your commits

shem uses the [Developer Certificate of Origin](https://developercertificate.org) (DCO). By signing off, you certify that you wrote the change or have the right to submit it under the project's [Apache 2.0 license](LICENSE). Add the sign-off with `-s`:

```sh
git commit -s -m "fix: handle a short read in pipe2"
```

This appends a line to the commit message, using your git name and email:

```
Signed-off-by: Your Name <you@example.com>
```

A check on pull requests fails when a commit lacks it. To fix a missing sign-off, run `git rebase --signoff main` and force-push your branch.

## Release

The [release workflow](.github/workflows/release.yml) publishes crates, the Python package, Linux and macOS binaries, GHCR images, [CPU weights](https://huggingface.co/jgalego/shem-rv32im), release notes, SBOMs and build provenance.

Set the same version in [Cargo.toml](Cargo.toml) and [neural/pyproject.toml](neural/pyproject.toml), then generate the changelog and tag the release. Replace `0.2.0` with the new version:

```sh
git cliff --tag v0.2.0 -o CHANGELOG.md
git add Cargo.toml Cargo.lock neural/pyproject.toml neural/uv.lock CHANGELOG.md
git commit -s -m "chore: release 0.2.0"
git tag v0.2.0
git push origin v0.2.0
```

Publishing setup:

| Service | Configuration |
|---|---|
| crates.io | `CARGO_REGISTRY_TOKEN` for the first publish; then trusted publishing for both crates with repository `JGalego/shem`, workflow `release.yml`, environment `release`. Delete the token afterwards. |
| PyPI | Pending trusted publisher for `shem-neural`: repository `JGalego/shem`, workflow `release.yml`, environment `pypi`. |
| GitHub | Environments `release` and `pypi`; optional required reviewers. |
| Hugging Face | Environment `huggingface` with an `HF_TOKEN` that can write to `jgalego/shem-rv32im`; override the destination with `HF_REPO`. |
| Homebrew (optional) | Repository `JGalego/homebrew-shem` and `HOMEBREW_TAP_TOKEN` with push access. Otherwise the formula is attached to the release only. |
| GitHub Pages | Deploy from GitHub Actions; repository variable `DOCS_SITE=true`. |
| GHCR | Make `shem-toolchain`, `shem-neural`, `shem/dockerfile` and example images under `shem/` public after their first release. |
