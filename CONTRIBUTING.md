# Contributing to shem

Thanks for helping. shem is experimental, so small, focused changes with tests are the easiest to review.

## Before you start

- For a bug or a small fix, open a pull request directly.
- For a change to the design, a new Shemfile instruction or a new device, open an issue first so we can agree on the approach.
- Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).
- Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Build and test

The [Development section of the README](README.md#development) has the full guide: the stack, the git hooks, how to run each test suite and where to add a system call, an instruction or a device. In short:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

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
