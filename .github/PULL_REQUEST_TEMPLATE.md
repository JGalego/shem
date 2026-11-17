<!-- For a change to the design, a new Shemfile instruction or a new device, open an issue first. -->

## What this changes, and why

## How it was checked

<!-- The tests you added or ran: `cargo test --workspace`, `uv run --project neural pytest neural/tests`, a `just` suite. If CI cannot see it, say how you saw it. -->

## Checklist

- [ ] Tests for the change, and the docs it touches updated (one line per paragraph, no hard wraps)
- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and for Python `ruff` and `scripts/mypy-ratchet`
- [ ] One-line commit messages, `<type>: <message>`, each signed off (`git commit -s`)
