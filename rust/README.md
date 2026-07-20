# Infera router (Rust data plane)

Multi-core drop-in for the Python router's hot path. Enabled with
`--router-backend rust`.

Supports mixed dispatch, round-robin, etcd discovery, SSE relay, and
SGLang-bootstrap PD. Configs outside this set (kv-aware routing, NATS, other PD
connectors) are served by the Python backend.

## Layout

```
router/src/lib.rs          # library crate (the router)
router/src/main.rs         # binary entry point
router/src/*.rs            # config / discovery / pool / policy / proxy / disagg / ...
router/tests/functional.rs # integration tests
```

## Toolchain

Stable Rust via [rustup](https://rustup.rs) (`clippy` + `rustfmt` included):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

## Build & run

```bash
cargo build --release      # -> target/release/infera-router
cargo run --release -- --host 0.0.0.0 --port 8000 --etcd-endpoint 127.0.0.1:2379
```

Use the same `--etcd-prefix` (default `/infera/workers/`) the workers register
under. The Docker images build the binary and put it on `PATH`, so
`--router-backend rust` works without a manual build.

## Test

```bash
cargo test                                  # unit + integration + doctests
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check                  # `cargo fmt --all` to fix
```

CI runs all three on every PR (`.github/workflows/ci.yml`, job `rust`).

Unit tests live inline (`#[cfg(test)] mod tests`) beside the code; functional
tests spin up mock workers + the real app in `router/tests/functional.rs`. Run
one by name: `cargo test mixed_round_robin`.
