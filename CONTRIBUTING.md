# Contributing to Infera

Thanks for your interest in contributing. Infera welcomes bug reports, features,
performance work, documentation, and reviews.

## Reporting issues

Use [GitHub Issues](../../issues) to report bugs or request features. Include a
clear description, reproduction steps, and your environment (OS, ROCm version,
Python version, engine).

For security issues, do **not** open a public issue — see [SECURITY.md](SECURITY.md).

## Development setup

Fork and clone, then install Infera with the dev tools and at least one engine,
and enable the pre-commit hooks:

```bash
pip install -e ".[dev,sglang]"     # or .[dev,vllm] / .[dev,atom]
pre-commit install                 # runs the formatters/linters on every commit
```

The optional Rust router builds with a [Rust toolchain](https://rustup.rs):
`cd rust && cargo build --release`.

## Code style, linting, and tests

- **Formatting & lint** are enforced by [pre-commit](.pre-commit-config.yaml):
  `ruff` (format + lint) for Python and `cargo fmt` for Rust. Run the whole set
  with `pre-commit run --all-files`.
- **License header** — new source files carry the MIT SPDX header:
  ```
  # SPDX-License-Identifier: MIT
  # Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
  ```
- **Tests** — add or update tests for behavior changes:
  - Unit / CPU: `pytest -m "not slow and not integration"`
  - Engine + end-to-end (GPU): `tests/run_tests.sh` (see `tests/`)

  CI runs lint, unit, the Rust suite, and the GPU e2e matrix; a docs/examples-only
  change skips the code jobs automatically.

## Pull request workflow

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/short-description
   ```
2. Make your change. Add tests and update docs if behavior changes.
3. Write clear commit messages — conventional prefixes are preferred
   (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`) — and **sign off every
   commit** (see [DCO](#developer-certificate-of-origin-dco) below):
   ```bash
   git commit -s -m "fix: ..."
   ```
4. Open a PR against `main`. Describe *what* changed and *why*; link any related
   issue. Ensure CI passes and request review from the relevant
   [CODEOWNERS](.github/CODEOWNERS).

## Developer Certificate of Origin (DCO)

Infera requires every commit to be **signed off** under the
[Developer Certificate of Origin](https://developercertificate.org/) (DCO). By
adding a `Signed-off-by` line you certify that you wrote the patch — or otherwise
have the right to submit it — and agree to contribute it under the project's
[MIT license](LICENSE). The DCO is a lightweight certification; it is **not** a
copyright assignment.

Add the sign-off with the `-s` flag — it appends a trailer built from your git
name and email:

```bash
git commit -s -m "feat: ..."
```

```
Signed-off-by: Your Name <you@amd.com>
```

Make it automatic so you don't forget:

```bash
git config --global format.signoff true
```

Forgot to sign off? Add it to the commits already on your branch:

```bash
git rebase --signoff main       # all commits on the branch
git commit --amend -s           # just the most recent commit
```

A CI check (`.github/workflows/dco.yml`) verifies that **every** commit in a PR
carries a `Signed-off-by` matching its author. The sign-off must use a real
identity: an auto-generated machine hostname (e.g. `you@buildhost.internal`) is
rejected — set a proper email with `git config --global user.email you@amd.com`
(and `git config --global user.useConfigOnly true` so git never invents one).

## External contributors

This repo is part of the AMD-AGI org. Non-AMD contributors need admin approval
before being added as collaborators.
