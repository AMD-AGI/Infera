#!/usr/bin/env bash
# Build + test the Infera #222 head on a compute node, the way release.yml (#223) does:
# unit tests, pure wheel + sdist (python -m build), infera-router in manylinux_2_28
# (build_router.sh), Linux x86_64 wheel (bundle_router_wheel.sh), install check.
#   sbatch build_infera.sh <infera-checkout> <out-dir>
#SBATCH --job-name=llying-infera-build
#SBATCH --partition=Compute-DCPT
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/apps/liyingli/jobs/%x-%j.out
set -euo pipefail
src="${1:?checkout}"; out="${2:?out dir}"
export PATH="$HOME/.local/bin:$PATH"
echo "node $(hostname) $(date -u +%FT%TZ) head $(git -C "$src" rev-parse --short HEAD)"
work="$(mktemp -d /tmp/llying-infera-build.XXXX)"
trap 'rm -rf "$work"' EXIT
# A private copy: build artefacts and caches stay off the shared checkout.
git -C "$src" worktree list >/dev/null
git clone -q --no-hardlinks "$src" "$work/src"
git -C "$work/src" checkout -q "$(git -C "$src" rev-parse HEAD)"
git -C "$work/src" fetch -q --tags "$src" 2>/dev/null || true
cd "$work/src"

uv venv -q --python 3.12 "$work/venv"
py="$work/venv/bin/python"
uv pip install -q --python "$py" -e '.[sglang,planner]' pytest pytest-asyncio build wheel 'auditwheel>=6.8,<7' twine
echo "== unit tests"
"$py" -m pytest -q -x tests/unit/engine tests/unit/kv tests/unit/server 2>&1 | tail -15

echo "== wheels"
"$py" -m build -o "$work/dist" >/dev/null
bash .github/scripts/build_router.sh "$work/router"
PATH="$work/venv/bin:$PATH" bash .github/scripts/bundle_router_wheel.sh "$work"/dist/*-py3-none-any.whl "$work/router/infera-router" "$work/dist"
"$py" -m twine check "$work"/dist/*
ls -la "$work/dist"

echo "== install check (each wheel in its own venv)"
for whl in "$work"/dist/*.whl; do
  rm -rf "$work/verify"; uv venv -q --python 3.12 "$work/verify"
  uv pip install -q --python "$work/verify/bin/python" "$whl"
  (cd "$work" && verify/bin/python -c "import infera._version as v; print('version', v.version)")
  case "$whl" in *-manylinux_*)
    bin="$(cd "$work" && verify/bin/python -c 'from infera.server.launch_rust import _find_binary; print(_find_binary())')"
    echo "bundled router: $bin"; "$bin" --help | grep -E 'decode-request-weight|kv-events' | head -5 ;;
  esac
done
mkdir -p "$out"; cp "$work"/dist/* "$out"/
(cd "$out" && sha256sum * > SHA256SUMS && cat SHA256SUMS)
echo "done $(date -u +%FT%TZ)"
