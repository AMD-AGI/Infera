#!/usr/bin/env bash
# Full infx test suite on a base and a head inferencex-e2e checkout; print failures only head has.
#   sbatch infx_compare.sh <base e2e dir> <head e2e dir>
#SBATCH --job-name=llying-infx-compare
#SBATCH --partition=Compute-DCPT
#SBATCH --qos=batch
#SBATCH --account=xiewen12
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/apps/liyingli/jobs/%x-%j.out
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
work="$(mktemp -d /tmp/llying-infxcmp.XXXX)"; trap 'rm -rf "$work"' EXIT
echo "node $(hostname) $(date -u +%FT%TZ)"
run() {  # run <name> <e2e dir>
    local name="$1" src="$2"
    echo "== $name $(git -C "$src" log -1 --format='%h %s')"
    rsync -a --exclude '.venv*' --exclude 'srt-slurm-*' "$src/" "$work/$name/"
    cd "$work/$name"
    UV_PROJECT_ENVIRONMENT="$work/venv-$name" uv sync -q --python 3.12 --all-extras --group test 2>&1 | tail -3
    UV_PROJECT_ENVIRONMENT="$work/venv-$name" uv run --no-sync python -m pytest -q -p no:cacheprovider \
        -n 32 -rfE infx/tests > "$work/$name.txt" 2>&1
    tail -1 "$work/$name.txt"
    grep -E '^(FAILED|ERROR) ' "$work/$name.txt" | sed 's/ - .*//' | sort -u > "$work/$name.fail"
}
run base "$1"
run head "$2"
echo "== failing on head only"
comm -13 "$work/base.fail" "$work/head.fail"
echo "== failing on base only"
comm -23 "$work/base.fail" "$work/head.fail"
echo "== head failures (all)"; cat "$work/head.fail"
for f in $(comm -13 "$work/base.fail" "$work/head.fail" | awk '{print $2}'); do
    grep -A30 "_ ${f##*::} _" "$work/head.txt" | head -40
done
echo "done $(date -u +%FT%TZ)"
