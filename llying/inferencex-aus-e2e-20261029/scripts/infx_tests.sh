#!/usr/bin/env bash
# infx unit tests for a checkout, plus a probe of the node facts the smci355 cluster record assumes.
#   sbatch infx_tests.sh <inferencex-e2e dir> [pytest args...]
#SBATCH --job-name=llying-infx-tests
#SBATCH --partition=Compute-DCPT
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/apps/liyingli/jobs/%x-%j.out
set -uo pipefail
e2e="${1:?inferencex-e2e dir}"; shift
export PATH="$HOME/.local/bin:$PATH"
echo "node $(hostname) $(date -u +%FT%TZ) $(git -C "$e2e" log -1 --format='%h %s')"
echo "== node probe"
ls -l /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so /lib/x86_64-linux-gnu/libionic.so* 2>&1
ls /sys/class/infiniband 2>&1 | tr '\n' ' '; echo
ip -br addr 2>&1 | grep -v "^veth\|^docker\|^lo"
ls -l /etc/slurm/task_prolog.sh /var/lib/hf-hub-cache 2>&1
srun --help 2>/dev/null | grep -c container-image
df -h /tmp / 2>&1 | tail -2
free -g | head -2
echo "== tests"
work="$(mktemp -d /tmp/llying-infx.XXXX)"; trap 'rm -rf "$work"' EXIT
rsync -a --exclude utils/srt-slurm --exclude '.venv*' "$e2e/" "$work/e2e/"
cd "$work/e2e"
UV_PROJECT_ENVIRONMENT="$work/venv" uv sync -q --python 3.12 --no-dev --all-extras 2>&1 | tail -3
uv pip install -q --python "$work/venv/bin/python" pytest pytest-xdist aiohttp numpy packaging 2>&1 | tail -3
UV_PROJECT_ENVIRONMENT="$work/venv" uv run --no-sync python -m pytest -q -p no:cacheprovider -n 16 "${@:-infx/tests/srt_slurm infx/tests/launch infx/tests/matrix}" 2>&1 | tail -25
echo "done $(date -u +%FT%TZ)"
