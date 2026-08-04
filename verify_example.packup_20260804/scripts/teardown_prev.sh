#!/usr/bin/env bash
# Tear down OUR OWN previous 1P1D deployment (the bench_run stack from
# par8/agentx) so the example kit can take the GPUs. Runs ON the node.
#
# Ownership was proven before writing this: `docker inspect bench_run` shows the
# exact bind set our reset_node.sh creates (/mnt/vast + our libionic mount), and
# its logs are in our own /mnt/vast/c_huggingface/bench_20260801/logs/.
#
# NOT touched: the slurm hold (yeandy-debug), any other container on these
# shared nodes, any image.
set -u

echo "===== containers (ours only) ====="
for c in bench_run bench_run_etcd merged_run merged_run_etcd; do
  docker rm -f "$c" >/dev/null 2>&1 && echo "  removed $c" || true
done

echo "===== stray engine processes (ours only) ====="
# Escaped dots: `pkill -f` matches a REGEX, and an unescaped 'infera.kvd' also
# matches the engine's own --infera-kvd-socket argument.
for p in 'infera\.engine\.sglang' 'infera\.server' '-m infera\.kvd ' 'sglang\.launch_server'; do
  pkill -9 -f -- "$p" 2>/dev/null && echo "  killed: $p" || true
done
sleep 8

echo "===== wait for VRAM to drain ====="
for i in $(seq 1 60); do
  n=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+' || true); n=${n:-0}
  [ "$n" -eq 0 ] && { echo "  GPUs idle after $((i*2))s"; break; }
  sleep 2
done
rocm-smi --csv --showmeminfo vram 2>/dev/null | tail -8 | awk -F, \
  '{printf "  %s %.1f GB used\n", $1, $3/1073741824}'
echo "===== $(hostname) clear ====="
