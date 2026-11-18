#!/usr/bin/env bash
# All per-run analyses of a P4D4 round into the branch's results directory (run on a compute node via a queue).
#   post_run.sh RUN_DIR_NAME [RESULT_NAME]     e.g. post_run.sh p4d4-m5-v3comm-render4 m5
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
cd "$R" || exit 1
run=${1:?run}
name=${2:-${run#p4d4-}}
d=runs/$run
O=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-p4d4-crusoe-20261117/results/$name
mkdir -p "$O"
ip=$(sed -n 's/.*IP=\([0-9.]*\).*/\1/p' "$d/engine-config.env" | head -1)
P=http://$ip:27201 D=http://$ip:27301
python3 scripts/window_tput.py "$d" 300 > "$O/window_tput.txt" 2>&1
python3 scripts/ttft_breakdown.py "$d" > "$O/ttft_breakdown.txt" 2>&1
python3 scripts/samples_summary.py "$d/samples.jsonl" "$P" "$D" "$d/agentx-c104/benchmark.log" > "$O/samples_summary.txt" 2>&1
python3 scripts/hit_timeline.py "$d/samples.jsonl" "$P" 300 > "$O/hit_timeline.txt" 2>/dev/null
python3 scripts/ideal_hit.py "$d" > "$O/ideal_hit.txt" 2>&1
grep -qa prefill_usage <(zcat -f "$d"/vpd-router*.log* 2>/dev/null | head -c 200000000) && python3 scripts/prefill_usage.py "$d" > "$O/prefill_usage.txt" 2>&1
python3 - "$d" "$P" "$D" > "$O/render_and_stages.txt" 2>&1 <<'EOF'
# Per-request render time, engine queue / prefill / TTFT, and prefill computed tokens over the measurement (from 300 s).
import json, sys, re, datetime
d, purl, durl = sys.argv[1:4]
log = open(f"{d}/agentx-c104/benchmark.log", errors="replace").read()
m = re.search(r"(\d\d):(\d\d):(\d\d)\.\d+ NOTICE\s+Phase profiling \(profiling\) started", log)
day = datetime.datetime.fromtimestamp(__import__("os").path.getmtime(f"{d}/agentx-c104/benchmark.log"), datetime.timezone.utc).date()
t0 = datetime.datetime(day.year, day.month, day.day, int(m[1]), int(m[2]), int(m[3]), tzinfo=datetime.timezone.utc).timestamp() + 300
pts = {}
for line in open(f"{d}/samples.jsonl"):
    x = json.loads(line)
    if not x.get("m") or x["t"] < t0:
        continue
    s = {}
    for e in x["m"].values():
        for k, v in e.items():
            s[k] = s.get(k, 0.0) + v
    pts.setdefault(x["url"], []).append((x["t"], s))
for url, name in ((purl, "prefill"), (durl, "decode")):
    s = pts.get(url, [])
    if len(s) < 2:
        continue
    a, b = s[0][1], s[-1][1]
    dl = lambda k: b.get(k, 0) - a.get(k, 0)
    print(f"{name} ({s[-1][0] - s[0][0]:.0f} s from minute 5):")
    for k in ("request_render_duration_seconds", "request_queue_time_seconds", "request_prefill_time_seconds", "time_to_first_token_seconds"):
        ks = [x for x in b if x.startswith(f"vllm:{k}_sum")]
        kc = [x for x in b if x.startswith(f"vllm:{k}_count")]
        su, co = sum(dl(x) for x in ks), sum(dl(x) for x in kc)
        if co:
            print(f"  {k:36s} {su / co:7.3f} s per request ({co:.0f})")
    c, n = dl("vllm:request_prefill_kv_computed_tokens_sum"), dl("vllm:request_prefill_kv_computed_tokens_count")
    if n:
        print(f"  computed tokens per request           {c / n:9.0f}  ({n:.0f})")
    g = dl("vllm:generation_tokens_total")
    print(f"  generation tok/s                      {g / (s[-1][0] - s[0][0]):9.0f}")
EOF
cp "$d/engine-config.env" "$O/" 2>/dev/null
cp "$d/agentx-c104/agentx_conc104.json" "$O/" 2>/dev/null
for f in container-state.txt meminfo-before.txt meminfo-after.txt; do [ -s "$d/$f" ] && cp "$d/$f" "$O/"; done
[ -s "$d/dmesg-after.txt" ] && grep -a -i -E "segfault|trap|killed process|out of memory|general protection|mce" "$d/dmesg-after.txt" > "$O/dmesg-faults.txt"
grep -a "$run" logs/summary.out | tail -1 > "$O/summary.txt"
ls "$O"
