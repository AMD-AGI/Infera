# Reproduction kit — GLM-5.2 mix, agentic at concurrency 1

Goal: reproduce the three-arm latency table in `README.md` from a clean machine
with cluster access.

**Two paths.** Pick the one you need:

- **§6 alone — offline re-analysis.** Every number in the results table is
  re-derivable from the `metrics.jsonl` files shipped in this packup, with no
  cluster, no GPU, and no model. Start here to audit the numbers.
- **§0–§5 — a full re-run.** Needs an 8-GPU MI355X node. ~2 h 20 min.

**Estimated wall clock for a full re-run:**

| step | time |
|---|---|
| container + etcd + kvd | ~1 min |
| engine cold start to `/health` | **390 s** (weights + JIT + CUDA-graph capture) |
| smoke / feature gate | ~2 min |
| p50 arm | **13 min** (180 s ramp + 600 s sustain) |
| p90 arm | **28 min** (180 s ramp + 1500 s sustain) |
| p99 arm | **73 min** (180 s ramp + 4200 s sustain) |

The three arms are **strictly serial** — they share one deployment and each one
must own the whole machine for its concurrency-1 measurement to mean anything.

## 0. Prerequisites

**Machine.** One MI355X (gfx950) 8-GPU node. We used `chi2835`.

```bash
ssh root@149.28.124.225      # jump host (chi2866)
ssh chi2835                  # the compute node
```

**Before committing to a node, read its VRAM directly** — the cluster-status
snapshot's VRAM column can lag:

```bash
rocm-smi --csv --showmeminfo vram | tail -8 \
  | awk -F, '{s+=$3} END {printf "VRAM in use: %.0f GB\n", s/1073741824}'
```

You need ~0 GB. If it is not, find the holder and **do not** displace someone
else's job. Never `scancel`; never prune images or `/tmp`.

**Secrets.** None beyond cluster SSH. See `environment.md` § "Required secrets".

**External dependencies (absolute paths, not in this repo):**

| what | path |
|---|---|
| model + tokenizer + EAGLE draft | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` |
| agentic driver (staged, patched) | `/mnt/vast/c_huggingface/bench_20260801/agbench` |
| driver venv (python 3.12.3) | `/mnt/vast/c_huggingface/bench_20260801/venv/bin/python` |
| image | `infera/engine-sglang:merged-e`, id `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a` |
| etcd | `quay.io/coreos/etcd:v3.5.14` |

Check the image is on the node before pulling:

```bash
docker images --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep engine-sglang
```

**The scripts and YAMLs in this packup ARE the ones that ran** (md5-verified
against the cluster copies — table in `environment.md`). No repo checkout needed.

### 0b. If the staged driver is gone — rebuild it from upstream + the patch

The driver lives on a shared mount, not in this repo. To recreate it:

```bash
git clone <Optimus-AgenticBench> agbench && cd agbench
git checkout 1cf01cbf169d9370a0bc8fe574055c5e975d1be9    # the exact commit
patch -p1 < /path/to/this/packup/patches/solo_m1_per_request_e2e_tpot.patch
```

**The patch is not optional.** Upstream records neither per-request end-to-end
latency nor an index-aligned TPOT array, so the E2E column simply does not exist
without it. See `patches/README.md`. Verify:

```bash
python3 -c "
import json,sys
# after a run, every tick must carry the SOLO_M1 keys
r=json.loads(open('<results>/metrics.jsonl').readline())
assert 'new_e2es' in r and 'new_tpots' in r, 'SOLO_M1 patch is NOT applied'
print('SOLO_M1 present')"
```

Create the venv with **python 3.12** and `pip install -r agent/requirements.txt`.
`PyTorch was not found` on startup is expected and harmless — the driver only
uses `tokenizers`.

## 1. Stage the kit on the node

`/mnt/vast` is shared to compute nodes; `/tmp` is not.

```bash
# from your workstation
scp -r scripts specs root@149.28.124.225:/mnt/vast/c_huggingface/glm52_solo_repro/
```

Then on `chi2835`:

```bash
cd /mnt/vast/c_huggingface/glm52_solo_repro/scripts && chmod +x *.sh
mkdir -p /mnt/vast/c_huggingface/glm52_solo_repro/{logs,results}
```

## 2. Point the workloads at your paths

Two values are site-specific.

In `scripts/mix_site.sh`:

```bash
export MY_IP=10.2.122.78     # chi2835 enp193s0f1np1 — the DATA plane.
                             # NOT the 45.76.x management address.
export MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
```

In each `specs/mix_solo_*.yaml`, the `tokenizer:` line — though `run_agentic.sh`
passes `--tokenizer` on the CLI, which **overrides** the YAML (the driver prints
`Skipped (CLI override): tokenizer`). Set both to the same path to avoid
confusion.

## 3. Bring the deployment up, and gate it

```bash
cd /mnt/vast/c_huggingface/glm52_solo_repro/scripts
bash mix_site.sh up 2>&1 | tee ../logs/up.log
```

Five stages: container → etcd → kvd → mix worker → router. **Expect a long
silence at stage 4/5** — the 390 s cold start is not a hang. Last line:

```
[mix] mix ready on http://<MY_IP>:8100 after N min
```

Snapshot the environment **while it is live** (the resolved engine cmdline and
kvd counters cannot be recovered later):

```bash
bash envsnap.sh > ../logs/env_$(hostname).txt   # compare against env/env_chi2835.txt
```

Then run the feature gate. **Read the blocks, not the exit code:**

```bash
bash mix_site.sh smoke 2>&1 | tee ../logs/smoke.log
```

Do not proceed unless all of these hold:

| check | must read |
|---|---|
| `/v1/workers` | exactly 1 worker, `disagg_mode: "mixed"`, active |
| chat completion | a coherent answer. **Garbage = the DSA-on-ROCm env block did not take effect** — not a sampling problem. |
| DP-attention | `dp_size=8`, `enable_dp_attention=True`, 8 live scheduler ranks |
| MTP accept-len | median in **2–3**. A median **at 4.00 is a failure**, not a win. |
| kv-aware | `router_policy: "kv-aware"` **and** tokenizer loaded — the policy line alone is insufficient; kv-aware degrades silently to load-only routing if the tokenizer did not load |
| kvd | **8 adapters** (one per DP rank) |
| prefix cache | `cached_tokens` non-zero on the *second* identical prompt |

The prefix-cache row matters more here than anywhere else in this bench: the
whole Phase-2 measurement is defined at a warm 0.89 hit rate.

## 4. Run the three arms — serially

```bash
cd /mnt/vast/c_huggingface/glm52_solo_repro/scripts
export W=/mnt/vast/c_huggingface/glm52_solo_repro
export ROUTER=http://10.2.122.78:8100

for A in p50 p90 p99; do
  WORKLOAD=$W/specs/mix_solo_$A.yaml TAG=solo_$A bash run_agentic.sh
  # run_agentic.sh backgrounds the driver via nohup and prints its pid.
  # WAIT for it before starting the next arm — overlapping them destroys
  # the concurrency-1 property that is the entire point of this phase.
  wait
done
```

`run_agentic.sh` deliberately passes **no load knobs on the CLI**. The YAML is
the single source of truth for offered load; passing `--initial-sessions` would
silently shadow it and make the run unreproducible from the file alone. It does
pass `--dashboard-mode`, which is **mandatory**: without it nothing structured is
persisted and the run is unrecoverable once the terminal scrolls.

Expected durations: 13 / 28 / 73 min. Run under `tmux`.

**Sanity check while an arm runs** — in-flight must never exceed 1:

```bash
tail -f $W/logs/agentic_solo_p50.log | grep -i "max_inflight"
# Any "Hit max_inflight" line means the session bookkeeping is suspect and no
# number from that run should be trusted. We saw ZERO such lines on all 3 arms.
```

## 5. Copy the artifacts out

```bash
# results are already on the shared mount:
ls $W/results/agentic_solo_p50/solo_p50/<timestamp>/
#   metadata.json  metrics.jsonl  summary.json

# the driver console logs
gzip -c $W/logs/agentic_solo_p50.log > $W/logs/agentic_solo_p50.log.gz

# engine / router / kvd logs live INSIDE the container
for L in glm52_mix_base router kvd; do
  docker exec glm52_mix bash -c "gzip -c /tmp/$L.log" > $W/logs/$L.log.gz
done
```

## 6. Re-derive the numbers — NO cluster access needed

This is the audit path. Everything below runs on the files in this packup.

```bash
cd <this packup>
for A in p50 p90 p99; do python3 scripts/analyze_solo.py results/solo/$A solo_$A; echo; done
```

Expected output — **this is exactly what was produced while assembling the
packup**, and it matches `README.md` to the decimal:

```
===== solo_p50 — sustain phase, n=106 requests =====
  prompt tokens : p50 74,013   (n=106)
  gen tokens    : p50 317   mean 317.5
  cache hit     : mean 0.8898
  TTFT  ms      : p50    1811.7  p90    2815.9  p99    3413.3  mean    2022.4  n=106
  E2E   ms      : p50    5111.1  p90    5910.2  p99    6798.8  mean    5170.0  n=106
  TPOT  ms      : p50      10.0  p90      10.7  p99      11.5  mean       9.9  n=106

===== solo_p90 — sustain phase, n=46 requests =====
  prompt tokens : p50 155,013   (n=46)
  gen tokens    : p50 3,297   mean 3,296.1
  cache hit     : mean 0.8897
  TTFT  ms      : p50    3674.7  p90    4730.9  p99    5349.1  mean    3902.4  n=46
  E2E   ms      : p50   30781.4  p90   33478.7  p99   39312.8  mean   31389.3  n=46
  TPOT  ms      : p50       8.1  p90       9.1  p99      10.8  mean       8.3  n=46

===== solo_p99 — sustain phase, n=28 requests =====
  prompt tokens : p50 235,013   (n=28)
  gen tokens    : p50 16,993   mean 16,980.6
  cache hit     : mean 0.8900
  TTFT  ms      : p50    5663.6  p90    5854.7  p99    7486.4  mean    5746.7  n=28
  E2E   ms      : p50  147404.6  p90  155157.9  p99  166228.8  mean  146988.7  n=28
  TPOT  ms      : p50       8.3  p90       8.8  p99       9.4  mean       8.3  n=28
```

`analyze_solo.py` counts **sustain-phase rows only** — ramp is a warm-up
exclusion window, not a load ramp — and **drops** `new_tpots` entries equal to
0.0 rather than averaging them in as zero-latency tokens (0.0 is the SOLO_M1
marker for "the driver filtered this sample").

### Verify the concurrency-1 claim yourself

```bash
python3 - <<'PY'
import json, glob
from collections import Counter
for a in ('p50','p90','p99'):
    rows=[json.loads(l) for l in open(f'results/solo/{a}/metrics.jsonl') if l.strip()]
    print(a, 'in_flight:', dict(Counter(r['in_flight'] for r in rows)),
             'active_sessions:', dict(Counter(r['num_sessions_active'] for r in rows)),
             'errors:', sorted({r['errors'] for r in rows}))
PY
```

Expected: `in_flight` only ever 0 or 1, `num_sessions_active` only ever 0 or 1,
`errors` always `[0]`. Our reading:

```
p50 in_flight: {0: 61, 1: 722}   active_sessions: {1: 723, 0: 60}  errors: [0]
p90 in_flight: {0: 43, 1: 1638}  active_sessions: {1: 1639, 0: 42} errors: [0]
p99 in_flight: {0: 29, 1: 4346}  active_sessions: {1: 4347, 0: 28} errors: [0]
```

### Verify the ramp really was a warm-up window

Compare the two phases in each `summary.json`. The cache hit rate climbing into
sustain is the evidence that the prefix became resident during ramp:

| arm | ramp completed | ramp cache hit | ramp TTFT p99 | sustain cache hit | sustain TTFT p99 |
|---|---|---|---|---|---|
| p50 | 26 | 0.8556 | 20875.8 ms | **0.8898** | 3413.3 ms |
| p90 | 5 | 0.7968 | 16015.1 ms | **0.8897** | 5349.1 ms |
| p99 | 1 | 0.5869 | 18462.2 ms | **0.8900** | 7486.4 ms |

The first request of each arm pays a cold TTFT (16–21 s) that never appears in
the reported numbers. That is the ramp doing its job.

## 7. Explain the p99 arm's success_rate 0.9667

`summary.json` reports `requests_sent 30 / requests_completed 29`, hence
`success_rate 0.9667`, while `errors` is **0**. Trace it:

```bash
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('results/solo/p99/metrics.jsonl') if l.strip()]
prev=0
for r in rows:
    if r['requests_sent']!=prev and r['requests_sent']>=29:
        print('sent ->%2d at elapsed=%8.1f (%s)'%(r['requests_sent'],r['elapsed_seconds'],r['phase']))
    prev=r['requests_sent']
last=rows[-1]
print('final: sent=%d done=%d errors=%d inflight=%d'%(
    last['requests_sent'],last['requests_completed'],last['errors'],last['in_flight']))
PY
```

Output:

```
sent ->29 at elapsed=  4158.4 (sustain)
sent ->30 at elapsed=  4302.7 (sustain)
final: sent=30 done=29 errors=0 inflight=1
```

The 30th request was sent 77 s before the 4380 s budget expired and was **still
in flight** at the cutoff — a request of this shape takes ~147 s. It is
truncation by the clock, not a failure, and it never entered the sustain-phase
arrays (`n=28`).

## 8. Re-derive the MTP acceptance figures — SCOPE BY TIME WINDOW

The engine log is appended across the **whole session**: Phase 1's sweep, these
three arms, and Phase 3 afterwards all land in one file. An unscoped grep mixes
them and is a guaranteed false reading.

```bash
# on the node, against the LIVE container
cat > /tmp/acc_win.sh <<'EOF'
#!/bin/bash
S="$1"; E="$2"; A="$3"
docker exec glm52_mix bash -c "strings /tmp/glm52_mix_base.log" 2>/dev/null \
 | awk -F'[][]' -v s="$S" -v e="$E" '/accept len:/ {split($2,t," "); if (t[2]>=s && t[2]<=e) print}' \
 | grep -o 'accept len: [0-9.]*' | awk '{print $3}' | sort -n \
 | awk -v A="$A" '{a[NR]=$1; if ($1>=4) f++}
     END {if (!NR) {printf "%s: no lines in window\n", A; exit 0}
          printf "%-4s n=%-6d p10=%-5s MEDIAN=%-5s p90=%-5s at4.00=%d (%.1f%%)\n",
                 A, NR, a[int(NR*0.1)+1], a[int(NR*0.5)+1], a[int(NR*0.9)+1], f, 100*f/NR}'
EOF
bash /tmp/acc_win.sh 09:53:41 10:07:30 p50
bash /tmp/acc_win.sh 10:07:44 10:38:40 p90
bash /tmp/acc_win.sh 10:38:57 11:52:00 p99
```

Our readings (2026-08-06 UTC):

```
p50  n=393    p10=2.42  MEDIAN=2.73  p90=3.05  at4.00=0 (0.0%)
p90  n=1266   p10=2.62  MEDIAN=3.55  p90=4.00  at4.00=147 (11.6%)
p99  n=3475   p10=3.10  MEDIAN=3.70  p90=4.00  at4.00=785 (22.6%)
```

Note the script must be **staged as a file**. A nested
`ssh … "ssh … 'docker exec … awk \"…\"'"` one-liner mangles the awk quoting and
silently produces a division-by-zero (we hit exactly that; see `notes.md` §8).

`scripts/accept_len.sh` does the **unscoped** version — use it only on a log you
know belongs to one run.

kvd counters, read live (cumulative across all phases, **not** attributable to
this one):

```bash
docker exec glm52_mix python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock
```

## 9. Tear down — only your own containers

```bash
bash mix_site.sh down   # reaps the engine, WAITS for VRAM to drain, removes
                        # the glm52_mix and glm52-mix-etcd containers
```

The VRAM wait is the point, not the kill: the infera wrapper exits before its
sglang child does, and that child keeps the KV-event port block bound — the next
launch then dies with "port_base is not available".

**Before removing any container you did not just create, prove it is yours:**
`docker inspect <ctr> --format '{{.Created}} {{json .HostConfig.Binds}}'`.

## Expected output

Three result directories, each with `metadata.json` / `metrics.jsonl` /
`summary.json`, and `analyze_solo.py` reproducing the §6 table. Headline numbers
to check:

| arm | TTFT p50 | E2E p50 | TPOT p50 | cache hit | errors |
|---|---|---|---|---|---|
| p50 | 1811.7 ms | 5111.1 ms | 10.0 ms | 0.8898 | 0 |
| p90 | 3674.7 ms | 30781.4 ms | 8.1 ms | 0.8897 | 0 |
| p99 | 5663.6 ms | 147404.6 ms | 8.3 ms | 0.8900 | 0 |

**Do not expect bit-identical numbers.** Sampling is at temperature 1.0 (the
checkpoint's own `generation_config` — never greedy-decode this model), and the
generation length varies request to request. The **cache hit rate should land on
0.89 within ~0.001** on all three arms; that one is controlled, and a miss there
means the ramp did not warm the prefix.

## If it doesn't reproduce

See `notes.md`. The short list:

- Silence at stage 4/5 for several minutes is the **390 s cold start**, not a hang.
- `new_e2es` missing from `metrics.jsonl` → the SOLO_M1 patch is not applied.
- Cache hit well below 0.89 → the ramp did not warm the prefix. Check
  `PYTHONHASHSEED=0` and that the earlier arm did not evict it.
- MTP acceptance pinned at 4.00 → check you did not set temperature 0.
- Model returns 200s with garbage → the DSA env block did not take effect.
- `Hit max_inflight` in the driver log → session bookkeeping is suspect; discard.
- Grepping the engine log without a time window will mix in the other phases.
