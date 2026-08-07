# Reproduction kit — GLM-5.2 mix, agentic Case A under load

Goal: reproduce the loaded-run result in `README.md` from a clean machine with
cluster access.

**Two paths. Pick the one you need:**

- **§2 + §6 alone — offline audit.** The workload diff, every number in the
  results tables, and the saturation reading are all re-derivable from the files
  shipped here, with **no cluster, no GPU, no model**. Start here.
- **§0–§5 — a full re-run.** Needs an 8-GPU MI355X node. ~1 h 15 min.

**Estimated wall clock for a full re-run:**

| step | time |
|---|---|
| container + etcd + kvd | ~1 min |
| engine cold start to `/health` | **390 s** (weights + JIT + CUDA-graph capture) |
| smoke / feature gate | ~2 min |
| the loaded run | **~67 min** (400 s ramp + 3600 s sustain + drain) |

---

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

**The script and YAML in this packup ARE the ones that ran** (md5-verified —
table in `environment.md`). No repo checkout needed.

### 0b. If the staged driver is gone — rebuild it from upstream + the patch

```bash
git clone <Optimus-AgenticBench> agbench && cd agbench
git checkout 1cf01cbf169d9370a0bc8fe574055c5e975d1be9    # the exact commit
patch -p1 < /path/to/this/packup/patches/solo_m1_per_request_e2e_tpot.patch
```

**The patch is not optional for the E2E column.** Upstream records neither
per-request end-to-end latency nor an index-aligned TPOT array. Verify after a run:

```bash
python3 -c "
import json
r=json.loads(open('<results>/metrics.jsonl').readline())
assert 'new_e2es' in r and 'new_tpots' in r, 'SOLO_M1 patch is NOT applied'
print('SOLO_M1 present')"
```

Create the venv with **python 3.12** and `pip install -r agent/requirements.txt`.
`PyTorch was not found` on startup is expected and harmless.

---

## 1. Stage the kit on the node

`/mnt/vast` is shared to compute nodes; `/tmp` is not.

```bash
# from your workstation
scp -r scripts specs root@149.28.124.225:/mnt/vast/c_huggingface/glm52_load_repro/
```

Then on `chi2835`:

```bash
cd /mnt/vast/c_huggingface/glm52_load_repro/scripts && chmod +x *.sh
mkdir -p /mnt/vast/c_huggingface/glm52_load_repro/{logs,results}
```

Two values are site-specific. In `scripts/mix_site.sh`:

```bash
export MY_IP=10.2.122.78     # chi2835 enp193s0f1np1 — the DATA plane.
                             # NOT the 45.76.x management address.
export MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
```

and the `tokenizer:` line in `specs/mix_load.yaml` — though `run_agentic.sh`
passes `--tokenizer` on the CLI, which **overrides** the YAML. Set both the same.

---

## 2. Verify the workload IS Case A at a different offered load — DO THIS FIRST

This is the evidence that the run measures Case A's request shape and nothing
else. **It needs no cluster access.** Strip comments and blank lines from both
files and diff:

```bash
UP=<Optimus-AgenticBench>/agent/workloads/glm52_crxx_caseA.fix.yaml
strip() { sed -e 's/#.*$//' -e 's/[[:space:]]*$//' "$1" | grep -v '^[[:space:]]*$'; }
diff -u <(strip "$UP") <(strip specs/mix_load.yaml)
```

Expected — **exactly four changed lines**, both files 27 significant lines:

```diff
-  initial_sessions:      32
+  initial_sessions:      8
   new_session_rate:      0.10
-  max_sessions:          128
-  max_inflight:          48
+  max_sessions:          24
+  max_inflight:          16
   ramp_duration:         400
   sustain_duration:      3600
   system_prompt_len:     2000
   max_prompt_tokens:     260000
-  tokenizer:             /path/to/GLM-5.2-MXFP4
+  tokenizer:             /mnt/vast/xiaobo/models/GLM-5.2-MXFP4
```

Three are the mission's load knobs; the fourth is a site path. Everything defining
the request shape is untouched.

**Strip the comments.** The two files also differ in comment text (this one carries
`# mission task 3` markers), and an unstripped diff buries the four real changes in
noise.

Also check the file is the one that ran:

```bash
md5sum specs/mix_load.yaml    # -> 092c7fc2a6f7ab77601d8ab63a38b618
```

---

## 3. Bring the deployment up, and gate it

```bash
cd /mnt/vast/c_huggingface/glm52_load_repro/scripts
bash mix_site.sh up 2>&1 | tee ../logs/up.log
```

Five stages: container → etcd → kvd → mix worker → router. **Expect a long silence
at stage 4/5** — the 390 s cold start is not a hang. Last line:

```
[mix] mix ready on http://<MY_IP>:8100 after N min
```

Snapshot the environment **while it is live** (the resolved engine cmdline and kvd
counters cannot be recovered later):

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

---

## 4. Run it

```bash
cd /mnt/vast/c_huggingface/glm52_load_repro/scripts
export W=/mnt/vast/c_huggingface/glm52_load_repro
export ROUTER=http://10.2.122.78:8100

WORKLOAD=$W/specs/mix_load.yaml TAG=load bash run_agentic.sh
# backgrounds the driver via nohup and prints its pid. ~67 min.
```

Run under `tmux`. `run_agentic.sh` deliberately passes **no load knobs on the CLI**
— the YAML is the single source of truth for offered load. That matters more here
than anywhere else in this bench: the load knobs *are* the experiment. It does pass
`--dashboard-mode`, which is **mandatory**; without it nothing structured is
persisted and the run is unrecoverable once the terminal scrolls.

**Watch the saturation signal while it runs.** Unlike Phase 2, this warning
firing is expected and is a *result*, not a fault:

```bash
grep -a "Hit max_inflight" $W/logs/agentic_load.log
# -> WARNING: Hit max_inflight (16) - sessions are being throttled; offered load is capped
# We saw exactly one such line, at elapsed 406.6 s (the first sustain tick).
```

---

## 5. Copy the artifacts out

```bash
# results are already on the shared mount:
ls $W/results/agentic_load/load/<timestamp>/
#   metadata.json  metrics.jsonl  summary.json

# the driver console log — KEEP IT. It is the only per-error record.
gzip -c $W/logs/agentic_load.log > $W/logs/agentic_load.log.gz

# engine / router / kvd logs live INSIDE the container
for L in glm52_mix_base router kvd; do
  docker exec glm52_mix bash -c "gzip -c /tmp/$L.log" > $W/logs/$L.log.gz
done

# and a window-sliced engine log — see §7 for why this is the one to keep
docker exec glm52_mix bash -c "strings /tmp/glm52_mix_base.log" \
 | awk -F'[][]' '{split($2,t," "); h=t[2]; if (h>="11:55:00" && h<="13:05:00") print}' \
 | gzip -c > $W/logs/engine_loadwindow.log.gz
```

---

## 6. Re-derive every number — NO cluster access needed

This is the audit path. Everything below runs on the files in this packup.

```bash
cd <this packup>
mkdir -p /tmp/audit/load
gunzip -c results/load/metrics.jsonl.gz > /tmp/audit/load/metrics.jsonl
cp results/load/summary.json results/load/metadata.json /tmp/audit/load/
python3 scripts/analyze_solo.py /tmp/audit/load load
```

Expected — **exactly what was produced while assembling this packup**, matching
`README.md` to the decimal:

```
===== load — sustain phase, n=1637 requests =====
  prompt tokens : p50 75,450   (n=1637)
  gen tokens    : p50 320   mean 923.6
  cache hit     : mean 0.8883
  TTFT  ms      : p50    4266.8  p90    7295.8  p99   12335.5  mean    4604.5  n=1637
  E2E   ms      : p50   13931.9  p90   70313.9  p99  188236.2  mean   28353.5  n=1637
  TPOT  ms      : p50      25.3  p90      41.6  p99      77.3  mean      27.6  n=1637
```

`analyze_solo.py` counts **sustain-phase rows only** — ramp is a warm-up exclusion
window — and **drops** `new_tpots` entries equal to 0.0 rather than averaging them
in as zero-latency tokens (0.0 is the SOLO_M1 marker for "the driver filtered this
sample"). Despite its name it makes **no** concurrency-1 assumption; it works
unchanged here.

### 6a. The saturation reading — the headline finding

```bash
python3 - <<'PY'
import json
from collections import Counter
rows=[json.loads(l) for l in open('/tmp/audit/load/metrics.jsonl') if l.strip()]
print("ticks:", len(rows), dict(Counter(r['phase'] for r in rows)))
sus=[r for r in rows if r['phase']=='sustain']
for k,cap in (('in_flight',16),('num_sessions_active',24)):
    v=[r[k] for r in sus]; at=sum(1 for x in v if x>=cap)
    print("%-20s max=%d mean=%.2f at_cap=%d/%d (%.1f%%) min=%d"%(
          k,max(v),sum(v)/len(v),at,len(v),100*at/len(v),min(v)))
    print("   ", dict(sorted(Counter(v).items())))
PY
```

Expected:

```
ticks: 3992 {'ramp': 399, 'sustain': 3588, 'drain': 5}
in_flight            max=16 mean=15.30 at_cap=2588/3588 (72.1%) min=6
    {6:2, 7:2, 8:5, 9:17, 10:32, 11:92, 12:94, 13:139, 14:239, 15:378, 16:2588}
num_sessions_active  max=24 mean=22.42 at_cap=1685/3588 (47.0%) min=14
    {14:3, 15:26, 16:68, 17:91, 18:112, 19:147, 20:187, 21:207, 22:369, 23:693, 24:1685}
```

Both caps hit; **`max_inflight` binds** (72.1 % vs 47.0 %). See `notes.md` §3.

### 6b. The error timeline — gradual, not a burst

```bash
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('/tmp/audit/load/metrics.jsonl') if l.strip()]
prev=0
for r in rows:
    if r['errors']!=prev:
        print("elapsed=%8.1f  %2d->%2d  sent=%4d done=%4d  rate=%.2f%%"%(
            r['elapsed_seconds'],prev,r['errors'],r['requests_sent'],
            r['requests_completed'],100*r['errors']/max(r['requests_sent'],1)))
        prev=r['errors']
last=rows[-1]
print("final: sent=%d done=%d errors=%d abandoned=%s"%(
    last['requests_sent'],last['requests_completed'],last['errors'],
    last['num_sessions_abandoned']))
PY
```

Expected: 35 increments, the first at elapsed 658.8 s, the running rate 0.41 % at
the first and 1.4–1.97 % thereafter; final
`sent=1804 done=1755 errors=35 abandoned=0`.

### 6c. The 35 errors are ALL client timeouts — from the driver log

The driver prints one line per error. Strip ANSI first or the counts are wrong:

```bash
zcat logs/agentic_load.log.gz | sed -e 's/\x1b\[[0-9;]*m//g' > /tmp/audit/al.plain
printf "timed out        : "; grep -o  "Request [0-9]* timed out"  /tmp/audit/al.plain | wc -l
printf "unique ids       : "; grep -o  "Request [0-9]* timed out"  /tmp/audit/al.plain | sort -u | wc -l
printf "generic error:   : "; grep -oE "Request [0-9]+ error: "    /tmp/audit/al.plain | wc -l
printf "failed: HTTP     : "; grep -oE "Request [0-9]+ failed: HTTP" /tmp/audit/al.plain | wc -l
printf "Traceback        : "; grep -c   "Traceback"                 /tmp/audit/al.plain
printf "Hit max_inflight : "; grep -c   "Hit max_inflight"          /tmp/audit/al.plain
```

Expected: `35 / 35 / 0 / 0 / 0 / 1`. All 35 are one class; nothing else fired.

The client budget is `aiohttp.ClientTimeout(total=240)`
(`agent_throughput.py:2253`). Check where the *completed* requests sit against it
— note `new_e2es` is in **SECONDS**:

```bash
python3 - <<'PY'
import json
e=[x for l in open('/tmp/audit/load/metrics.jsonl') if l.strip()
     for r in [json.loads(l)] if r['phase']=='sustain'
     for x in (r.get('new_e2es') or []) if x]
e.sort(); n=len(e)
p=lambda q: e[min(n-1,round(q*(n-1)))]
print("n=%d  p50=%.1fs p90=%.1fs p99=%.1fs p99.9=%.1fs MAX=%.1fs"%(
      n,p(.5),p(.9),p(.99),p(.999),e[-1]))
print("completed with E2E > 240 s:", sum(1 for x in e if x>240))
PY
```

Expected: `n=1637 p50=13.9s p90=70.3s p99=188.2s p99.9=228.6s MAX=239.0s`, and
**0** above 240 s. The observed distribution is censored at the budget by
construction. `notes.md` §4, §5.

### 6d. Cross-check the two error records agree 1:1

```bash
python3 - <<'PY'
import re,json
txt=open('/tmp/audit/al.plain').read()
cur=None; outs=[]
for t,rid in re.findall(r'\[\s*(\d+\.\d)s\]|Request (\d+) timed out', txt):
    if t: cur=float(t)
    else: outs.append(cur)
rows=[json.loads(l) for l in open('/tmp/audit/load/metrics.jsonl') if l.strip()]
prev=0; inc=[]
for r in rows:
    if r['errors']!=prev: inc.append(r['elapsed_seconds']); prev=r['errors']
d=[abs(a-b) for a,b in zip(outs,inc)]
print("log timeouts=%d  metrics increments=%d  max offset=%.1fs  mean=%.1fs"%(
      len(outs),len(inc),max(d),sum(d)/len(d)))
PY
```

Expected: `35 / 35 / max offset 1.9s / mean 1.0s` against a ~1 s tick period.

---

## 7. Re-derive the MTP acceptance — SCOPE BY TIME WINDOW

The engine log is appended across the **whole session**: Phase 1's sweep
(07:13–09:45), Phase 2's three solo arms (09:53–11:52), and this run
(11:56–13:03) all land in one file spanning 07:02:13 → 13:02:48. An unscoped grep
mixes them.

**From the sliced log shipped here — no cluster needed:**

```bash
zcat logs/engine_loadwindow.log.gz | grep -o 'accept len: [0-9.]*' \
 | awk '{print $3}' | sort -n \
 | awk '{a[NR]=$1; if($1>=4)f++}
    END{printf "n=%d p10=%s MEDIAN=%s p90=%s at4.00=%d (%.1f%%)\n",
        NR,a[int(NR*0.1)+1],a[int(NR*0.5)+1],a[int(NR*0.9)+1],f,100*f/NR}'
```

Expected (this run's window only):

```
n=8918 p10=2.48 MEDIAN=3.14 p90=3.82 at4.00=400 (4.5%)
```

The **whole-log** read over all three phases gives a materially different answer —
`n=51685 p10=2.73 MEDIAN=3.55 p90=3.98 at4.00=4716 (9.1%)`. Both are reported in
this packup, each labelled. That they differ is exactly why scoping matters
(`notes.md` §7).

**On the node, against the live container**, stage the script as a **file** — a
nested `ssh … "ssh … 'docker exec … awk \"…\"'"` one-liner mangles the awk quoting
and silently produces a division-by-zero:

```bash
cat > /tmp/acc_win.sh <<'EOF'
#!/bin/bash
S="$1"; E="$2"
docker exec glm52_mix bash -c "strings /tmp/glm52_mix_base.log" 2>/dev/null \
 | awk -F'[][]' -v s="$S" -v e="$E" '{split($2,t," "); if (t[2]>=s && t[2]<=e) print}' \
 | grep -o 'accept len: [0-9.]*' | awk '{print $3}' | sort -n \
 | awk '{a[NR]=$1; if($1>=4)f++}
     END{if(!NR){print "no lines in window";exit}
         printf "n=%d p10=%s MEDIAN=%s p90=%s at4.00=%d (%.1f%%)\n",
                NR,a[int(NR*0.1)+1],a[int(NR*0.5)+1],a[int(NR*0.9)+1],f,100*f/NR}'
EOF
bash /tmp/acc_win.sh 11:56:00 13:05:00
```

`scripts/accept_len.sh` does the **unscoped** version — use it only on a log you
know belongs to one run. `scripts/scan_err.sh` takes a window prefix as its second
argument and exists precisely because of this hazard.

**The engine-side error scan** (this is how "the engine rejected nothing" was
established):

```bash
bash scripts/scan_err.sh /tmp/glm52_mix_base.log "2026-08-06 1[12]:"
```

Our reading over 11:56–13:05: **0** error/abort/reject/invalid/Traceback lines,
and **1802 / 1802 HTTP 200**.

kvd counters, read live (cumulative across all phases, **not** attributable to
this one):

```bash
docker exec glm52_mix python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock
```

Our reading: 71,879 entries / 85.4 GB host / 68.7 GB L3 / gets 69,126 /
sets 486,924 / hits 67,060 / misses 2,066 / evictions 319,447.

---

## 8. Tear down — only your own containers

```bash
bash mix_site.sh down   # reaps the engine, WAITS for VRAM to drain, removes
                        # the glm52_mix and glm52-mix-etcd containers
```

The VRAM wait is the point, not the kill: the infera wrapper exits before its
sglang child does, and that child keeps the KV-event port block bound — the next
launch then dies with "port_base is not available".

**Before removing any container you did not just create, prove it is yours:**
`docker inspect <ctr> --format '{{.Created}} {{json .HostConfig.Binds}}'`.

> The deployment used for this packup was **left running** on purpose. If you are
> reproducing on that same node, check whether it is still up before launching a
> second one.

---

## Expected output

One result directory with `metadata.json` / `metrics.jsonl` / `summary.json`, and
`analyze_solo.py` reproducing the §6 table. Headline numbers to check:

| metric | value |
|---|---|
| sent / completed / errors | 1804 / 1755 / 35 |
| success rate | 0.9728 |
| TTFT p50 (sustain) | 4266.8 ms |
| E2E p50 (sustain) | 13931.9 ms |
| TPOT p50 (sustain) | 25.3 ms |
| cache actual / ideal | 0.8806 / 0.8899 |
| `in_flight` at cap | 72.1 % of sustain ticks |
| `num_sessions_active` at cap | 47.0 % of sustain ticks |

**Do not expect bit-identical numbers.** Sampling is at temperature 1.0 (the
checkpoint's own `generation_config` — never greedy-decode this model), and this
workload samples a real distribution rather than pinning it. The **cache hit rate
should land near 0.88–0.89**; that one is controlled, and a large miss there means
the ramp did not warm the shared prefix.

**Do expect the caps to be hit again** at this offered load, on comparable
hardware. If they are not, the deployment is faster than the one measured here and
the numbers are not comparable.

## If it doesn't reproduce

See `notes.md`. The short list:

- Silence at stage 4/5 for several minutes is the **390 s cold start**, not a hang.
- `new_e2es` missing from `metrics.jsonl` → the SOLO_M1 patch is not applied.
- E2E looks 1000× too small → `new_e2es` is in **seconds**, not ms.
- TTFT p99 far above the table → you forgot the `phase == "sustain"` filter and
  are including the ramp's cold requests.
- Cache hit well below 0.88 → the ramp did not warm the shared prefix. Check
  `PYTHONHASHSEED=0`.
- MTP acceptance pinned at 4.00 → check you did not set temperature 0.
- Model returns 200s with garbage → the DSA env block did not take effect.
- Grepping the engine log without a time window will mix in the other two phases.
