# Reproduce

Ordered and copy-pasteable. Read `environment.md` first for what must exist
before step 0.

**Total wall clock: ~55 min** on a live deployment (~37 min run + ~10 min
prefill relaunch + checks), or **~70 min** from cold.

Throughout: `J` = jump host `root@149.28.124.225`, `P` = chi2879 (prefill),
`D` = chi2867 (decode), `W` = `/mnt/vast/c_huggingface/bench_20260801`.

> **Shared-node rules, non-negotiable.** The slurm hold on both nodes belongs to
> `yeandy-debug`. **Never `scancel`.** Kill only your own `bench_run` container
> and your own sglang processes. Do not prune images. Do not fill disks.
>
> **Never bare-`pkill -f infera.kvd`** — `-f` is a regex and `.` is a wildcard,
> so it matches the engine's own `--infera-kvd-socket` argument and kills the
> engine too. Use `scripts/restart_kvd.sh`.
>
> **Do NOT restart the decode leg.** It carries the runtime-applied
> `GLM52_P1V3` patch (`patches/0004`), which is lost on restart, and without it
> decode dies within minutes on this request shape.

---

## 0. Prerequisites

This experiment assumes the deployment from
`../solo.glm52.latencyfloor.packup_20260802/` is already up. If `/health`
returns `active_workers: 2`, skip to step 1.

From cold, follow that kit's `REPRODUCE.md` steps 0–3 (stage, reset nodes,
launch both legs, patch decode with `apply_p1v3.py`, patch the driver with
`apply_solo_metrics.py`, start the router). Every script it needs is duplicated
here in `scripts/` so this kit stands alone.

Confirm the starting point:

```bash
ssh root@149.28.124.225 "curl -sf -m10 http://10.2.122.10:8100/health"
# {"active_workers":2,"status":"ok"}

# driver must carry SOLO_M1 (patches/0005) or the run measures nothing useful
ssh root@149.28.124.225 "grep -c SOLO_M1 /mnt/vast/c_huggingface/bench_20260801/agbench/agent/agent_throughput.py"
# 8
```

---

## 1. ⚠️ Apply the TWO script patches — MANDATORY

**Without these the run is invalid.** Neither failure is visible at runtime;
both produce a clean 105-sample run of the wrong deployment. See
`patches/0006-ep-decouple-from-dpa.md` and `patches/0007-dpa-passthrough.md`.

```bash
W=/mnt/vast/c_huggingface/bench_20260801
scp scripts/patch_leg_epsize.py root@149.28.124.225:$W/scripts/

ssh root@149.28.124.225 "cd $W/scripts && \
  cp glm52_leg.sh glm52_leg.sh.bak_dpaon_\$(date -u +%Y%m%d-%H%M) && \
  python3 patch_leg_epsize.py glm52_leg.sh"
# expect: patched OK - EP_DECOUPLE occurrences: 1
```

Patch 0007 (`start_leg.sh` hardcodes `DPA=1`) — apply by hand, or copy this
kit's already-patched `scripts/start_leg.sh` over it:

```bash
scp scripts/start_leg.sh root@149.28.124.225:$W/scripts/start_leg.sh
ssh root@149.28.124.225 "grep -n 'DPA' $W/scripts/start_leg.sh | head -3"
# 85: DPA="${DPA:-1}"
# 91:   CTX=262144 ISL=8192 TP=8 DPA="$DPA" ...
```

**Verify patch 0006 did not change the DPA-on path** (it edits a shared line):

```bash
ssh root@149.28.124.225 "cd $W/scripts && for f in glm52_leg.sh.bak_dpaon_* glm52_leg.sh; do
  DPA=1 TP=8 ROLE=prefill DELAYER=1 PREFILL_DELAY_MS=5000 bash -c '
    DP_ARGS=()
    '\"\$(sed -n '/^DP_ARGS=()/,/^fi\$/p' \$f | grep -v '^#')\"'
    echo \"\$f : \${DP_ARGS[@]}\"'
done"
# both must list the same flags (order may differ):
#   --dp-size 8 --enable-dp-attention --ep-size 8 --enable-prefill-delayer ...
```

## 2. Relaunch the PREFILL leg only, with DPA off

```bash
ssh root@149.28.124.225 "ssh chi2879 'cd $W && \
  ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MTP=0 TAG=p7 DPA=0 \
  bash scripts/start_leg.sh'"
```

`TAG=p7` gives a fresh log file, so fault greps cannot pick up the DPA-on run's
lines. Cold start is 5–8 min and is **not** a hang.

### 2a. ⚠️ Read back the LIVE COMMAND LINE — do not trust the launcher's echo

This is the check that caught patch 0007. The launcher prints a success line
either way.

```bash
ssh root@149.28.124.225 "ssh chi2879 'ps -eo pid,lstart,cmd | grep \"[s]glang.launch_server\" | head -1'"
```

Required:

- contains **`--ep-size 8`**              ← patch 0006 worked
- contains **no `--dp-size`**, **no `--enable-dp-attention`**  ← patch 0007 worked
- contains `--chunked-prefill-size 8192`

Then confirm in the engine log:

```bash
ssh root@149.28.124.225 "ssh chi2879 'strings $W/logs/p7_prefill.log \
  | grep -aE \"DSA with TP mode|max_total_num_tokens|ready to roll\" | tail -3'"
# DSA with TP mode is active, dp_size=1, tp_size=8, attn_tp_size=8 ...
# [TP0 EP0] max_total_num_tokens=3263680 ... max_running_requests=2048
#           ^^^^^^ note: TP0, not "DP0 TP0" -- one scheduler, not eight
# The server is fired up and ready to roll!
```

## 3. Verify the deployment, and that decode was NOT disturbed

```bash
ssh root@149.28.124.225 "curl -sf -m10 http://10.2.122.10:8100/v1/workers"
# prefill must show "dp_size": null   (collapses to 1 routing target -- expected)
# decode  must show "dp_size": 8

# decode PID must be UNCHANGED from before the relaunch
ssh root@149.28.124.225 "ssh chi2867 'ps -eo pid,lstart | grep -m1 2420132'"

# and still carry P1V3 in the LOADED module (not the file on disk)
ssh root@149.28.124.225 "ssh chi2867 'docker exec bench_run python3 -c \"
import sglang.srt.layers.attention.dsa.dsa_indexer as m, inspect
print(\\\"P1V3:\\\", inspect.getsource(m).count(\\\"GLM52_P1V3\\\"))\"'" 2>/dev/null | tail -1
# expect: P1V3: 3
```

Functional smoke test through the router (**never probe a leg's own port — it
hangs**):

```bash
ssh root@149.28.124.225 "curl -sf -m180 http://10.2.122.10:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"glm5.2-mxfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],
       \"max_tokens\":24,\"temperature\":1.0,\"top_p\":0.95}'"
```

Coherent text = cross-node PD transfer is working. Use `temperature 1.0` /
`top_p 0.95` from the model's own `generation_config.json`: **`temperature: 0`
with MTP is indistinguishable from KV corruption.**

## 4. The run — 37 min

```bash
ssh root@149.28.124.225 "ssh chi2879 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'" \
  > $W/results/solo_dpaoff.kvd_before.json

ssh root@149.28.124.225 "cd $W && TAG=dpaoff RAMP=400 SUSTAIN=1800 \
  setsid bash scripts/solo_run.sh < /dev/null"
```

Watch (`In-flight` must never read 2):

```bash
ssh root@149.28.124.225 'tail -c 300 '$W'/logs/solo_dpaoff.log | tr "\r" "\n" | tail -2'
```

**Abort live if** `In-flight` shows 2 or the driver prints `Hit max_inflight`.

## 5. Collect

```bash
ssh root@149.28.124.225 "ssh chi2879 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'" \
  > $W/results/solo_dpaoff.kvd_after.json
# artifacts: $W/results/solo_dpaoff/solo_dpaoff/<timestamp>/{summary,metadata}.json metrics.jsonl
```

Two gates that must both pass:

```bash
# (a) the safety valve never fired
ssh root@149.28.124.225 "grep -c 'Hit max_inflight' $W/logs/solo_dpaoff.log"   # want 0

# (b) no engine faults DURING THE WINDOW. Scope by timestamp -- these logs are
# appended all day. Use `strings`, never plain grep: they contain binary bytes.
for n in "chi2879 p7_prefill" "chi2867 p6_decode"; do set -- $n
  ssh root@149.28.124.225 "ssh $1 'strings $W/logs/$2.log \
    | grep -E \"^\\[2026-08-02 0[56]:\" \
    | grep -icE \"Expected lengths.size|Aborted|OUT_OF_RESOURCES|Traceback|exception\"'"
done   # want 0 0
```

> **Two known false positives** — do not chase either:
> `server_args=` contains the substring `abort_on_priority_when_disabled`, and
> the **boot-time** disaggregation-warmup line contains `'num_retractions': 0`,
> which matches a `retract` pattern. Scoping by timestamp removes the first;
> the second is inside the window and must be read, not counted. Real fault
> count for this run: **0 on both legs.**

## 6. Reproduce the analysis

Every table recomputes from the raw arrays; nothing is read from `summary.json`.

```bash
python3 scripts/solo_analyze.py results/metrics.jsonl.gz --phase sustain
python3 scripts/compare_dpa.py          # no args: baseline is bundled in results/
```

---

## Expected result

| | |
|---|---|
| duration | 2,200 s (ramp 400 + sustain 1,800) |
| requests | 145 total, **105 in the measured window** |
| errors | **0** |
| in_flight distinct values | **{0, 1}** — never 2 |
| duty cycle | 95.2 % |
| TTFT p50 / p90 / p99 | **778** / **1,536** / 11,185 ms |
| TPOT p50 | 11.17 ms (min 7.43) |
| E2E p50 | **3,751 ms** |
| paired speedup vs DPA-on | **2.01× median, 49/50 faster** |
| MTP acceptance | 2.028 (engine `accept len` mean 2.834 agrees) |
| kvd | **+432 gets** (100 % hits), +11,076 sets, +10,249 evictions |
| aggregate KV | 3,263,680 tok (vs baseline 22,639,616) |

## If it doesn't reproduce

- **Command line still shows `--enable-dp-attention`** → patch 0007 not applied.
  This is the failure mode that produces a *perfect-looking* run of the wrong
  config. Always do step 2a.
- **`max_total_num_tokens` ≈ 2.8 M and ranks log as `DP0 TP0`** → same problem.
  DPA-off must show one scheduler and ~3.26 M.
- **TTFT barely improves** → check `--ep-size 8` is present. If `ep_size`
  collapsed to 1 (patch 0006 missing) you changed MoE too and the numbers are
  not comparable to the baseline.
- **Decode dies at 2–13 min on `Expected lengths.size(0) == B`** → the decode
  leg got restarted and lost `GLM52_P1V3`. Re-apply `scripts/apply_p1v3.py`.
- **`In-flight: 2`** → the workload file was not the one loaded; `solo_run.sh`
  deliberately passes no load knobs on the CLI.
