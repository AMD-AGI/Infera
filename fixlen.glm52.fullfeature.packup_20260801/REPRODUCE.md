# Reproduction kit

From two clean nodes to the eight numbers in `results/fixlen_summary.csv`.

**Estimated time ≈ 2 h 15 m**: ~35 min bring-up (two cold starts at 3–9 min each, plus
the feature-proof matrix), ~1 h 25 m for the eight rounds, the rest checks.

Cold start is **not a hang** — see `notes.md` §2.

## 0. Prerequisites

**Machines.** Two nodes, 8 × MI355X (gfx950), same ionic RoCE fabric:

| role | host | data-plane IP |
|---|---|---|
| prefill + etcd + router + kvd | `chi2879` | 10.2.122.10 |
| decode | `chi2867` | 10.2.122.44 |

Substitute your own hosts and IPs — they are `MY_IP` / `ETCD_IP` arguments, never
hardcoded inside a leg script. **`MC_GID_INDEX` is discovered per node**, so you do not
need to know your fabric's index in advance (that is `patches/0001`).

**Access.** Jump host `root@149.28.124.225`, then `ssh <node>`. Key-based; arrange your
own (`environment.md` §Secrets). Commands below run **on the node** unless stated.

> **The slurm hold on these nodes belongs to `yeandy-debug`, not us.** Kill only your own
> containers and sglang processes. **Never `scancel`.**

**Image.** `infera/engine-sglang:merged-e` present on both nodes. If absent, build it
from branch `yihou.dev.glm52.merged.experiment` per
`liying_rest_pr56.packup_20260801/REPRODUCE.md` §5 (~15 min/node, build on each node —
do not ship a tarball).

**Model.** `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` on shared VAST, mounted `/mnt/vast`.

**Disk.** ≥ 250 GB free on each node's `/`. Check `df -h /` **before** starting — kvd's
L3 tier lands there, and a full root disk makes every `docker exec` fail with a message
that reads like a docker bug (`notes.md` §5).

## 1. Stage this kit onto the shared filesystem

From your workstation, in this packup:

    J=root@149.28.124.225
    tar czf /tmp/bench_scripts.tgz scripts
    ssh $J 'mkdir -p /mnt/vast/c_huggingface/bench_20260801 && cat > /tmp/bench_scripts.tgz' < /tmp/bench_scripts.tgz
    ssh $J 'cd /mnt/vast/c_huggingface/bench_20260801 && tar xzf /tmp/bench_scripts.tgz && chmod +x scripts/*'

Both nodes read `/mnt/vast`, so this stages once for both.

## 2. Reset both nodes (parallel, ~1 min each)

**Do not skip the reset.** It tears down the container and engines, *waits for the GPUs
to go idle*, starts a fresh container, and verifies 8 `PORT_ACTIVE` before anything
else. Each skipped step maps to a concrete failure (`notes.md` §4).

    W=/mnt/vast/c_huggingface/bench_20260801
    # chi2879 (prefill; ETCD=1 also starts etcd here)
    ssh $J "ssh chi2879 'ROLE=prefill MY_IP=10.2.122.10 ETCD=1 bash $W/scripts/reset_node.sh'"
    # chi2867 (decode)
    ssh $J "ssh chi2867 'ROLE=decode  MY_IP=10.2.122.44 bash $W/scripts/reset_node.sh'"

Both must end with `===== node <host> ready for <role> (image=…:merged-e, unpatched) =====`
and must have printed:

    GPUs idle after Ns
    PORT_ACTIVE: 8 (want 8)
    kvd socket OK
    etcd up            # chi2879 only

`PORT_ACTIVE: 8` is load-bearing: a failed libionic injection drops mooncake to TCP and
the run measures nothing.

## 3. Boot both legs at the frozen config (~6 min, parallel)

One server serves **all eight rounds** and Case A. It is sized for the largest pair
(ISL 155,000 + OSL 3,300) and then frozen — no per-workload retuning.

    ssh $J "ssh chi2879 'ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MTP=0 TAG=p2 bash $W/scripts/start_leg.sh'"
    ssh $J "ssh chi2867 'ROLE=decode  MY_IP=10.2.122.44 ETCD_IP=10.2.122.10 MTP=1 TAG=p2 bash $W/scripts/start_leg.sh'"

Each prints its resolved config, e.g. `mtp=1 ctx=262144 gmu=0.85`. MTP is **decode-only**
(`MTP=1` also implies `--disable-custom-all-reduce`; the aiter custom all-reduce kernel
deadlocks on gfx950 during EAGLE verify).

Poll the HTTP endpoint — **never grep the log for readiness**, it matches a previous
run's line within seconds (`notes.md` §2):

    ssh $J "ssh chi2879 'docker exec bench_run curl -sf -m5 http://10.2.122.10:30000/health'"
    ssh $J "ssh chi2867 'docker exec bench_run curl -sf -m5 http://10.2.122.44:30000/health'"

Prefill takes ~4 min, decode ~8 min (MTP adds a second EAGLE draft-graph capture pass).

**Confirm the GID discovery worked** — this is the one that killed the first attempt:

    ssh $J "ssh chi2867 'strings $W/logs/p2_decode.log | grep -oE \"Using user-specified GID index: [0-9]+\" | sort -u'"
    # chi2867 -> index 2 (NOT 1). chi2879 -> index 1.
    ssh $J "ssh chi2867 'strings $W/logs/p2_decode.log | grep -c \"Mooncake Transfer Engine initialization failed\"'"
    # -> 0

## 4. Start the Rust router

    ssh $J "ssh chi2879 'BACKEND=rust bash $W/scripts/start_router.sh'"
    # -> {"active_workers":2,"status":"ok"}  router healthy (backend=rust pw=20.0 dw=2.0)

Rust is deliberate: it is the only backend where group E's bigram kv-event fix is live,
and MTP is exactly what makes SGLang emit bigram `BlockStored` token_ids. Two Rust deltas,
both known and neither a blocker: no `--kvd-socket-path` (so no `/v1/cache/prewarm`), and
no `/v1/admin/cache-view` route. `BACKEND=python` is the fallback.

The module is `python -m infera.server`, **not** `infera.router` (`notes.md` §7).

## 5. Feature-proof gate — all six rows, before spending a measured window

A green run that proves nothing is the default outcome here. Every row below goes red if
its feature is absent.

    ssh $J "ssh chi2879 'TAG=p2 bash $W/scripts/feature_proof.sh'"
    ssh $J "ssh chi2879 'docker cp $W/scripts/workers.py bench_run:/tmp/ ; docker exec bench_run bash -c \"curl -s http://10.2.122.10:8100/v1/workers | python3 /tmp/workers.py\"'"

| # | feature | expected |
|---|---|---|
| 1 | PD | both `prefill` and `decode` `active`, `dp_size=8` |
| 2 | DPA | `ps -eo args \| grep -c '^[s]glang::scheduler_DP'` = **8** on each node |
| 3 | RDMA | mooncake init failures **0**, `MC_FORCE_TCP` **0** |
| 4 | MTP | decode `accept len` **2.1–2.6** — see below |
| 5 | kvd | restart-replay: gets/hits climb, `sets` **flat** — §6 |
| 6 | kv-aware | per-rank picks with `cache_hits` > 0 — §7 |

For row 2, use `ps -eo args`, not `pgrep -f` — the latter matches its own shell and
reports 9.

**Row 4 — `accept len` is read after traffic, and 4.00 is BAD news**, not good: it means
the draft model is perfectly predicting a repetition loop. Healthy is 2.1–2.6:

    ssh $J "ssh chi2867 'strings $W/logs/p2_decode.log | grep -oE \"accept len: [0-9.]+\" | tail -8'"

## 6. Correctness + the kvd attribution test

    ssh $J "ssh chi2879 'docker exec bench_run python3 /tmp/probe.py http://10.2.122.10:8100 glm5.2-mxfp4'"   # -> 4/4
    ssh $J "ssh chi2879 'docker exec bench_run python3 /tmp/prefix_reuse.py'"                                  # -> 16/16 + 16/16
    ssh $J "ssh chi2879 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'"       # -> sets 102, gets 0

`gets: 0` is **correct** here — the in-GPU radix cache is serving.

**The attribution test.** A latency win proves nothing: sglang's radix cache serves a
repeated prefix without touching L3. Restarting the prefill engine empties that cache
while the kvd daemon and its L3 keep running, so any reuse afterwards can only be L3:

    ssh $J "ssh chi2879 'TAG=p2 bash $W/scripts/restart_replay.sh'"    # ~190 s
    ssh $J "ssh chi2879 'docker exec bench_run python3 /tmp/prefix_reuse.py'"
    ssh $J "ssh chi2879 'docker exec bench_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock'"

**Expect `gets 102, hits 102, sets 102 (unchanged), misses 0`.** `sets` staying put is
the load-bearing part: reads, not re-writes.

## 7. kv-aware per-DP-rank proof

The Rust router has no `/v1/admin/cache-view`, so the signal is its policy log line
(`rust/router/src/policy.rs:314`), parsed by `cache_view.py`:

    ssh $J "ssh chi2879 'docker cp $W/scripts/cache_view.py bench_run:/tmp/ ; docker exec bench_run python3 /tmp/cache_view.py'"

**Read it after driving traffic** — a freshly restarted router reports nothing, which
looks exactly like the fix having failed (`notes.md` §6). After §6 you will see picks
across `#dp0`…`#dp7` on both roles, prefill `cache_hits` non-zero.

> A single-prefix probe legitimately concentrates on one rank. Only under sweep
> concurrency (§8) do all 8 ranks appear — see `notes.md` §8.

## 8. The eight rounds (~1 h 25 m)

Sequential by necessity: all eight share **one** server, so two at once would measure
each other. Run detached on the node so a dropped SSH does not kill the sweep:

    ssh $J "ssh chi2879 'nohup bash $W/scripts/sweep.sh > $W/logs/sweep.log 2>&1 &'"

Default `ROUNDS="p50:32 p50:64 p50:128 p90:1 p90:32 p90:64 p90:128"`; run `p50:1` first
with `PAIR=p50 C=1 bash $W/scripts/fixlen_round.sh`, or set `ROUNDS` to the full eight.

Poll (do not hold a long foreground SSH):

    ssh $J "ssh chi2879 'grep -E \"^### \" $W/logs/sweep.log | tail -2'"
    ssh $J "ssh chi2879 'strings $W/logs/p3_prefill.log | grep -c HSA_STATUS_ERROR'"   # must stay 0

Each round writes `results/fixlen_<pair>_c<C>.{jsonl,log}` plus kvd before/after
snapshots. The tqdm bar only advances on **completion**, so at conc ≥ 32 it can sit still
for minutes while everything is in flight — check the engine log's `#running-req` /
`#queue-req` before concluding it is stuck (`notes.md` §9).

Per-round settings, for reference:

| pair | ISL | OSL | conc | num-prompts |
|---|---:|---:|---|---:|
| p50 | 74,000 | 320 | 1 / 32 / 64 / 128 | 8 / 64 / 128 / 256 |
| p90 | 155,000 | 3,300 | 1 / 32 / 64 / 128 | 4 / 32 / 64 / 128 |

`--random-range-ratio 1.0` pins every prompt to exactly ISL (a fixlen sweep wants a
delta, not a distribution). Sampling is the model's own `generation_config` — temp 1.0 /
top_p 0.95, **never** temperature 0 (`notes.md` §1).

## 9. Collect and summarise

    ssh $J "ssh chi2879 'cd $W && tar czf /tmp/res.tgz results logs/sweep*.log'"
    ssh $J 'cat /tmp/res.tgz' > /tmp/res.tgz && tar xzf /tmp/res.tgz

Then regenerate `results/fixlen_summary.csv` with the extractor in `scripts/extract_results.py`.

## 10. What "reproduced" means here

All of these, or you have not reproduced it:

| check | expected |
|---|---|
| reset, both nodes | `PORT_ACTIVE: 8`, `kvd socket OK` |
| GID discovery | chi2879 → 1, chi2867 → **2**; mooncake failures **0** |
| feature matrix | all six rows green |
| correctness | 4/4 and 16/16 + 16/16 |
| **kvd restart-replay** | **gets 102, hits 102, sets unchanged, misses 0** |
| **kv-aware** | **all 8 DP ranks picked; prefill `cache_hits` non-zero** |
| MTP | `accept len` 1.5–3.7 (**not** 4.00) |
| rounds | **8/8 completed, 100 % success in each** |
| `HSA_STATUS_ERROR` after the gmu fix | **0** |
| throughput shape | p50 saturates ~47.6K tok/s at c128; p90 ~15.5K |

The two bolded rows are the ones that would go red if a fix were absent or wrong.
Everything else can pass for uninteresting reasons.

## If it doesn't reproduce

`notes.md`, ordered by how likely each trap is to bite. The three that cost the most
time on this run:

1. **`MC_GID_INDEX` is node-dependent** (§3) — decode dies at init on all 8 ranks.
2. **Prefill HSA OOM at long ISL is an *activation* problem** (§4) — lower
   `mem-fraction-static`, do not raise it.
3. **Editing a script on `/mnt/vast` does not change what the container runs** (§10) —
   the leg script is `docker cp`'d in, and the stale copy silently ran for a full cold
   start after its own fix.
