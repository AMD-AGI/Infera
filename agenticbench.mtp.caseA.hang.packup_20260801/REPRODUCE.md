# REPRODUCE — the hang, and the control arm that settles its cause

Two procedures. §1–5 reproduce the hang. **§6 is the MTP-off control arm, which
was specified but NOT run** — it is the single most valuable next step and is
written to be picked up cold.

Prerequisite: the deployment from
`agenticbench.mtp.sweep.packup_20260801/REPRODUCE.md` §0–8 (two held nodes, the
image built on each, both legs booted, gate passed). Everything below assumes
that kit's bring-up and reuses its scripts.

    export PJOB=<prefill job>  PIP=<prefill ens3 ip>
    export DJOB=<decode job>   DIP=<decode ens3 ip>

## 1. Boot the two legs — MTP ON the decode leg

```bash
bash scripts/boot.sh prefill 262144 1 0 armA    # ctx, kvd=1, mtp=0
bash scripts/boot.sh decode  262144 0 1 armA    # ctx, kvd=0, mtp=1
bash scripts/wait_ready.sh 1800
bash scripts/router.sh 8190
```

**Restart BOTH legs together, always.** Rebooting one while the other keeps
running orphans the survivor's c10d/TCPStore state; the survivor then dies of an
NCCL heartbeat failure minutes later. That is exactly how attempt 3 was lost —
see `notes.md` §3.

Verify the arm is the one you think it is:

```bash
strings <log> | grep -oE "speculative_algorithm='[A-Z]+'|disable_custom_all_reduce=[A-Za-z]+"
```

Expect on decode: `speculative_algorithm='EAGLE'` **and**
`disable_custom_all_reduce=True`. The second is not optional — see §7.

## 2. Calibrate (900 s, and it passes — this is the control that matters)

```bash
bash scripts/run_bench.sh probe caseA_probe
```

Expected, and observed: **509 completed, 3 errors**, live sessions p50 20,
in-flight p50 13 / max 19 against a cap of 48, **0 ticks at cap**, cache hit
0.8818. A clean, unsaturated run.

That the probe passes is important: it shows the hang is not a bring-up defect
and not present at 900 s of the same traffic shape.

## 3. Run Case A (this is where it dies)

```bash
bash scripts/run_bench.sh full caseA_full     # ramp 400 + sustain 3600
```

`--dashboard-mode` is set inside the script and is **mandatory** — without it
`summary.json` / `metrics.jsonl` / `metadata.json` are never written
(`agent_throughput.py:1674`), and the freeze would leave no evidence at all.

**Watch for the signature.** It is not gradual:

| t | in-flight | completions |
|---|---|---|
| 0–556 s | oscillating **8–29**, 0 ticks at cap | growing, reaches **401** |
| ~556 s | — | **freezes at 401, permanently** |
| 556–900 s | ratchets 29 → 38 → 42 → **48**, sticks | still 401 |

Distinguishing this from saturation, which matters because they look alike in
the driver's one-line status: **a saturated server keeps completing requests,
just slowly.** Completions going to exactly zero while in-flight climbs is a
hang. Confirm from the server side:

```bash
strings <prefill.log> | grep -oE "token usage: [0-9.]+" | tail -30   # ~0.03
strings <prefill.log> | grep -oE "#queue-req: [0-9]+"  | tail -30   # ~0.7
```

3 % KV usage with an empty queue while the client reports 48 in flight is the
proof that the load generator is not the constraint and the server is not busy.

## 4. Capture the evidence BEFORE restarting anything

The hung state is the whole finding, and a restart destroys it.

```bash
# both legs: is it hung or dead?
docker exec <ctr> curl -sf -m10 http://$DIP:30001/health     # times out => hung
docker exec <ctr> curl -sf -m10 http://$PIP:30000/health

# spinning schedulers => busy-wait in a collective, not idle
docker exec <ctr> ps aux --sort=-pcpu | grep scheduler_DP | head -4

# the decisive artefact: every rank's Python stack
docker exec <ctr> pip install py-spy -q
for P in $(docker exec <ctr> pgrep -f scheduler_DP); do
  docker exec <ctr> py-spy dump --pid $P > hang_stacks/dp_pid$P.txt
done
```

**Read the odd rank out.** Do not just note "everyone is in `all_gather`" — find
the one that is *not*. Here 7 ranks sat in
`prepare_mlp_sync_batch → all_gather_into_tensor` and **DP1** sat in
`pop_preallocated → mooncake/conn.py:1923 send_metadata → zmq send`. The
minority rank is the lead.

Then establish causal order across the legs by timestamp:

```bash
strings <prefill.log> | grep -nE "TCPStore|HeartbeatMonitor|recvValue failed" | head
strings <decode.log>  | tail -3
```

Here prefill's `TCPStore recvValue failed` at **12:22:06** precedes decode's last
line at **12:22:08** — which is what makes prefill the first failure, not decode.

## 5. Recovering

```bash
# bracketed patterns -- a bare `pkill -f infera.engine.sglang` matches the
# bash -c command string containing it and kills its own shell (notes.md §1)
docker exec <ctr> pkill -9 -f "[s]glang.launch_server"
docker exec <ctr> pkill -9 -f "[i]nfera.engine.sglang"
# poll every GPU to VRAM 0% -- release is asynchronous
# then reboot BOTH legs (§1), not just the dead one
```

---

## 6. THE CONTROL ARM — MTP off, everything else identical (NOT YET RUN)

This is what turns "MTP is the leading suspect" into a result. It is a
**single-variable** comparison against §1–3: same image, same rate, same ctx,
same kvd/kvaware wiring, same `--disable-custom-all-reduce`.

```bash
# both legs, MTP=0 on decode -- the ONLY change from §1
bash scripts/boot.sh prefill 262144 1 0 armB
bash scripts/boot.sh decode  262144 0 0 armB     #   ^ mtp=0
bash scripts/wait_ready.sh 1800
bash scripts/router.sh 8190
```

Confirm the single variable actually held:

```bash
strings <armB_decode.log> | grep -oE "speculative_algorithm=[^,]*|disable_custom_all_reduce=[A-Za-z]+"
# want: speculative_algorithm=None   AND   disable_custom_all_reduce=True
```

`disable_custom_all_reduce=True` on the MTP-**off** arm is the part that is easy
to get wrong — see §7.

Then the identical workload, unchanged:

```bash
bash scripts/run_bench.sh full caseA_armB_mtpoff    # workloads/caseA_full.yaml, rate 0.15
```

### Reading the outcome

| result | conclusion |
|---|---|
| completes 4,000 s | **the hang is MTP-attributable** on this workload — and Case A data for the MTP-off configuration falls out for free |
| hangs the same way | MTP is **not** the variable; suspicion moves to the mooncake `send_metadata` path under sustained long-context PD, independent of speculation |
| hangs differently | record both signatures; the shared factor is the lead |

Either way the run is worth its 70 minutes: one outcome gives a cause, the other
eliminates the leading hypothesis and delivers a completed Case A.

### If it hangs too — the third arm

`exp2_indexshare_off` showed PD + DPA + MTP working without patches 2/4 by
turning IndexShare off. This deployment runs it **on** (model default). Try:

```bash
EXTRA_ARGS="--json-model-override-args '{\"index_share_for_mtp_iteration\":false}'" \
  bash scripts/boot.sh decode 262144 0 1 armC
```

Mitigation, not diagnosis — but it is the one configuration a sanctioned kit has
already seen survive this bug class.

---

## 7. Do not remove `--disable-custom-all-reduce`, and do not tie it to MTP

Checked against four sanctioned kits (`exp1_patch1_v2`, `exp2_indexshare_off`,
`exp3_merged`, `exp3b_patch4_32209`): **8 of 8 legs ran
`disable_custom_all_reduce=True`**. Every "MTP is fixed" PASS in this repo rests
on it being on, and `main_converged` records that the custom all-reduce path
`remains unexercised`.

It guards an **aiter kernel** defect on gfx942/gfx950 (sglang #28815 / #31071 /
PR #31478) — a different defect class from the DSA/DP-attention/draft-graph
rank-divergence bugs the branch patches fix. Fixing those does not make that
kernel safe.

`scripts/glm52_leg_spur_mtp.sh` therefore passes it on **both** arms:

```bash
CUSTOM_AR="${CUSTOM_AR:-0}"        # independent of MTP, by design
```

It previously followed MTP, which would have silently re-enabled the broken
kernel on the MTP-off arm and made §6 a two-variable comparison. The converged
kit had already removed that exact trap from its leg script; it was reintroduced
here and is now fixed.

## 8. What "reproduced" means

| check | expected |
|---|---|
| probe (900 s) | passes: ~509 completed, in-flight ≤ ~19, 0 at cap |
| Case A, first ~556 s | healthy: completions growing, in-flight 8–29, cache ≈ 0.89 |
| the freeze | completions stop at a fixed number and never advance |
| in-flight afterwards | ratchets monotonically to the cap and sticks |
| server-side | prefill KV usage ~3 %, queue ~0.7 — **idle, not saturated** |
| decode `/health` | times out |
| schedulers | ~105 % CPU each (busy-wait) |
| py-spy | 7 ranks in `all_gather`, **1 rank elsewhere** |
| `Traceback` / `Memory access fault` / `Scheduler hit an exception` | **0** everywhere |

The last row is why this is hard to see coming: nothing anywhere reports an error.
