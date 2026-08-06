# Patches

Three changes were needed to run this arm. **None of them is a code fix to the
product** — all three are to the *harness*, and two of them exist specifically to
stop the DPA-off arm from silently changing more than one variable.

The engine-side fixes this arm depends on (the DSA patch set, the ROCm hicache
allocator, `GLM52_P1V3`) are **baked into the image by the branch's Dockerfile**
and verified in bytecode at container start. There is no in-container patching in
this kit — see `../REPRODUCE.md` step 3. That is a change from the predecessor
kits, whose `REPRODUCE.md` had a MANDATORY hand-patch step.

---

## 0001 — leg script: EP_DECOUPLE + global chunk (+ two inherited fixes)

`0001-leg-script-ep-decouple-and-global-chunk.diff`, against
`../../agenticbench.mtp.caseA.packup_20260801/scripts/glm52_leg_spur_mtp.sh`.

Four hunks. Two are the point of this arm; two are carried along.

### Hunk A — `--ep-size` moved out of the DPA branch  ← **load-bearing**

**What.** `DP_ARGS=()` … `--dp-size N --enable-dp-attention --ep-size N` inside
`if DPA=1` becomes `DP_ARGS=(--ep-size N)` with only `--dp-size` and
`--enable-dp-attention` inside the branch.

**Why.** GLM-5.2 is a MoE. `--ep-size` selects **expert** parallelism;
`--enable-dp-attention` selects **attention** parallelism. Different axes. Gating
both on one `if` means `DPA=0` also collapses the MoE from ep8 to the TP default,
changing the expert-dispatch collective at the same moment as the attention
layout. A "DPA on vs off" comparison then differs in two things.

**How.** Applied to the working copy before the run; the diff here is the record.

**Context.** Not discovered here — the spur nodpa kit
(`../../agenticbench.mtp.nodpa.packup_20260802/notes/nodpa_design.md` §2) hit it
first and named it EP_DECOUPLE. Verified live on this run: the prefill leg's
command line shows `--ep-size 8` with no `--dp-size` and no
`--enable-dp-attention`.

### Hunk B — `--chunked-prefill-size` matched at the GLOBAL level  ← **load-bearing**

**What.** `if DPA=1 then CHUNK=ISL*TP else CHUNK=8192` becomes
`CHUNK=${CHUNK:-$((ISL*TP))}` unconditionally.

**Why.** `--chunked-prefill-size` is a **global** per-step token budget, and
sglang divides it by `dp_size` *only* when DP-attention is on
(`server_args.py:4902` — a division, not a clamp). At dp8, 65,536 resolves to
8,192 per rank but stays 65,536 machine-wide. The old `else` branch's hardcoded
8,192 is **global**, i.e. ⅛ of what the DPA arm gets. Left alone, flipping DPA
off would also have cut the global chunk 8×, and any TTFT change would be
misattributed to DP-attention.

**How.** Same as A. This arm passes **no** `CHUNK` — the default `ISL*TP` = 65,536
is exactly what it wants, and DPA then divides it to 8,192/rank. Arm B passes
65,536 explicitly and gets 65,536 (no division). **The two arms match at the
GLOBAL level and deliberately differ at the per-forward level.**

**Context.** The two reference kits **disagree** and the disagreement is recorded
rather than resolved:

| kit | cluster | concurrency | `CHUNK` | rationale given |
|---|---|---|---|---|
| nodpa | spur | 1 | **65,536** | match the *global* budget |
| par8 | vultr | 24 | 16,384 | "near the measured sweet spot" |

Neither is per-forward-matched (that would be 8,192). This run follows **nodpa**,
because it is the first-hand result *on this cluster*. The consequence — 8×
per-forward prefill work vs the vultr par8 run — is a live uncontrolled variable
in the cross-cluster row of `../README.md` and is flagged there.

### Hunk C — `INFERA_SGLANG_READY_TIMEOUT` → `INFERA_ENGINE_READY_TIMEOUT`

**What / why / how / context.** The branch under test no longer reads the old
name: main's `e190d65` generalised the knob so it also covers the vLLM worker
(`infera/engine/sglang/worker.py:39`, `infera/engine/vllm/worker.py:30`). Copying
the predecessor kit's export across would leave the 1,800 s default in force. A
408 GB checkpoint with two legs off one filesystem exceeds that, so the failure
mode would be a spurious "engine never became ready" on a cold start that was
merely slow. Inherited from the acceptance run, not new to this arm.

### Hunk D — default `LOG` path retargeted to this workspace

Cosmetic. Both legs pass `LOG` explicitly anyway (see `../env/`).

---

## 0002 — `ab_router.sh`: `--router-tokenizer-path` is unconditionally required

**What.** The first version of the script passed `--router-tokenizer-path` only
on the `kv-aware` arm, reasoning that round-robin discards it.

**Why it was wrong.** The argument is `required=True` at the parser level, not
gated on the policy. Omitting it exits **2** with a usage dump before any policy
code runs. `round-robin` genuinely does discard the value afterwards
(`_build_round_robin(**_)` drops every kwarg) — but it must be on the command
line.

**How.** Moved into the unconditional part of `POLICY_ARGS`. The **overlap
weights** stayed conditional: those really are kv-aware-only, and on a
round-robin arm they would read as an active knob in the recorded command line
when nothing consumes them.

**Context.** Hit on **this** arm — it is the round-robin one, and the first
`ab_router.sh` invocation for it exited 2 with a usage dump before the router ever
started. The kv-aware arm had been passing the flag all along and never saw it. No
`.diff` file: the script is short and shipped verbatim in `../scripts/`.

---

## 0003 — the `WARNING: engines still present after 40s` false alarm

**Not a patch.** Recorded so a reproducer does not chase it.

`ab_boot.sh` kills the previous engine tree, then waits up to 40 s for it to
disappear, then `docker exec -d`s the new one. The wait loop counts matching
processes *after* the new leg has already been started, so it reliably prints the
warning on a clean boot. Verified by timestamp: the process it was complaining
about had `lstart` equal to the launch time and carried the *new* arguments.

**Left unfixed on purpose** — this arm's scope is the A/B measurement, not
harness cleanup, and the pkill/wait discipline it implements is load-bearing (see
the bracketed-pattern comment in the script: a bare `pkill -f` matches the
`bash -c` string that contains the pattern, i.e. the shell itself).
