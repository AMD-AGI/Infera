# Pitfalls and wrong turns — Bug 2b round

Each entry: what happened / why / how it was caught or avoided / context.

---

## P1 — Voting on the wrong process group would have been a silent no-op

**What.** The first draft of the fix used `get_attention_tp_group()`, which was already
imported in the file and looked like the obvious choice.

**Why wrong.** Under DP-attention with `dp_size == tp_size == 8`, the *attention* TP group
is `attn_tp_size = tp/dp = 1` rank wide. An all-reduce over a 1-rank group returns the
local value unchanged, so the fix would have compiled, run, added a collective, and
changed nothing — while appearing to work on any run that happened not to hit the race.

**How caught.** Read `dp_attention.py:331-341` before running, and traced which group
`dp_attn.py` itself all-gathers the MLP-sync info over (`get_tp_group()`, line 91).

**Context.** This is the single most dangerous class of bug in this investigation: a
change that is inert but not obviously inert. Three prior "fixes" were falsified; at least
one of them looked plausible for the same reason.

---

## P2 — `grep -q eh_proj` gave a false positive and cost a boot cycle

**What.** Before launching the mix regression server, I checked whether the nextn patch
was applied with `grep -q eh_proj deepseek_nextn.py`. It matched, so I skipped the patch.
The server then died at weight load with the documented
`size of tensor a (3072) must match ... b (6144)` crash.

**Why.** `eh_proj` appears in that file in several unrelated places. The patch changes a
*specific* line — `ckpt_prefix = f"model.layers.{n}"` → `...{n}.eh_proj"` — so the correct
probe is `grep -c 'num_hidden_layers}.eh_proj'`, not a bare substring.

**How caught.** The boot failed. ~9 minutes lost.

**Lesson.** An idempotency check must test for the *exact* post-patch text, not a token
that also exists pre-patch. All the `fix_*.py` scripts here use a unique marker for this
reason; the ad-hoc shell grep did not.

---

## P3 — Backgrounding the router inside `spur exec` killed it instantly

**What.** Launched the router with `nohup ... &` inside `spur exec`. It never came up;
even its log file did not exist.

**Why.** The `spur exec` namespace is torn down when the command returns, taking any
backgrounded child with it. This is documented in CLAUDE.md and I did it anyway.

**Fix.** `docker exec -d` so the process is owned by the container, not the exec session.

---

## P4 — The router's open circuit masqueraded as a persisting hang

**What.** Immediately after the fix was applied and the decode leg rebooted healthy, the
probe still returned `0/4` — all four failing in **0.4 s** with HTTP 503.

**Why.** The router had opened its circuit breaker during the *previous* (unfixed) run's
deadlock and never re-closed it. The 503 was the router refusing to dispatch, not the
decode leg failing.

**How distinguished.** The timing. A real deadlock times out at the client limit (120 s);
a circuit-breaker 503 returns in under a second. Restarting the router on a fresh port
gave 4/4 immediately.

**Lesson.** Always restart the router between arms, on a fresh `--port` *and*
`--prometheus-port`, and read the failure *latency* before concluding anything.

---

## P5 — `analyze.py`'s `final` column does not verify the fix

**What.** After applying the fix, the guard analyzer still reported "final diverges on 4
iterations", which reads like the fix failed.

**Why.** That analyzer reconstructs the **pre-fix** decision from the four raw terms — by
design, for diagnosing. After the fix the raw terms still diverge (they are inherently
rank-dependent); what becomes uniform is the *voted* value the code acts on, which that
probe never sees.

**Fix.** A second probe (`probe_voted.py`) logging `local` vs `voted` immediately after
the all-reduce, plus `analyze_vote.py` to check uniformity of the acted-on value.

**Lesson.** An instrument built to characterize a bug is usually the wrong instrument to
validate its fix. Nearly recorded a false negative here.

---

## P6 — A comment-only marker cannot be verified in bytecode

**What.** `verify_pyc.sh` reported `pyc count: 0` for the marker `GLM52_BUG2B_UNIFORM`,
which looks exactly like the stale-`.pyc` failure that invalidated an earlier experiment.

**Why.** The marker appears only inside a `#` comment, and comments are discarded by the
compiler. The *code* was present.

**Fix.** Verify with a token that survives compilation — a variable name
(`_needs_eager_local`) rather than a comment tag.

**Lesson.** Patch markers intended for bytecode verification must be identifiers or
string literals.

---

## P7 — Warmup passing means nothing (re-confirmed)

The PD warmup is an 8-way concurrent burst, one request per DP rank — so **all** ranks are
busy and the guard terms agree. Every configuration in this round, broken or fixed,
cleared warmup. The failure needs *partial* occupancy, which only post-warmup single
requests produce.

Measured: during warmup, t2 and t4 both diverged but **cancelled**, leaving the decision
uniform 7-graph/1-eager on every rank.

---

## P8 — Instrumentation had to be written to survive the graph blind spot

A replayed CUDA graph executes **no Python**, so any probe placed inside the draft forward
is invisible on the graph arm. An earlier session drew a false conclusion from this
("0 all-gathers on the graph path") and had to retract it.

This round's probes are all placed on the **host decision path**, before the branch, so
they fire identically whichever arm is taken. That is why the it=9 record exists for all
8 ranks including the ones that then entered the graph.

---

## P9 — "Catastrophic regression" that was a dead endpoint

**What.** The 8-round durability run reported **0/128 on every round**. It looked like the
fix had a fatal soak-time defect.

**Why.** The run was pointed at router port 8104, whose *prefill* leg no longer existed —
I had killed the PD prefill server on that node to free GPUs for the mix regression, so
port 30000 was serving mix, not prefill. The router returned 503 for everything.

**How caught.** Checked the live process args for `disaggregation-mode`; it was absent,
so the thing answering on 30000 was a mix server. Re-running against the mix endpoint
directly gave 1024/1024.

**Lesson.** Before believing a regression, confirm that the thing under test is the thing
that is running. The failure latency was also a tell — instant 503s, not timeouts.

---

## P10 — Spur evicted both jobs right after the final run

**What.** Both jobs went `FAILED` minutes after the last measurement; `spur exec` started
returning "job is not running".

**Why.** Spur evicts without warning. Documented, expected, and it has happened before in
this project.

**Why it cost nothing this time.** Every log, result and script was written under
`/home/yihou`, which is NFS-backed and bind-mounted into the container — so all 17 MB of
raw evidence survived the eviction. Had they been written to container-local paths or
`/tmp` on the node, the entire round would have been unreproducible.

**Lesson.** On this cluster, treat node-local storage as volatile. Write everything that
matters to the NFS bind-mount as you go, not at the end.

---

## P11 — The patches are not applied at image build time

Not a mistake made here, but a trap for the next person: `Dockerfile.sglang` has no patch
loop (the previous `patches/sglang/*.py` mechanism was removed), so a stock image build
contains **none** of the four diffs. Every result in this kit was obtained by applying
them into a running container. A reproduction that builds the image and expects the fix
to be present will silently test unpatched code — which is the same class of failure as
the stale-`.pyc` trap, one layer up.

---

## P12 — A shell loop clobbered `spur`'s arguments

`for j in "11428 ip port role"; do set -- $j; spur exec $1 ...` — `set --` overwrote the
positional parameters that `spur` itself parsed, yielding
`Caused by: invalid digit found in string`. Trivial, but it wasted a cycle looking for a
cluster problem that did not exist. Wrote the calls out explicitly instead.
