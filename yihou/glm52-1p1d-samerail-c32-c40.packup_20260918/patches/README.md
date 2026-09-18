# Patches

Both are required. Apply against
`AMD-AGI/Infera` @ `5a342acffe10e09729662ff40e81b22a4367fd75`
(branch `dev/pd_opt/glm_5.2_agentx`):

```bash
git apply patches/01-router-pd-dp-rank-affinity.patch
git apply patches/02-bench-pd-dp-rank-affinity-wiring.patch
```

Both were uncommitted working-tree changes at pack-up time.

---

## 01 — `router-pd-dp-rank-affinity.patch`

**What.** The `rust/router/` half of commit
[`99fa0406`](https://github.com/AMD-AGI/Infera/commit/99fa0406cc2ce3e7eedb8a3349b2626eca1339b2)
("router: constrain Decode to the Prefill DP rank under PD"), which lives only
on `origin/llying/bench/sglang_glm5p2_agentx`. Touches `config.rs`, `disagg.rs`,
`handlers.rs`, `main.rs`, `policy.rs`, `pool.rs`, `tests/functional.rs`.

The `Policy` trait gains a required `pick_at_rank(…, required_dp_rank)`; `pick`
becomes a default method over it, so unconstrained behaviour is bit-identical.
PD dispatch picks Prefill first as before, then asks Decode for that same
effective rank, returning **503 rather than silently crossing ranks**.

**Why.** In a symmetric DPA deployment the Prefill and Decode policies each pick
a rank independently, so a request routed to Prefill rank 1 can land on Decode
rank 0. Mooncake then moves KV across rails instead of along the target GPU's
local one — the failure `bench/glm5p2_pd/issue.md` §3.1 documents.

**How.** `git show 99fa0406 -- rust/ | git apply -` applies cleanly on this
branch; that is how this patch was produced. Only two `Policy` implementors
exist (`RoundRobin`, `KvEventAwarePolicy`) and both were converted, so nothing
else breaks. Validated with `cargo check --release --bin infera-router` **inside
the base image** (no local rust toolchain) — exit 0.

**Context.** Off by default; `INFERA_PD_DP_RANK_AFFINITY` turns it on. This run
produced **0** affinity-503s, meaning every Decode pick found its matching rank.

## 02 — `bench-pd-dp-rank-affinity-wiring.patch`

**What.** The bench half of the same commit, hand-ported. Adds
`PD_DP_RANK_AFFINITY="${PD_DP_RANK_AFFINITY:-1}"` to `config.sh` and
`config.full.sh`, and in `launch.sh` translates `1`/`0` to `true`/`false` and
passes `-e "INFERA_PD_DP_RANK_AFFINITY=…"` into the router `docker run`.

**Why — do not skip this one.** On this branch the router container receives
**zero `-e` flags**. Without this patch the rust change in 01 is dead code: the
feature stays off and the run silently does the cross-rail thing it was built to
prevent.

**How.** Hand-ported rather than applied, because the upstream commit expects
three helpers this branch does not have (`bool01`, `log`, `ssh_exec`; this branch
uses a plain `[[ == 1 ]]` test, `echo`, and `ssh_run`), and because it patches
`config.sh.example`, which has since been renamed to `config.sh`. Verified with
`bash -n` plus a both-ways check of the translation.

**Context — the type trap.** The router's clap uses `ArgAction::Set`, so
`INFERA_PD_DP_RANK_AFFINITY` accepts only `true`/`false` and **rejects `1`/`0`**.
The bench knob is numeric. Passing the bench value straight through fails; the
translation is the point of the patch.

---

## Not a patch: the same-rail NIC pinning

The other half of "same rail" needs **no code change** — it is configuration,
and lives in `scripts/config.yihou.sh`:

```bash
RDMA_DEVICE='{"0":"ionic_2","1":"ionic_3","2":"ionic_4","3":"ionic_5"}'
MC_TE_FILTERS=ionic_2,ionic_3,ionic_4,ionic_5
```

`sglang`'s `--disaggregation-ib-device` already accepts a per-GPU JSON map; the
stock shared-list form is what leaves HCA choice to auto-discovery. Full
reasoning in `notes.md` §1.
