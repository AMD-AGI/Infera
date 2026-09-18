# Patches

Two, with different status. Only the first is needed to reproduce the **fix**;
the second is needed to reproduce the **rounds**, because two of the four carry
it.

---

## `0001-engine-sh-extra-args-env.patch`

**What.** Adds role-scoped escape hatches to `bench/glm5p2_pd/engine.sh`:
`PREFILL_EXTRA_ARGS` / `PREFILL_EXTRA_ENV` and the `DECODE_` equivalents.
`_EXTRA_ARGS` items are word-split and appended to the engine command line;
`_EXTRA_ENV` items are validated against `^[A-Za-z_][A-Za-z0-9_]*=.*$` and passed
as `-e` to `docker run`. Both default to empty, so the patch is a **no-op when
unused**.

**Why.** `engine.sh` had no way to pass a one-off engine flag or container env
var. Without it, every bisection step would have meant editing the script — which
makes each round a code change rather than a config change, and makes "one
variable at a time" hard to audit afterwards. R05 and R06 need
`DECODE_EXTRA_ARGS=--disable-custom-all-reduce`.

**How.** `git apply patches/0001-engine-sh-extra-args-env.patch` from the repo
root, against HEAD `5a342acffe10e09729662ff40e81b22a4367fd75`. Pure addition, no
deletions.

**Context.** Adapted from a parallel session's
`patches/0001-engine-sh-spur-fixes.patch`. **Their version also sets
`NCCL_SOCKET_IFNAME=$nic`**; that hunk is deliberately **excluded** here. Their
nodes needed it (on a Kubernetes host NCCL otherwise picks the `kube-ipvs0` dummy
interface and `ncclCommInitRank` dies with "Connection refused"); ours do not —
this stack initialises fine and has completed full benchmarks. Including it would
have added a second variable in the middle of a bisection. If you hit that NCCL
symptom, that hunk is the fix, and it is a good one; it just is not part of *this*
result.

---

## `nextn-fusion-fork-4350d37c5.patch`

**What.** Adds one class attribute to sglang's `models/glm4_moe.py`:

```python
class GlmMoeDsaForCausalLMNextN(DeepseekV3ForCausalLMNextN):
    fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"
```

Origin: `xiaobochen-amd/sglang`, branch `fix/glm52-nextn-shared-experts-fusion`,
commit `4350d37c5ba7aaa9dfaa91261f57e1aae1fd49da`. Not in `sgl-project/sglang`
main as of 2026-09-18.

**Why.** `configs/model_config.py` rewrites a GLM DSA draft's architecture string
to `"GlmMoeDsaForCausalLMNextN"`, but the class inherited
`DeepseekV3ForCausalLMNextN`'s value, so the name compare in
`deepseek_v2.py::shared_experts_fusion_disable_reason` could never match and the
**draft** silently lost shared-experts fusion — the gate returns a reason string,
not an error.

**How.** Do **not** apply this with `patch`/`git apply`: the target line moves
between sglang trees. Use `scripts/apply_nextn_fusion_fix.sh`, which anchors on
the class statement, is idempotent, and exits non-zero if its marker is absent
afterwards. It runs inside the container — `scripts/Dockerfile.yihou.nextnfix`
bakes it into a thin layer so it survives container recreation. Verify by
**importing the class with a GPU attached**, not by grepping the source
(REPRODUCE.md §3).

**Context — status.** The defect is **real and present** on this stack
(confirmed by code read and by the log signature), and this patch **does not fix
the garbled decode**. R03 applied it and the output was still wrong; R06 removed
it and the cure still worked. It is an independent, upstream-correct defect,
included here because reproducing R03 and R06 requires both arms.

Whether it improves MTP acceptance on this stack is **open** — see `notes.md`
§7d. One sample each, overlapping ranges; this kit declines the claim.
