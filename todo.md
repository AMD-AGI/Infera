
### T78 — `apply_patch` fails on a REAL stage-4 artefact and succeeds on a REPLAYED one, and nobody has explained why

**This is the phase-⑤ blocker and it currently has no owner.** Recorded 2026-09-05 by the
leader from a comparison of four runs, not from a diagnosis — **the observation is
first-hand, the mechanism is unknown, and nobody should treat the two as the same.**

| run | `mock_stages` | stage 4 | `apply_patch` |
|---|---|---|---|
| `p9` (093) | `none` | **real** | **FAILED** — 22 s, wrote nothing to its grant |
| m3's `217d` (217) | `none` | **real** | **FAILED** — same stopping point |
| m5's `keep12` (237) | `m1,m2,m3,m4` | replayed | **succeeded**, chain completed 17/17 |
| m5's `keep13` (237) | `m1,m2,m3,m4` | replayed | reached `integrate_and_verify` |

**Two independent full-real chains, two nodes, same stop. Two mock-1-4 chains, same node,
straight through.** The variable that tracks it is whether `kernel_optimization` was
produced by a live `optimize_kernel` or replayed from the corpus.

**What is measured about the failure** (m2, from `p9`):

```
agent: '${m5_agent:-runner}'      program body, no transcript, no zone logs
18:10:58 -> 18:11:20              22 seconds
did:     generated a 456-line patch from m4's optimized_kernel.py,
         staged trees/0/{a,stock,tree}
wrote:   nothing to .../v0/content -- "the agent's grant and it is empty"
event:   escalated   -> 2a5b4e8, program body, no agent to instruct
```

**It did the substantive work and then did not write its handoff.** Not resource, not
timing, not contention — **deterministic, in 22 seconds, with no GPU involved.**

**Candidate directions, none investigated:**

- the real `kernel_optimization` differs from the replayed one in a field `apply.py`
  reads — **note that the leader's builder produces the replayed one, and it has already
  been wrong three times today** (`packup_noop_` naming, missing `environment.yaml`,
  absolute host paths);
- `optimized_kernel.py` from a live `optimize_kernel` is `build_candidate.py`'s
  **three-part composite** (see bug 29), while the corpus's is whatever the corpus holds
  — and `apply.py`'s `surface_regressions` is `ast`-based, so a composite's defs live
  inside a string literal and read as *dropped*;
- something about `trees/0/{a,stock,tree}` staging that only a real artefact triggers.

**The second is the one I would look at first** and it is a five-minute read, not an
experiment: `apply.py:117-121` versus what a live stage 4 emits.

**Why this is worth a numbered item rather than a message:** it is the only thing standing
between us and a completed full-real chain. Stages 1–4 are sealed and passing on real
artefacts (`p9`, twelve tasks). **`apply_patch` is the first door of stage 5 and it is
shut for exactly the runs that matter.**
