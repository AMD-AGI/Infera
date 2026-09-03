# infera repo changes this run depends on

Branch `yihou.dev.glm53.expr`. These were **uncommitted working-tree changes
while the experiment ran**; they were committed by the team's checkpoint pass at
**2026-09-02 05:06 UTC**, while this packup was being written. The copies here
are byte-identical to the committed versions (verified with `diff`).

Base of the branch at the time of the run: `f48b79d04316907f29478f9f037d893bdf50cd4a`.

| commit | subject | what it contains |
|---|---|---|
| `abcbccf0e8eee73f7e2d3ba4a6d3504c9bc80fb3` | *image: move the sglang base to v0.5.18, and carry the two anchors that moved* | `Dockerfile.sglang` base bump v0.5.17 → v0.5.18, `patches/sglang_dsa/patch_draft_cuda_graph_dp_vote_v0518.py`, `scripts/apply_sglang_dsa_patches.sh`, `scripts/reanchor_sglang_disagg_glm53.sh` |
| `ea989b3b39c30a25d659fb331a0d0dce2ab9c3e1` | *image: a GLM-5.3-Flash recipe, pinned to #36607's head rather than its first commit* | **`Dockerfile.sglang.glm53`** (the image this experiment used), `scripts/verify_glm53_overlay.py`, `scripts/print_ionic_abi.py` |
| `61e7ca41804ec57cafe85fc87e534299e7e31cda` | *test: park the four GLM-5.3 mixed-e2e cases, recipes intact* | `tests/e2e/pd_mixed/sglang/matrix.py`, `tests/e2e/harness/matrix.py` — CI-disabled e2e rows (mission deliverable 3) |
| `37f2a8fbf50cd68cbd383aa06f3af385794770f2` | *examples: a single-node MIX kit for all four GLM-5.3 checkpoints* | `examples/sglang_mix_glm5.3/` (mission deliverable 4) |

The last two are **repo deliverables written from this result**, not inputs to
it. This packup's `scripts/` are the scratch scripts that actually ran;
`examples/sglang_mix_glm5.3/` in the repo is their productised form and is what
a new user should reach for.

### What those two deliverables contain

- **`tests/e2e/pd_mixed/sglang/matrix.py`** — four GLM-5.3 mixed-e2e rows, all
  **`enable=False`**, i.e. carried but disabled in CI, which is exactly what
  mission deliverable 3 asks for. Four matching model-id constants in
  `tests/e2e/harness/matrix.py`.
- **`examples/sglang_mix_glm5.3/`** — `env.sh`, `engine/{worker,up,smoke,bench,down}.sh`,
  `README.md`. One kit, all four checkpoints behind a `VARIANT` switch.

### Scope: what this packup validates, and what merely ships

The example README carries an explicit validation-status table. Mirrored here so
the packup does not read as broader than it is — **only the first row is what
this packup measured**:

| variant | status |
|---|---|
| `flash-mxfp4` | **validated end to end — this packup.** Full infera stack, two separate nodes, plus the fixed-length sweep. |
| `big-fp8` | validated (all smoke blocks green, `max_total_num_tokens=1148288`) — **big-model track, not this packup** |
| `big-mxfp4` | validated (AITER FP4 path confirmed dispatching `torch.float4_e2m1fn_x2` / `per_1x32` rather than dequantising to BF16) — **big-model track, not this packup** |
| `flash-fp8` | **recipe carried over, not validated in the kit.** A prior bring-up served this checkpoint on a different image and SHA. Whether it needs `--disable-shared-experts-fusion` is **unverified** — that checkpoint's shared/routed precision split has not been read. |
| PD (1P1D), any variant | **not covered at all.** For the big pair the shape matches `examples/sglang_1p1d_glm5.2/`, which is validated for GLM-5.2 only. |

Mission deliverable 4 asks for examples covering PD as well as MIX. **PD is not
delivered** by this kit and is not claimed anywhere in this packup.

## Files copied here

- `tracked-modified.diff` — the diff of the two previously-tracked files
  (`Dockerfile.sglang`, `apply_sglang_dsa_patches.sh`) from `f48b79d` to
  `abcbccf`.
- `deploy/docker/Dockerfile.sglang.glm53` — **the file that built the image**.
  Read its header: it carries the pin rationale, the "when to delete this file"
  test, and the reason the overlay is not folded into `Dockerfile.sglang`
  (doing so would put a fetch of unreleased source into the default build path
  for Kimi-K3 and GLM-5.2).
- `deploy/docker/scripts/reanchor_sglang_disagg_glm53.sh` — two
  `patches/sglang_disagg/` anchors moved on the v0.5.18 tree; this re-anchors
  them into a copy so the v0.5.17 Kimi-K3 image keeps building.
- `deploy/docker/scripts/verify_glm53_overlay.py` — build-time assertion that
  the overlay actually landed at the pinned ref.
- `deploy/docker/scripts/print_ionic_abi.py`
- `deploy/docker/patches/sglang_dsa/patch_draft_cuda_graph_dp_vote_v0518.py` —
  the one DSA patch that needed re-anchoring for v0.5.18 (upstream renamed
  `tp0_info` → `tp0_info_cpu` and dropped `.item()`). **Not used by this
  experiment** — `Dockerfile.sglang.glm53` skips the DSA set entirely — but it
  is part of the same base bump.
