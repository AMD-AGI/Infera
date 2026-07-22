# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Conventions

- <CRITICAL><MUST> USE ENGLISH WHEN WORKING, USE CHINESE WHEN COMMUNICATING WITH THE USER, NO JAPANESE. </MUST></CRITICAL>
- `temp/` is gitignored local scratch. Do not put anything meant for review there.
- Lint/format: `ruff check . && ruff format --check .` (line-length 100, double quotes).
- Tests: `pytest` (unit only by default; `-m slow` and `-m integration` gate
  hardware/service tests and are skipped by default). `asyncio_mode = auto`.

## What this is

`infera` (PyPI `amd-infera`) — a lightweight inference orchestration layer for the
ROCm ecosystem. Four cooperating processes coordinated through **etcd**:

- `infera.server` — FastAPI OpenAI-compat frontend + router. Stateless.
- `infera.engine.{sglang,vllm,atom}` — engine workers, self-register via etcd lease,
  each run as `python -m infera.engine.<name>`.
- `infera.kvd` — optional tiered KV-cache daemon (RAM → SSD → NFS).
- `infera.gaie` — optional k8s Endpoint Picker (Envoy ext_proc gRPC).

Engines share the `infera/engine/base.py` (`BaseEngine`, `EngineConfig`)
self-registration + lifecycle contract. Each engine's `__main__.py` applies ROCm
env defaults before spawning the inference subprocess so they are inherited.

## Container images

Each engine has one canonical Dockerfile under `deploy/docker/`, built from the
repo root. `deploy/docker/patches/` holds per-engine source patches applied at
build. **gfx942 (MI325X/CDNA3) needs a separate sglang image**
(`Dockerfile.sglang.gfx942`, MI30x base) because the default `Dockerfile.sglang`
targets MI355X (gfx950, mi35x base) and one sglang image cannot serve both arches.

## CURRENT TASK (2026-07-22) — MI325X (gfx942) + DeepSeek-V4 support

**Design spec: `docs/superpowers/specs/2026-07-22-mi325-dsv4-design.md` (APPROVED).**
Bring gfx942 dsv4 support in under an enforced support matrix; replace the current
fp4-only patch-enabling mechanism with a no-third-party-patch policy (rule 3).

### Support matrix (the single contract — code, docs, image, patch, test all express it)

| variant | quant | vllm | sglang | atom |
|---------|-------|------|--------|------|
| Pro     | fp4   | ✅ native (triton auto-dequant) | ❌ fail-fast | ❌ fail-fast |
| Pro     | fp8   | ❌ fail-fast | ✅ env + CLI | ✅ env |
| Flash   | fp4   | ✅ native | ❌ fail-fast | ❌ fail-fast |
| Flash   | fp8   | ❌ fail-fast | ✅ env + CLI + MTP | ✅ env + MTP |

Why: gfx942 has no native FP4 MoE kernel — only vLLM upcasts fp4→bf16 in-kernel
(`triton_unfused`, no patch); sglang/atom only ran fp4 via now-forbidden source
patches → fail fast. fp8 runs natively on sglang/atom (not validated on vllm).
Flash-fp8 decode kernel is broken on gfx942 → must route decode through MTP/EAGLE
(Pro-fp8 is correct without it). Detect variant by config dims (Pro=hidden 7168/
61 layers, Flash=4096/43), quant by `quantization_config`.

### Auto-injected knobs (set-if-unset; operator always overrides)

- **fp8 sglang** env: `HSA_NO_SCRATCH_RECLAIM=1`, `SGLANG_USE_ROCM700A=0`,
  `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`, `AITER_BF16_FP8_MOE_BOUND=0`;
  CLI: `--attention-backend dsv4`, `--disable-shared-experts-fusion`; Flash adds
  `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4`.
- **fp8 atom** env: `HSA_NO_SCRATCH_RECLAIM=1`; Flash adds CLI `--method mtp
  --num-speculative-tokens 3`.
- **NOT injected**: `--cpu-offload-gb`, `--max-total-tokens`, `--max-running-requests`,
  `--mem-fraction-static` (experiment-only VRAM band-aids, not product defaults).

### Implementation checklist (COMPLETE — branch yihou.dev.mi325)

- [x] Create + push branch `yihou.dev.mi325.fp4patch.legacy` from current HEAD
      (preserve fp4 patches), then return to `yihou.dev.mi325`.
- [x] New module `infera/engine/dsv4_gfx942.py`: `detect_dsv4`,
      `Dsv4UnsupportedError`, `apply_gfx942_dsv4(model_path, *, engine, argv)`.
- [x] Remove `apply_dsv4_gfx942_env_defaults` + `_is_dsv4_fp4_model` from
      `rocm_rdma_env.py` (keep `is_gfx942` + RDMA fns).
- [x] Wire the new call into sglang/atom/vllm `__main__.py` (vllm = enforce/env only).
- [x] Delete `patches/sglang-dsv4/` and `patches/atom/patch_dsv4_fp4_dequant_gfx942*`;
      de-patch `Dockerfile.sglang.gfx942` (keep it, MI30x base) and `Dockerfile.atom`.
- [x] Tests `tests/engine/test_dsv4_gfx942.py`: detection, matrix enforcement,
      env/CLI set-if-unset + override, Flash MTP, no-op off-gfx942/non-dsv4.
- [x] Docs: `manual/wip/mi325-deepseek-v4.md` (WIP, not linked into the release
      TOC) + module docstrings (what/why/how/context; no process narrative — rule 5).

Known follow-up (pre-existing, out of scope): `.github/scripts/build_test_push.sh`
has no case for the `Dockerfile.sglang.gfx942` variant — the gfx942 image isn't
CI-wired. Wire it in a separate task if the image should be CI-built.

### Reference materials

Experiment packups (support matrix + exact knobs) live in
`~/dev/git.16-19/legacy.infera/infera.fuck/` (dsv_flash_fp8_mi325x.sglang,
atom_dsv4_flash_fp8.packup_20260721, vllm_dsv4_flash_mxfp4.packup_20260721,
dsv4_mi325x.{vllm,atom,sglang}, pro_fp8_nospec_control.packup_20260721).
