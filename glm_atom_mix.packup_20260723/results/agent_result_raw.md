# GLM-5.1-FP8 on ATOM (single-node, MI355X/gfx950, chi2866) — RESULT: PASS

Date: 2026-07-23. Identity: c_huggingface. Card4-7 only.

## 1. Did GLM come up on ATOM? YES
Arch GlmMoeDsaForCausalLM (model_type glm_moe_dsa) loaded cleanly — no
unregistered-arch / missing-model-class abort. ATOM allocated MLA
chunked-prefill workspaces (handles the GlmMoeDsa MLA + DSA path), captured
cudagraphs [1..512], post-init VRAM 91.6%, "Server started successfully and
ready to accept requests".

Final launch cmd (tp4, plain decode, fp8 kv, card4-7):
  docker run -d --name glm_atom_c_hf \
    --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=32g \
    --group-add video --group-add render --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=4,5,6,7 -e ROCM_VISIBLE_DEVICES=4,5,6,7 \
    -e HSA_NO_SCRATCH_RECLAIM=1 -e AITER_LOG_LEVEL=WARNING \
    -v /mnt/vast:/mnt/vast --entrypoint bash infera/engine-atom:kimi -lc \
    "python -m atom.entrypoints.openai_server \
      --model /mnt/vast/xiaobo/models/GLM-5.1-FP8 \
      --kv_cache_dtype fp8 -tp 4 --host 0.0.0.0 --server-port 8000"

## 2. temp=0 probes (verbatim) — PASS
Via /v1/chat/completions (GLM ships chat_template.jinja). GLM is a THINKING
model; with default thinking ON + max_tokens=32 it emits only reasoning
preamble (truncated, NOT wrong). With chat_template_kwargs={"enable_thinking":
false}, max_tokens=200, temp=0.0:
  "The capital of France is" -> 'Paris'                              [PASS, finish=stop]
  "The capital of China is"  -> 'The capital of China is Beijing.'   [PASS]
  "2+2="                     -> '2 + 2 = 4'                           [PASS]
Facts correct, coherent, non-empty, clean stop. NOT first-token-only.

## 3. Flags vs DSv4 ref / mtp / gfx950 notes
- Kept HSA_NO_SCRATCH_RECLAIM=1 (harmless gfx950; mandatory on gfx942).
- -tp 4 (card4-7) instead of -tp 8. --kv_cache_dtype fp8 kept.
- mtp NOT used and NOT possible: config num_nextn_predict_layers=1 but weight
  index has 0 mtp/nextn tensors -> GLM ships no MTP draft weights. gfx950 plain
  decode is CORRECT (decode advances fully); the gfx942 broken-plain-decode bug
  that forced MTP on DSv4 does NOT reproduce here.
- No ATOM_USE_TRITON_MOE / dp-attn / tbo / FP4 knobs needed.

## 4. Disk headroom during run
Started 34G free (96%); model RO-mounted from /mnt/vast so container writable
layer stayed tiny. Low of 33G free (97%) during compile — never near the 8G
stop line. Node stable throughout.

## 5. Teardown
Container glm_atom_c_hf removed; card4-7 freed to ~283MB; atom image KEPT
(c7505e171e31); card0-3 titan/zirui/primus untouched; disk restored to ~34G
free. No prune, no slurm scancel.
