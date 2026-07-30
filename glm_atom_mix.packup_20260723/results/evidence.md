# Evidence — GLM-5.1-FP8 ATOM single-node

Run: 2026-07-23, chi2866 card4-7, `infera/engine-atom:kimi` (c7505e171e31), TP4.

## temp=0 factual probes (via /v1/chat/completions, temperature=0, thinking OFF) — VERBATIM

| Prompt | Reply (verbatim) | Expect | Verdict |
|--------|------------------|--------|---------|
| `The capital of France is` | `Paris` | Paris | ✅ PASS |
| `The capital of China is` | `The capital of China is Beijing.` | Beijing | ✅ PASS |
| `2+2=` | `2 + 2 = 4` | 4 | ✅ PASS |

**VERDICT: ALL PASS** — facts correct, coherent, non-empty, clean stop (finish
reason `stop`). NOT first-token-only (the gfx942 broken-decode failure mode did
not occur). GLM is a thinking model; probes used
`chat_template_kwargs={"enable_thinking": false}`, max_tokens 200.

## Final working launch command (verbatim)

```
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
```

## What ATOM loaded / chose

```
arch GlmMoeDsaForCausalLM (model_type glm_moe_dsa) loaded cleanly (no arch abort)
allocated MLA chunked-prefill workspaces (handles GlmMoeDsa MLA + DSA path)
cudagraph capture [1..512] ; post-init VRAM ~91.6% ; "ready to accept requests"
NO --method mtp (GLM ships 0 mtp/nextn draft tensors; gfx950 plain decode correct)
```

## Disk headroom during run
Started 34G free (96%); low 33G (97%) during compile — never near the 8G stop
line (model RO-mounted keeps the writable layer tiny). Node stable throughout.

## Source
- Raw bring-up report: `agent_result_raw.md` (this dir).
- Full server log: `../logs/glm_atom_mix.log.gz`.
