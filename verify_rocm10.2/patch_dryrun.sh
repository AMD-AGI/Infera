#!/usr/bin/env bash
# Apply the sglang patch steps of Dockerfile.sglang.rocm102, in order, inside a
# throwaway container of the base; report each step instead of stopping.
R=/repo/deploy/docker/patches
step() { echo "=== $1"; shift; if "$@" >/tmp/step.log 2>&1; then echo "OK"; tail -4 /tmp/step.log; else echo "FAIL rc=$?"; tail -25 /tmp/step.log; fi; }
step glm_nextn_fused_arch python $R/sglang_dsa/v0521/patch_glm_nextn_fused_shared_experts_arch.py
step dsa bash -c "cp -r $R/sglang_dsa /tmp/sglang_dsa && PATCH_DIR=/tmp/sglang_dsa bash /tmp/sglang_dsa/v0522/apply_sglang_dsa_patches_v0522.sh"
step mooncake_wait_consumer python $R/sglang_disagg/v0521/patch_mooncake_wait_event_consumer.py
step disagg_nonfinal_chunk python $R/sglang_disagg/v0522/patch_prefill_nonfinal_chunk_wait_event.py
step responses_tool_args python $R/sglang_responses/v0522/patch_responses_unstreamed_tool_args.py
step rocm_staged_write_back python $R/sglang_rocm/patch_hicache_rocm_staged_write_back.py
step kv_metrics_noblock python $R/sglang_kv_events/patch_kv_metrics_send_noblock.py
step pp_embed_dim python $R/sglang_pp/v0522/patch_deepseek_v2_pp_embed_dim.py
echo "=== import check"; python -c "import sglang.srt.disaggregation.prefill, sglang.srt.disaggregation.mooncake.conn, sglang.srt.models.glm4_moe; print('imports OK')" 2>&1 | tail -3
