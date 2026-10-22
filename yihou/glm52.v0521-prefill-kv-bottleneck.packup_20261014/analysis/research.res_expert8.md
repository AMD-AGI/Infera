Short answer: there is a first-hand result of about 26k with 8 experts, but it is not comparable with today's 22,306 setup. It came from a v0.5.19 image, used 8 experts on **both** legs, and was CONC=56, not 52.

## Q3. The "~26k" result

| run | node | CONC | score (tok/s/chip) | HiCache | date |
|---|---|---|---|---|---|
| `r001-hc274` | crsuse2-m2m-136, same-node P4D4 | 56 | **26,462.16935** | 274 GB, default chunk | 2026-09-24 |
| `r001-hc450chunk512` | crsuse2-m2m-137 | 56 | **26,071.69187** | 450 GB, `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512` | 2026-09-24 |
| arm A (`rounds/022-c56-136`) | 136 | 56 | 23,892.367 | 154 GB | 2026-09-24 |

- The two top scores are read directly from `"total_tput_tps"` in each run's `agentx_conc56.json`, under `/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-hicache-ceiling-sweep/glm52.hicache-ceiling.packup_20260924/results/<run>/`. Arm A's raw JSON is not in that packup; its number comes from `.../yihou-expert8-hicache-c56/RESULTS.md:18`. That file also quotes a "historical C40 ref" of 21,032.478, which is the 256-expert model at EP1.
- Packup README: lines 17-23 (headline) and 131-142 (deployment). Config: `scripts/config.yihou.base.sh`. `r001-hc274/params.txt` says: `hicache=274 write_policy=write_through conc=56 duration=3600 max_running=64 extra_env=''`.
- How it was configured:
  - Synthetic 8 experts in **both target and draft, on both legs**. TP4/DP4/DPA with **EP4**.
  - MTP EAGLE 5/1/6 with `SGLANG_SIMULATE_ACC_LEN=3.61`. Triton DSA, fp8_e4m3 KV, page size 64.
  - `index_share_for_mtp_iteration=false`, `--disable-shared-experts-fusion`, mem 0.85.
  - **`--max-total-tokens 3670016` on both legs** (config.yihou.base.sh:113-120).
  - HiCache on prefill only: write_through, `io_backend=kernel`, `page_first`, ratio 1.5.
- Image: `infera-sglang:v0519-yihou-0917-nextnfix-hicache` (`sha256:fd7220a5...`), sglang 0.5.19 at `7ccbf5fd`.
- Decode capacity in that run: 3,670,016 tokens/rank, KV 188.76 + 2.42 GB, 57.42 GB still available after pool init (`/home/yihou/dev/git/infera.glm52.view/decode.kvcache.report.md:74`).
- For comparison, real 256 experts on v0.5.21 at decode mem 0.95 gives 2,773,376 tokens/rank (report:84). e2g at C40 scored 22,114 (report:86).
- I found no other 25-27k number in `$PK`, `$VIEW` or `/shared_nfs/yihou_glm52`. The `/shared_nfs` hits are only docker-ps and provenance captures.
- `glm52.oai-samenode-1p1d-p4d4-xgmi.packup_20261009/analysis/ANALYSIS.md:36`: an 8-expert run with prefill HiCache at 450 also hit SEGV (`r003-e4-expert8`), so expert count was ruled out as the cause of that crash.

## Q1. How the 8-expert model is built and turned on

- **Overlay:** `prepare_model.yihou.py:17-24` copies `config.json` with these changes: `n_routed_experts=8, num_experts_per_tok=8, n_group=1, topk_group=1, ep_size=4, yihou_synthetic_expert8=True, index_share_for_mtp_iteration=False`. It symlinks the 282 shards and copies the remaining files.
- Existing overlays on NFS (all 290 entries, `n_routed_experts: 8`, `yihou_synthetic_expert8: true`, `GlmMoeDsaForCausalLM`, `num_nextn_predict_layers: 1`):
  - `/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-expert8-pd/model`
  - `.../yihou-expert8-hicache-c56/model`
  - `.../yihou-hicache-ceiling-sweep/model`
- **Switch:** only the config key `yihou_synthetic_expert8`. There is no env var. Without the key, the patch does nothing.
- **Two patch versions:**
  - `$PK/yihou-expert8-pd.packup_20260924/patches/yihou-expert8.patch` is target-only. It rejects nextn (`deepseek_v2` hunk: `if ... is_nextn ... raise`), rejects `draft_model_idx`, and drops all layers ≥ 78, including the MTP layer. So it **cannot run with MTP**.
  - `.../glm52.hicache-ceiling.packup_20260924/patches/yihou-expert8-nextn.patch` is the MTP-capable version used for the 26k runs. It adds `validate_role(is_nextn)`, and the draft loads only layer 78 minus `embed_tokens`/`shared_head.head`. Draft detection uses `source.model_config.is_draft_model`.
- **Files touched (4):**
  - new `srt/models/yihou_expert8.py`
  - `srt/models/deepseek_v2.py`: in `DeepseekV2MoE.__init__`, after `self.topk = TopK(...)`, sets `custom_routing_function=route_all_eight` and output format STANDARD.
  - `srt/model_loader/loader.py`: in `_get_weights_iterator`, a filtered safetensors iterator that never reads experts ≥ 8 and slices the gate and its bias to 8.
  - `srt/layers/moe/topk.py`: in `select_experts`, a `yihou_expert8` branch before `dynamic_expert_bias`.
- **How they get into the image:** bind-mounts listed in `patches/mounts.yihou.tsv`.
- **Required flags:** `--disable-shared-experts-fusion`. The patch raises if fused shared experts are on, and the v0.5.21 base config has fusion ON (`config.yihou.v0521-base.sh:235`). Every run so far used EP4 (the overlay sets `ep_size=4`); EP1 with 8 experts has never been tested.
- Original kit (target-only): cross-node 138→136, MTP and HiCache off, fixed 4096/1024 at C32, 7,120 tok/s total. Decode had 4,097,408 tokens/rank and 31.44 GB/rank of weights (`README.md:8-22`).

## Q2. Porting to v0.5.21

- **Text applies cleanly.** I dry-ran both patches against `$CAP/src/image-9d755dd6` with `patch -p2 --dry-run` (rc=0, no files written). Hunk offsets: deepseek_v2 +92 (lands at 786), loader +17 (at 621), topk +189 (at 2599).
- **What I read in v0.5.21 (all lines in `$CAP/src/image-9d755dd6/sglang/srt`):**
  - The `TopK` construction context is unchanged (`models/deepseek_v2.py:786-789`).
  - `is_hash` and `is_nextn` still exist (`models/deepseek_v2.py:677-680`).
  - `loader.py` still imports `time` and still has `source.model_config` and `counter_before_loading_weights`.
  - `ModelConfig.is_draft_model` exists (`configs/model_config.py:501`).
  - In `select_experts`, a custom routing function forces the ROCm router partials to be reduced first (`topk.py:2548-2563`).
  - With output format STANDARD, `TopK.forward_cuda` goes into `select_experts` (`topk.py:675-740`).
- **Not verified:** runtime behaviour. Nothing has been launched on v0.5.21 with the patch.
- **The old int32 workaround is not needed on v0.5.21.** `kernels/ops/kvcache/triton_store_cache.py:65,173` already loads `loc` as int64. The old int64-page patch therefore doesn't apply, and the `--max-total-tokens 3670016` cap existed only for that int32 bug.
- **No conflict with current mounts.** `mounts.yihou.v2-regbase.tsv` mounts only `kv_events_publisher.py`, `forward_mla_rocm.py` and `mooncake/conn.py`, none of the four expert8 files.
- **HiCache chunk default is still 256 GiB in v0.5.21** (`environ.py:770`). The current recipe already sets 512 (`config.yihou.E2-target.sh:12`).

## Q4. 8 experts on one leg only

- **No sglang-side blocker found.** At bootstrap, PD only checks `page_size`, `kv_cache_dtype` and the DSV4.1 DSpark layout (`disaggregation/common/conn.py:1168-1203`). I found no check on experts or weights in `disaggregation/`.
  - Per the expert8 kit's `environment.md:161`, the overlay keeps all attention/KV dimensions and 78 layers, so the KV geometry should match. [inferred: per-layer KV item lengths match on both sides]
- **Decode only:** decode runs MTP, so it needs the **nextn** patch. The draft loads from decode's model path and also becomes 8-expert. That combination ran on v0.5.19, just not one-leg. [inferred: an alternative is `--speculative-draft-model-path` pointing at the original model, untested]
- **Prefill only:** prefill has no MTP, so even the target-only patch is enough. Prefill's first token then comes from the synthetic model, which is fine because correctness is waived.
- **Harness gap:** `MODEL` is one variable shared by both roles (`engine.sh:253`), and the router tokenizer and the AgentX validator read it too (`launch.sh:200`, `agentx_bench.sh:66`). The per-role hooks that exist are `{PREFILL,DECODE}_EXTRA_ARGS`, appended last (`engine.sh:318-323`), and `{PREFILL,DECODE}_MOUNTS_ADD` (`engine.sh:170-180`). Two untested ways to use them:
  - (a) Role-only mount of the overlay `config.json` over `$MODEL/config.json`, plus the four patched sources. This keeps the model path the same on both legs. [inferred]
  - (b) Pass `--model-path <overlay>` in one role's `EXTRA_ARGS`. This depends on last-flag-wins parsing in `infera.engine.sglang`, which I did not verify, and the overlay directory would have to be mounted.
- **The leg with 8 experts must also get `--disable-shared-experts-fusion`.** EP is your choice.

=====VERIFY=====
**Adversarial check of the expert8 report**

I re-read the cited files on the cluster, read-only. Short answer: the scores, both patches, the configs and the harness hooks check out. The report gets four things wrong: arm A's raw JSON is in the packup after all, the source of the r003-e4-expert8 crash note, the HiCache setting of that crash run, and the line refs and value for decode capacity on v0.5.21.

**Key claims**

1. **WRONG (minor).** The report says ~26k came from a v0.5.19 image, 8 experts on both legs, at C56 rather than 52. The run facts are right, but the C56-vs-52 contrast mixes up two different numbers.
   - The user's "52" is the C to use for the new experiments, i.e. the highest from yesterday's v0.5.21 sweep. The script `$WS/harness/a4_q1c52.yihou.sh` exists; I did not check its result.
   - 56 is only the CONC of the old 26k runs.
   - Both legs ran 8 experts. `config.yihou.base.sh:119-120` gives both `PREFILL_EXTRA_ARGS` and `DECODE_EXTRA_ARGS` `--disable-shared-experts-fusion --max-total-tokens $YIHOU_MAX_TOTAL_TOKENS`, and decode also has `--disable-custom-all-reduce`. There is a single `MODEL`.
2. **CONFIRMED.** The two top scores and their params, read first-hand:
   - `r001-hc274/agentx_conc56.json`: `"total_tput_tps": 26462.16935`.
   - `r001-hc450chunk512`: 26071.69187.
   - `params.txt` matches the report: max_running=64, write_through, duration 3600. The chunk512 run sets `extra_env='SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512'`.
   - The README headline table at lines 17-23 matches.
3. **WRONG.** "Arm A raw JSON is not in that packup." It is there: `.../glm52.hicache-ceiling.packup_20260924/results/armA-reference-S154/armA-136-agentx_conc56.json`, with `"total_tput_tps": 23892.36677`.
4. **Missing from the report.** The same packup has a C64 attempt, `results/r003-c64/`.
   - Its params: hicache=450, chunk 512, conc=64, max_running=64.
   - `status.tsv` shows `bringup FAIL:launch.sh rc=1`. README:172 and :195 call this a "harness false negative", so there is no C64 score.
5. **CONFIRMED, but the reason is incomplete.** The `--max-total-tokens 3670016` cap is at `config.yihou.base.sh:113-117`.
   - The comment there gives the int32 device-side triton store bound, and also "Verified to remove all Mooncake auto-split and memory-region registration failures". So "existed only for that int32 bug" understates it.
6. **CONFIRMED, with a path fix.** The v0.5.21 `triton_store_cache.py` loads `loc` as int64 at lines 65 and 173.
   - The real path is `$CAP/src/image-9d755dd6/sglang/kernels/ops/kvcache/triton_store_cache.py`, not under `sglang/srt/`.
   - The old `patches/yihou-sglang40846-int64-page.patch` targeted this same file.
   - Whether v0.5.21 needs no cap at 8 experts is still [inferred]. It has not been run.
7. **WRONG line refs, and a value off for the current recipe.** The report cites `decode.kvcache.report.md:84` and `:86` for 2,773,376 tokens/rank and e2g 22,114.
   - Line 74 (P1: 3,670,016 tokens; 188.76 + 2.42 GB; 57.42 GB still available) is CONFIRMED.
   - Line 84 is actually the L2 bs16 row. The 0.95 row in the L1 table is v0519.
   - The v0.5.21 numbers are at lines 90-91:
     - V2a, e2b4 with the fused indexer off: 2,773,376.
     - V2b, e2c/e2g with the fused indexer on, which is the current recipe: **2,762,176**.
   - e2g C40 = 22,114 tok/s/chip (line 91): CONFIRMED.
8. **WRONG path and detail.** The report cites `ANALYSIS.md:36` and says r003-e4-expert8 ran "with prefill HiCache at 450".
   - The file is under `$VIEW` (`/home/yihou/dev/git/infera.glm52.view/glm52.oai-samenode-1p1d-p4d4-xgmi.packup_20261009/analysis/ANALYSIS.md:36`), not `$PK`.
   - `working_process.full.md:31,168` shows that run was prefill-only on node 274, ep4, **hicache 0**.
9. **CONFIRMED.** The overlay changes in `payload/scripts/prepare_model.yihou.py:19-21`.
   - The target-only patch rejects nextn: `yihou-expert8.patch:77-78` (`if self.num_fused_shared_experts or is_nextn or self.is_hash: raise`) and `:98` (`draft_model_idx is not None`).
   - The nextn patch is in the ceiling packup's `patches/`.
10. **CONFIRMED.** v0.5.21 base has fusion ON: `config.yihou.v0521-base.sh:235`, comment "Shared-experts fusion ON (no --disable-shared-experts-fusion); no --max-total-tokens".
11. **CONFIRMED.** The harness hooks in `$WS/harness/bench-harness/engine.sh`:
    - Line 253 is `--model-path "$MODEL"`.
    - The role mount hooks `{PREFILL,DECODE}_MOUNTS_ADD` are at 170-180.
    - The role `extra_args` are appended at 318-323.
    - `environ.py:770` has `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB = EnvInt(256)`.
12. **WRONG/UNVERIFIED.** "EP1 with 8 experts has never been tested."
    - The ceiling packup runs, r003-e4 and the original kit all used EP4.
    - `r004-e5-ep1` tested EP1, but with the real 256-expert model (`working_process.full.md:32,169`).
    - `ANALYSIS.md:37` and `:240` state that EP=1 is the correct value, matching the real checkpoint's `ep_size: 1`. I did not search further for an 8-expert EP1 run.

**Important facts the report misses**

- **Decode KV may not be what limits today's runs.** `decode.kvcache.report.md:116`:
  - Decode token usage was 0.27→0.45 in e2g A1 (ledger R046) at 0.95, and 0.69 on one rank in c44c (ledger R004).
  - This bears directly on experiment 1a, which aims to free decode KV. It is measured on the v0.5.21 0.95 runs (C40/C44), not at C52.
- **The int32 per-layer-offset theory was refuted.** The ceiling packup's `notes.md:179-198` reports that `r001-hc274` ran a per-layer span 1.64× 2^31 with zero faults. That theory (a HiCache host-pool fault theory) should not be reused.
- **Arm A as a same-model control.** It shows +10.76% for hc274 over hc154 at C56 (README:24). Both arms used the 8-expert model, so this gain is HiCache size, not expert count.