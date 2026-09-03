# logs/ — what each file proves

All gzipped (`gzip -9`). Total ~320 KB. Single node, so no per-node split —
everything here is from `smci355-ccs-aus-n05-29`.

**Warning:** `worker_glm53_f8_mix.log.gz` and the verify transcripts contain the
engine's `server_args` dump — **~20 KB on one line**. `zgrep` it; never `zcat` it
into an agent's context.

| file | what it proves |
|---|---|
| `build.log.gz` | the image is what it claims. Contains the `c821c425` checkout, `[verify-glm53-overlay] static: Glm5NextForConditionalGeneration defined and exported`, and the libionic gate printing `LIBIONIC_ABI ... = 4..4`. Also records that the overlay verifier's **import** check was skipped for want of a GPU on the builder. |
| `mix_up_r0.log.gz` | round 0 bring-up. Shows the VRAM measurement choosing `--mem-fraction-static = 0.60`, `worker serving after 820s`, **and** the two failures worth keeping: `ETCD FAILED` (peer-flag bug, `notes.md` §3a) and the terminal `error reading input file: Stale file handle` (NFS mid-run edit, §3b). |
| `mix_up_r1.log.gz` | round 1 bring-up, graphs on. Shows the **VRAM-drain wait** working — `t+5s: worst card 18 GiB … plateaued at 20 GiB -- that is someone else, proceeding` — which is the fix for §3c, and a neighbour holding ~20 GiB at launch. |
| `worker_glm53_f8_mix.log.gz` | the engine's own log. **This is the file every §E/§F/§G/§H claim in the README was grepped out of.** Copied at 11:21 UTC, so it covers the long serving period but **stops before** the lead's 11:32 sweep. |
| `router.log.gz` | the infera router. Confirms `router-policy=kv-aware`, worker registration with `disagg=DisaggMode.MIXED`, and carries one **negative worth knowing**: `kv events: NOT subscribing to 192.168.3.26:31400 — it registered no kv_block_size` — i.e. kv-aware routing was effectively off for this single worker. Harmless with one worker; would matter with several. |
| `f8_c1_by_lead.jsonl.gz` | per-request records, conc-1 arm — **measured by the team lead** |
| `f8_c8_by_lead.jsonl.gz` | per-request records, conc-8 arm — **measured by the team lead**, see `../results/README.md` |

## Useful greps

```bash
# the health check that matters -- expect 4 and 4
zgrep -c 'Using AITER gfx950 mHC pre/post kernels'        worker_glm53_f8_mix.log.gz
zgrep -c 'Using fused AITER mHC attention-to-FFN boundary' worker_glm53_f8_mix.log.gz

# the FP8 control arm: this line must be PRESENT here (absent on MXFP4)
zgrep 'Shared experts fusion optimization enabled' worker_glm53_f8_mix.log.gz

# the resolved clamp -- NOT the server_args value, which lies (notes.md §1)
zgrep -m1 'capped to' worker_glm53_f8_mix.log.gz

# both memory pools on every decode line
zgrep 'Decode batch' worker_glm53_f8_mix.log.gz | grep 'full token usage' | grep -c 'mamba usage'

# faults -- expect nothing
zgrep -E 'memory access fault|HIP error|Traceback' worker_glm53_f8_mix.log.gz | grep -v '_dynamo\|metrics_context'
```

## What is missing, and why

- **Round 0's worker log does not exist here.** Its container was replaced by
  round 1 before the log was copied out. This is why `notes.md` §1 records
  round 0's resolved `max_running_requests` as permanently unknown. The lesson —
  `docker cp` the log *before* teardown, always — is the same one the previous
  packup on this project learned.
- **No log covers 11:21 → 11:35:42 UTC**, the window containing the lead's
  sweep. That log died with the container, and is no longer needed to settle
  anything: the `flash_fp8_crash` directory name is explained in `notes.md` §4 —
  a health check against the wrong port, and no crash.
