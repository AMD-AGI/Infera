# KV Cache Cliff bench

Mirrors LMCache's June 1 chart: shared 18K-token prefix + 2K-token
per-client unique suffix, ISL=20K, OSL=1 (prefill-only), sweep
concurrency 1 → 250. Two arms compared:

| Arm | Setup | Expected behavior |
|---|---|---|
| `vram_only` | vLLM with default prefix cache in VRAM, no kvd | Throughput cliffs around the concurrency at which concurrent KV exceeds VRAM budget; cached prefixes get evicted and every new request re-prefills 20K tokens. |
| `kvd_v2` | vLLM + our v2 chunked-fusion connector with NVMe-backed (or Vast-NFS-backed) hipfile_root | Throughput keeps climbing past the VRAM cliff because evicted prefixes spill to kvd's file tier; subsequent requests load chunks from there in ~5 ms / 4.5 MiB per `infera-kvd-l3-bench` (the connector round-trip bench that absorbed `bench_packed_v2.py`). |

The two arms must be benched against separately-launched vLLM
servers (each requires different startup flags). This dir provides
the two launch helpers, the bench driver, and a plotter.

## Layout

| File | What it does |
|---|---|
| `launch_vllm_vram_only.sh` | Starts vLLM gpt-oss-120b MXFP4 TP=1 on `$GPU_IDX`, port 8802, no kv_transfer_config. |
| `launch_vllm_kvd_v2.sh` | Same model + flags, but attaches `InferaKvdConnector` (v2 chunked-fusion) on port 8803. Expects a kvd daemon already running. |
| `run_cliff.py` | Drives a pre-launched vLLM endpoint: shared-prefix prompts at varying concurrency, writes per-(arm, concurrency, iter) CSV row. |
| `plot_cliff.py` | Reads one or more CSVs, draws a single throughput-vs-concurrency chart (one line per arm) à la LMCache slide. |
| `results/` | CSVs + PNGs land here. |

## Full run sequence (example)

```bash
# === Arm A: vram-only baseline ===

# 1. Pick a free GPU (devices 1-7; device 0 is Qwen).
#    GPU=4 fits our 60 GB MXFP4 weights + concurrent KV budget.
docker exec -d jj_vllm_gptoss bash -lc '
  GPU_IDX=4 PORT=8802 \
    bash $INFERA_ROOT/bench/kvcache/kv_cache_cliff/launch_vllm_vram_only.sh \
    > /tmp/vllm-cliff-vram-only.log 2>&1
'

# 2. Wait for /v1/models to respond (gpt-oss-120b loads in ~3-5 min).
until ssh "$NODE" 'curl -fsS --max-time 3 http://127.0.0.1:8802/v1/models' >/dev/null; do
  sleep 15
done

# 3. Run the sweep.
docker exec -w $INFERA_ROOT jj_vllm_gptoss bash -lc '
  PYTHONPATH=. python -u -m bench.kvcache.kv_cache_cliff.run_cliff \
      --endpoint http://localhost:8802 \
      --model gpt-oss-120b \
      --arm vram_only \
      --isl 20000 --shared-prefix-tokens 18000 \
      --concurrencies 1,2,4,8,16,32,48,64,80,100,128,160,200,250 \
      --iters 3 --warmup-iters 1 \
      --out bench/kvcache/kv_cache_cliff/results/cliff-vram-only.csv
'

# 4. Stop the vram-only vLLM.
docker exec jj_vllm_gptoss bash -lc 'pkill -f "vllm serve.*8802"'

# === Arm B: kvd_v2 with NVMe-backed file tier ===

# 5. Start kvd daemon with file-tier regions on /mnt/nvme8.
docker exec -d jj_vllm_gptoss bash -lc '
  python -m infera.kvd \
      --socket /tmp/kvd-cliff.sock --max-bytes $((4 << 30)) \
      --long-region /mnt/nvme8/kvd-cliff-long --long-bytes $((80 << 30)) \
      --short-region /mnt/nvme8/kvd-cliff-short --short-bytes $((80 << 30)) \
      > /tmp/kvd-cliff.log 2>&1
'

# 6. Launch vLLM attached to kvd.
docker exec -d jj_vllm_gptoss bash -lc '
  GPU_IDX=4 PORT=8803 KVD_SOCKET=/tmp/kvd-cliff.sock \
    bash $INFERA_ROOT/bench/kvcache/kv_cache_cliff/launch_vllm_kvd_v2.sh \
    > /tmp/vllm-cliff-kvd-v2.log 2>&1
'

# 7. Wait + sweep.
until ssh "$NODE" 'curl -fsS --max-time 3 http://127.0.0.1:8803/v1/models' >/dev/null; do
  sleep 15
done
docker exec -w $INFERA_ROOT jj_vllm_gptoss bash -lc '
  PYTHONPATH=. python -u -m bench.kvcache.kv_cache_cliff.run_cliff \
      --endpoint http://localhost:8803 \
      --model gpt-oss-120b \
      --arm kvd_v2 \
      --isl 20000 --shared-prefix-tokens 18000 \
      --concurrencies 1,2,4,8,16,32,48,64,80,100,128,160,200,250 \
      --iters 3 --warmup-iters 2 \
      --out bench/kvcache/kv_cache_cliff/results/cliff-kvd-v2.csv
'

# === Plot ===

docker exec -w $INFERA_ROOT jj_vllm_gptoss bash -lc '
  PYTHONPATH=. python -u -m bench.kvcache.kv_cache_cliff.plot_cliff \
      bench/kvcache/kv_cache_cliff/results/cliff-vram-only.csv \
      bench/kvcache/kv_cache_cliff/results/cliff-kvd-v2.csv \
      --out bench/kvcache/kv_cache_cliff/results/cliff.png \
      --title "KV Cache Cliff — MI355X TP=1 — gpt-oss-120b MXFP4 — ISL=20K OSL=1"
'
```

## Optional: Vast NFS arm

To bench against an NFS mount instead of `/mnt/nvme8`, override
the hipfile roots in step 6:

```bash
HIPFILE_LONG=/mnt/<nfs-mount>/kvd-cliff-long \
HIPFILE_SHORT=/mnt/<nfs-mount>/kvd-cliff-short \
INFERA_KVD_GPU_DIRECT=true \
  bash launch_vllm_kvd_v2.sh
```

`INFERA_KVD_GPU_DIRECT=true` turns on hipFile for the chunk save/load
path; on Vast NFS this delivers 2-5× the bandwidth of POSIX per
`bench/kvcache/hipfile/bench0_sanity.py`.

## Warm-cache pattern

The default workload aggressively warms the shared prefix:

- Warmup pass at the lowest concurrency primes the kvd file tier
  with the 18K-token prefix (Arm B only — Arm A's VRAM cache is
  similar but evicts under load).
- All subsequent measurement requests share the SAME 18K-token
  prefix → cache-friendly. The 2K-token per-client unique suffix
  is the only part that triggers real attention computation per
  request.

When Arm A cliffs, it's because:
- vLLM's prefix cache evicts the shared prefix to fit in-flight KV
  for the per-client suffixes.
- Eviction causes the next batch of requests to re-prefill the full
  20K tokens (no longer cache-hit on the prefix).
- Prefill of 20K tokens dominates total wall time → throughput
  collapses.

Arm B should not cliff because evicted prefixes go to kvd-on-NVMe
(or kvd-on-Vast) and reload in milliseconds.

## Caveat

Numbers heavily depend on:
- vLLM's default `--cpu-offload-gb` (we don't set it; pure VRAM
  prefix cache for Arm A)
- The exact `--gpu-memory-utilization` (default 0.9 here)
- ISL × concurrency × KV-per-token (model-dependent)
- Whether the workload actually exercises prefix sharing (our shared
  prefix is identical bytes across clients, so vLLM's content-hash
  prefix cache will hit it)

The LMCache slide footnote noted the same caveats; we mirror them
here so readers know the curve is hardware/workload-specific.
