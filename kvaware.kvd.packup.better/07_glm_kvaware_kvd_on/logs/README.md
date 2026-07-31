# Raw engine logs — the kvaware+kvd ON run

Both PD legs, copied verbatim off the shared FS
(`/mnt/vast/c_huggingface/glm52_kvexp/`, visible from both nodes, which is why
there is one copy rather than one per host). Neither is gzipped.

| File | Node | Role | Size |
|---|---|---|---|
| `pd_prefill_kv.log` | chi2879 (10.2.122.10) | prefill TP8, gmu 0.88, KVAWARE=1 KVD=1 | 294 KB |
| `pd_decode_kv.log` | chi2867 (10.2.122.44) | decode TP8, gmu 0.85, KVAWARE=1 KVD=1 | 443 KB |

Run 2026-07-30, ~11:20–11:35. Router policy `kv-aware`, `infera.kvd` daemon on
both nodes, transport real mooncake RDMA.

## Grep recipes — every claim in `results/`, straight from these files

Runnable from this directory.

**kvd was wired on every DP rank of both legs:**

```bash
grep -ac 'infera-kvd adapter connected' pd_prefill_kv.log   # 8  (one per DP rank)
grep -ac 'infera-kvd adapter connected' pd_decode_kv.log    # 8
```

**infera's own KV probe plane attached** — this line is the one that refuted an
earlier hypothesis (see below):

```bash
grep -ac 'KV plane up:' pd_prefill_kv.log   # 1
grep -ah 'KV plane up:' pd_prefill_kv.log
# INFO:__main__:KV plane up: events_bind=tcp://0.0.0.0:5557
#   events_advertise=tcp://10.2.122.10:5557 snapshot=http://10.2.122.10:8801
#   engine_block_size=64 index_block_size=64
grep -ah 'KV plane up:' pd_decode_kv.log
# ... events_advertise=tcp://10.2.122.44:5557 snapshot=http://10.2.122.44:8801 ...
```

**The host pool was bounded by `--hicache-size`, not the 2.0 ratio:**

```bash
grep -a 'Allocating .* host memory for hierarchical' pd_prefill_kv.log
# 8 lines, one per DP rank, each "Allocating 16.00 GB host memory for
# hierarchical KV cache." — timestamps 11:21:32/33.
```

Eight × 16 GB = 128 GB of host RAM on a 3023 GB box. Left to the default
`--hicache-ratio 2.0` this figure is computed off `max_total_num_tokens` and can
run to hundreds of GB *per rank*.

**The decode leg is only legal because infera auto-appended the radix flag:**

```bash
grep -ac 'disaggregation-decode-enable-radix-cache' pd_decode_kv.log   # 1
grep -aoE 'disable_radix_cache=[A-Za-z]+' pd_decode_kv.log | sort -u   # False
```

A PD decode leg sets `disable_radix_cache=True` by itself, and sglang forbids
`enable-hierarchical-cache` alongside it. infera appends
`--disaggregation-decode-enable-radix-cache` — but only when kv-events are on.
So turning kvaware off silently disables kvd on the decode leg.

**Patch 0001 held — the two legs got different kv-events port bases:**

```bash
grep -aoE '"endpoint": "tcp://\*:[0-9]+"' pd_prefill_kv.log | sort -u  # tcp://*:25075
grep -aoE '"endpoint": "tcp://\*:[0-9]+"' pd_decode_kv.log  | sort -u  # tcp://*:1649
```

Two independent draws from the randomised scan. Pre-fix both would have been
`32764` and the second leg would have died with `ZMQError: Address already in
use`. (Here they are on separate hosts so it would not actually have collided —
but the *values* are the evidence that the patched code path ran.)

**Hierarchical cache really is on, unlike the switches-off run:**

```bash
grep -aoE 'enable_hierarchical_cache=[A-Za-z]+' pd_prefill_kv.log | sort -u
#   enable_hierarchical_cache=True
grep -aoE 'hicache_storage_backend=[^,]*'       pd_prefill_kv.log | sort -u
#   hicache_storage_backend='dynamic'
grep -ah 'Tree cache initialized' pd_prefill_kv.log | head -1
#   ... impl=HiRadixCache ... hierarchical=True ...
```

**PD + DP-attention symmetric on both legs, and it was real RDMA:**

```bash
for f in pd_prefill_kv.log pd_decode_kv.log; do
  echo "$f:"
  grep -aoE "disaggregation_mode='[a-z]+'|disaggregation_transfer_backend='[a-z]+'|enable_dp_attention=True|dp_size=8|ep_size=8" "$f" | sort -u
done
grep -ac 'MC_FORCE_TCP'        pd_prefill_kv.log   # 0
grep -ac 'HIP dmabuf disabled' pd_prefill_kv.log   # 8
grep -ac 'ready to roll'       pd_prefill_kv.log   # 1
grep -ac 'ready to roll'       pd_decode_kv.log    # 1
```

## What these logs do NOT contain

- **The kvd counters.** They are the most important negative result of this
  experiment (all zero) and they are **not in these files** — they were read
  from the daemon over its unix socket with `scripts/kvdstats.sh`. Quoted
  verbatim in `results/kvd_served_zero_traffic.txt`. Grepping these logs for
  `gets_total` will find nothing; that is expected, not a missing file.
- **kvd daemon logs** (`/tmp/kvd.log`, container-local; the containers were
  removed at teardown). The daemon's own view — arena negotiation, L3
  self-check, socket bind — is therefore unavailable for *this* run. Equivalent
  lines from an earlier Qwen3 round survive in
  `results/kvaware_kvd_activation_evidence.txt`, clearly marked as such.
- **The router log** (`/tmp/router.log`, same reason). The worker registrations
  with their non-null `kv_events_endpoint` values (`tcp://10.2.122.10:25075`,
  `tcp://10.2.122.44:1649`) were captured in-session and are quoted in
  `results/step1_kvaware_kvd_4of4.txt`. Note those are the *advertised* infera
  endpoints; the `tcp://*:25075` / `tcp://*:1649` bases greppable above are the
  same numbers seen from the sglang side, which is what makes the two records
  cross-checkable.
- **The probe transcript as a file.** Captured in-session, quoted in full
  (including the completion text) in `results/step1_kvaware_kvd_4of4.txt`.

**Provenance warning on one results file.** Parts of
`results/kvaware_kvd_activation_evidence.txt` come from the **Qwen3-1.7B
single-node** MVP round, not from this GLM-5.2 run — the file says so at the
top, and the tell is `model=qwen3` / `--dp-size 4` / `tcp://*:32764` in those
excerpts. That container was removed and its logs are gone. Anything in that
file matching the two `.log` files here is re-derivable; the Qwen3 excerpts are
not.

## Regenerating

```bash
bash ../scripts/run.sh
```

~6 min cold start plus probing, 16 GPUs across two nodes, plus a kvd daemon per
node. Writes `results/step1_kvaware_kvd_4of4.observed.txt`, which includes the
kvd counters both before and after the probe. To keep the whole log files:

```bash
KEEP=1 bash ../scripts/run.sh
# then, on the prefill node:
cp /mnt/vast/c_huggingface/glm52_kv07/pd_*_kv.log ./
# and, inside either container, the daemon log this packup is missing:
docker exec glm52_kv07 cat /tmp/kvd.log > kvd_prefill.log
```
