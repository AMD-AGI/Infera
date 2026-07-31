# Raw engine logs — the baseline (control) run

Both PD legs, copied verbatim off the shared FS
(`/mnt/vast/c_huggingface/glm52_kvexp/`, visible from both nodes, which is why
there is one copy rather than one per host). Neither is gzipped.

| File | Node | Role | Size |
|---|---|---|---|
| `pd_prefill_base.log` | chi2879 (10.2.122.10) | prefill TP8, gmu 0.88 | 193 KB |
| `pd_decode_base.log` | chi2867 (10.2.122.44) | decode TP8, gmu 0.85 | 400 KB |

Run 2026-07-30, starting ~10:00. Configuration: **kvaware OFF, kvd OFF,
router-policy round-robin** — the control.

## Grep recipes — every claim in `results/`, straight from these files

The point of listing the exact commands is that a reviewer can audit rather
than trust. All of them are runnable from this directory.

**It was real mooncake RDMA, not the `MC_FORCE_TCP` fallback:**

```bash
grep -ac 'MC_FORCE_TCP'        pd_prefill_base.log   # 0  -> not the TCP fallback
grep -ac 'HIP dmabuf disabled' pd_prefill_base.log   # 8  -> one per active ionic rail
grep -ac 'HIP dmabuf disabled' pd_decode_base.log    # 8
```

Eight is the number that matters: one line per ionic NIC means the mooncake RDMA
path was engaged on all 8 rails. A degraded/TCP run does not produce them.

**The switches really were off — in the engine, not just on the command line:**

```bash
grep -ac 'kv-events-config'             pd_prefill_base.log   # 0
grep -ac 'infera-kvd adapter connected' pd_prefill_base.log   # 0
grep -ac 'HiRadixCache'                 pd_prefill_base.log   # 0
grep -aoE 'enable_hierarchical_cache=[A-Za-z]+' pd_prefill_base.log | sort -u
                                                              # enable_hierarchical_cache=False
grep -aoE 'hicache_storage_backend=[^,]*'       pd_prefill_base.log | sort -u
                                                              # hicache_storage_backend=None
grep -ah 'Tree cache initialized' pd_prefill_base.log | head -1
# ... impl=RadixCache ... hierarchical=False ...
```

**PD + DP-attention symmetric on both legs** (asymmetric DPA would mismatch the
KV shard layout across the mooncake transfer, so this is a precondition, not a
detail):

```bash
for f in pd_prefill_base.log pd_decode_base.log; do
  echo "$f:"
  grep -aoE "disaggregation_mode='[a-z]+'|disaggregation_transfer_backend='[a-z]+'|enable_dp_attention=True|dp_size=8|ep_size=8" "$f" | sort -u
done
```

**Geometry and readiness:**

```bash
grep -aoE 'mem_fraction_static=[0-9.]+|tp_size=[0-9]+' pd_prefill_base.log | sort -u  # 0.88, 8
grep -aoE 'mem_fraction_static=[0-9.]+|tp_size=[0-9]+' pd_decode_base.log  | sort -u  # 0.85, 8
grep -ac 'ready to roll' pd_prefill_base.log   # 1
grep -ac 'ready to roll' pd_decode_base.log    # 1
```

**The decode leg's self-imposed radix disable** — visible here, and the reason a
hierarchical cache is illegal on a decode leg without something legalising it:

```bash
grep -aoE 'disable_radix_cache=[A-Za-z]+' pd_decode_base.log  | sort -u   # True
grep -aoE 'disable_radix_cache=[A-Za-z]+' pd_prefill_base.log | sort -u   # False
```

## What these logs do NOT contain

- **The router log** (`/tmp/router.log`). It lived inside the container, which
  was removed at teardown. The one thing that matters from it — both workers
  registering with `"kv_events_endpoint":null` — was captured in-session and is
  quoted in `results/switches_were_off.txt`. It is **not** re-derivable from the
  files here, and that is a real gap rather than a formality: the leg logs show
  kvaware was never wired, but only the router view shows how the *router* saw
  it.
- **kvd daemon logs.** There is no kvd daemon in this run; kvd is OFF. Nothing
  is missing.
- **The probe transcript as a file.** The 4/4 output was captured in-session and
  is quoted verbatim in `results/baseline_probe_4of4.txt`, including the full
  completion text for all four prompts.
- **Per-GPU VRAM at the moment of the run.** Not captured for this round.

## Regenerating

```bash
bash ../scripts/run.sh
```

~6 min cold start plus a minute of probing, 16 GPUs across two nodes. It writes
`results/baseline_probe_4of4.observed.txt`. To keep the whole log files rather
than the extracts, pull them before the script's `trap cleanup EXIT` fires — or
just run with `KEEP=1`, which leaves the deployment up:

```bash
KEEP=1 bash ../scripts/run.sh
# then, on the prefill node:
cp /mnt/vast/c_huggingface/glm52_base06/pd_*_base.log ./
```
