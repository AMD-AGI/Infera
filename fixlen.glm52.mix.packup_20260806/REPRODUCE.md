# Reproduction kit — GLM-5.2 mix fixlen sweep

Goal: reproduce the 12-point fixed-length sweep in `README.md` from a clean
machine with cluster access.

**Estimated wall clock: ~2 h 45 min.** Breakdown, measured:

| step | time |
|---|---|
| container + etcd + kvd | ~1 min |
| engine cold start to `/health` | **390 s** (weights + JIT + CUDA-graph capture) |
| smoke / feature gate | ~2 min |
| p50 arm (4 points) | ~6 min |
| p90 arm (4 points) | ~32 min |
| p99 arm (4 points) | **~1 h 40 min** — this is the long pole |

## 0. Prerequisites

**Machine.** One MI355X (gfx950) 8-GPU node. We used `chi2835`. Access:

```bash
ssh root@149.28.124.225      # jump host (chi2866)
ssh chi2835                  # the compute node
```

**Before committing to a node, read its VRAM directly.** The cluster-status
snapshot's VRAM column can lag — see `notes.md` §1.

```bash
rocm-smi --csv --showmeminfo vram | tail -8 \
  | awk -F, '{s+=$3} END {printf "VRAM in use: %.0f GB\n", s/1073741824}'
```

You need this at ~0 GB. If it is not, find the holder and **do not** displace
someone else's job. Never `scancel`; never prune images or `/tmp`.

**Secrets.** None beyond cluster SSH. See `environment.md` § "Required secrets".

**External dependencies (absolute paths, not in this repo):**

- Model + tokenizer + EAGLE draft: `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4`
- Image `infera/engine-sglang:merged-e`, id
  `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a`.
  Check it is already on the node before pulling:
  `docker images --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep engine-sglang`
- `quay.io/coreos/etcd:v3.5.14`

**The scripts in `scripts/` ARE the ones that ran** (md5-verified against the
cluster copies). No repo checkout is needed.

## 1. Stage the scripts on the node

`/mnt/vast` is shared to compute nodes; `/tmp` is not. Stage under `/mnt/vast`.

```bash
# from your workstation
scp -r scripts root@149.28.124.225:/mnt/vast/c_huggingface/glm52_mix_repro/
```

Then, on `chi2835`:

```bash
cd /mnt/vast/c_huggingface/glm52_mix_repro/scripts
chmod +x *.sh
```

## 2. Set the site values

`mix_site.sh` is **the only file carrying site values**. Everything else takes
env vars. Edit exactly two if you are on a different node:

```bash
export MY_IP=10.2.122.78     # chi2835 enp193s0f1np1 — the DATA plane.
                             # NOT the 45.76.x management address.
export MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
```

Find your node's data-plane IP with `ip -o -4 addr show` — it is the `10.2.x`
one, not `45.76.x` and not `172.17.x`.

## 3. Bring the deployment up

```bash
cd /mnt/vast/c_huggingface/glm52_mix_repro/scripts
bash mix_site.sh up 2>&1 | tee /mnt/vast/c_huggingface/glm52_mix_repro/logs/up.log
```

Five stages: container → etcd → kvd → mix worker → router. **Expect a long
silence at stage 4/5.** Cold start is 390 s and is not a hang. Compare against
`logs/up.log` in this packup; the last line should read:

```
[mix] mix ready on http://<MY_IP>:8100 after N min
```

The router is started *after* the worker is serving, deliberately: it discovers
workers out of etcd, and an empty registry at router start is a race not worth
having.

## 4. Snapshot the environment — do this while it is LIVE

Some of it (resolved engine cmdline, kvd counters) cannot be recovered later.

```bash
bash envsnap.sh > /mnt/vast/c_huggingface/glm52_mix_repro/logs/env_$(hostname).txt
```

Compare against `env/env_chi2835.txt`.

## 5. GATE — prove every feature is on, before measuring anything

```bash
bash mix_site.sh smoke 2>&1 | tee /mnt/vast/c_huggingface/glm52_mix_repro/logs/smoke.log
```

**Read the blocks, not the exit code.** Do not proceed unless all seven match:

| check | must read |
|---|---|
| `/v1/workers` | exactly 1 worker, `disagg_mode: "mixed"`, active |
| chat completion | a coherent answer. **Garbage or repeated tokens = the DSA-on-ROCm env block did not take effect** — not a sampling problem. |
| DP-attention | `dp_size=8`, `enable_dp_attention=True`, 8 live `scheduler_DP` ranks |
| MTP accept-len | median in **2–3**. A median **at 4.00 is a failure**, not a win. We read median 2.80 (n=25, p10 2.48, p90 3.08, 0 % at 4.00). |
| kv-aware | `router_policy: "kv-aware"` **and** `tokenizer-loaded > 0`. The policy line alone is insufficient — kv-aware degrades silently to load-only routing if the tokenizer did not load. |
| kvd | **8 adapters** (one per DP rank), `statctl` returns counters |
| prefix cache | `cached_tokens` non-zero on the *second* identical prompt (we read 1st `None`, 2nd 1984/2018) |

If MTP reads a median of 4.00, stop: the output has degenerated into a
repetition loop the draft model predicts perfectly, and every latency number
downstream is meaningless.

## 6. Run the sweep

```bash
export OUT=/mnt/vast/c_huggingface/glm52_mix_repro/results/fixlen
ARMS="p50 p90 p99" CONCS="1 8 16 24" \
  bash mix_bench_fixlen.sh 2>&1 \
  | tee /mnt/vast/c_huggingface/glm52_mix_repro/logs/fixlen.log
```

Arms run in **ascending ISL order on purpose** — a mis-set flag is then caught
on the 6-minute p50 arm rather than the 100-minute p99 arm.

Three flags are load-bearing in non-obvious ways (all already set in the script):

- `--random-range-ratio 1.0` pins every prompt to **exactly** ISL. The default
  draws uniformly, and the reported percentiles then mix request sizes.
- `--temperature 1.0 --top-p 0.95` — the checkpoint's own `generation_config`,
  deliberately **not** greedy. At temperature 0 this reasoning model repeats on
  a long prompt, MTP pins acceptance at 4.00, and the run reads like KV
  corruption.
- `--num-prompts = conc × 10` — the InferenceX convention
  (`benchmark_lib.sh::run_benchmark_serving`). A fixed N would leave the high
  arms with too few requests to reach steady state.

Traffic goes to the **router** (`:8100`), not the engine (`:30000`).

Run it under `nohup`/`tmux`: the p99 arm alone is ~100 minutes.

## 7. Summarise

```bash
python3 summarize_fixlen.py \
  /mnt/vast/c_huggingface/glm52_mix_repro/results/fixlen
```

Writes `summary.csv` next to the jsonl and prints the markdown table. Note it
takes the **last** json object per file: `bench_serving` *appends* to
`--output-file`, so a re-run of the same arm leaves two objects behind.

## 8. Re-derive the post-sweep counters

Both come off the live deployment / its log. Run these before tearing anything
down.

```bash
# MTP acceptance over the sweep. Scope it BY TIME WINDOW — these logs are
# appended across runs and a whole-file grep mixes in the smoke traffic and
# anything that ran afterwards.
docker exec glm52_mix bash -c "strings /tmp/glm52_mix_base.log" \
 | awk -F'[][]' '/accept len:/ {split($2,t," ");
     if (t[2]>="<sweep start HH:MM:SS>" && t[2]<="<sweep end HH:MM:SS>") print}' \
 | grep -o 'accept len: [0-9.]*' | awk '{print $3}' | sort -n | awk '
   {a[NR]=$1; if ($1>=4) f++}
   END {printf "n=%d p10=%s MEDIAN=%s p90=%s at-4.00=%d (%.1f%%)\n",
        NR, a[int(NR*0.1)+1], a[int(NR*0.5)+1], a[int(NR*0.9)+1], f, 100*f/NR}'

# kvd counters
docker exec glm52_mix python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock
```

`scripts/accept_len.sh` does the unscoped version of the first command; scope it
yourself as above if anything else has used the deployment.

Our readings (sweep window 07:13:00–09:46:00 on 2026-08-06):
`n=37,630 p10=2.85 MEDIAN=3.61 p90=3.98 at-4.00=9.0 %`, and kvd
`70,676 entries / 84.8 GB host / 68.7 GB L3 / gets 64,741 / sets 266,289 /
hits 62,697 / misses 2,044 / evictions 151,525`.

## 9. Copy the logs out

They live **inside the container**; `/tmp` is not on the shared mount.

```bash
for L in glm52_mix_base router kvd; do
  docker exec glm52_mix bash -c "gzip -c /tmp/$L.log" \
    > /mnt/vast/c_huggingface/glm52_mix_repro/logs/$L.log.gz
done
```

The engine log is ~11 MB raw / ~650 KB gzipped.

## 10. Tear down — only your own containers

```bash
bash mix_site.sh down     # reaps the engine, WAITS for VRAM to drain, removes
                          # the glm52_mix and glm52-mix-etcd containers
```

The VRAM wait is the point, not the kill: the infera wrapper exits before its
sglang child does, and that child keeps the KV-event port block bound — the next
launch then dies with "port_base is not available".

**Before removing any container you did not just create, prove it is yours:**
`docker inspect <ctr> --format '{{.Created}} {{json .HostConfig.Binds}}'`.

## Expected output

`results/fixlen/summary.csv`, 12 rows, `completed` equal to `conc × 10` on every
row and no error entries. The headline numbers to check against `README.md`:

| arm | conc | out tok/s | ttft_p50_ms | e2e_p50_ms |
|---|---|---|---|---|
| p50 | 24 | 746.75 | 2276.03 | 10312.64 |
| p90 | 24 | 1331.73 | 2690.71 | 57663.13 |
| p99 | 24 | 1935.51 | 3522.31 | 200049.47 |

Do not expect bit-identical numbers: sampling is at temperature 1.0, and the
prefix-cache hit rate on a `random` dataset is emergent, not controlled.
Throughput within a few percent and TTFT within tens of percent is a match.

## If it doesn't reproduce

See [`notes.md`](notes.md). The short list:

- Silence at stage 4/5 for several minutes is the **390 s cold start**, not a hang.
- `mix_up.sh` exiting rc 0 with no router — the `pgrep -f` self-kill. Fixed in
  the shipped `mix_common.sh`; `notes.md` §2 explains what to look for if it
  resurfaces.
- Model returns 200s with garbage → the DSA env block did not take effect.
- MTP acceptance pinned at 4.00 → check you did not set temperature 0.
- Second launch dies with "port_base is not available" → the previous sglang
  child is still alive. Use `reap`, do not just `docker restart`.
- Grepping the engine log without a time window will mix in other runs.
