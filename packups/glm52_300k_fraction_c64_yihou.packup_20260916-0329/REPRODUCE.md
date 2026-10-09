# Reproduce

Total machine time ~30 minutes for the six points, plus model load per run (~2 min each).

## 0. Prerequisites

| | |
|---|---|
| node | `smci355-ccs-aus-n10-29`, 8x MI355X (see `environment.md`) |
| container | `yihou-glm52-tp8ep1-pr50-51` |
| image | `sha256:bdd783512f3db4d046fcf6e76e7039da054fe2634ad1d831a16a97da65dad656` |
| model | `/perf_apps/data/models/GLM-5.2-MXFP4` (absolute path on the shared mount) |
| code | repo at `bce29162` or later — `--input-len-spec` must exist |

No secrets. Access needs SSH to the node and membership of `docker`, `video`, `render`.

If the container is gone, rebuild the image from
`packups/glm52_tp8_ep1_c256_pr51_50_54_yihou.packup_20260915-1019/`, then create the container with
that packup's `create_container_n10_29_yihou.sh`.

## 1. Confirm the code has the feature

```bash
cd <repo>/sglang_decode_internal_bench_and_profiling
python3 -m pytest tests -q          # 60 passed, 40 subtests passed
```

No GPU, no container, no venv needed for this step.

## 2. Confirm the GPUs are actually free

**Do this every time. Holding the Slurm allocation does not mean the GPUs are idle** — containers
on this host ignore the scheduler, and a colleague's container did appear mid-session while this
work was running.

```bash
ssh smci355-ccs-aus-n10-29 'rocm-smi --showpids'     # want: No KFD PIDs currently running
ssh smci355-ccs-aus-n10-29 'squeue -w $(hostname -s) -t R -o "%.10i %.12u %.22j %N"'
```

## 3. Run the series

```bash
cd <repo>
bash temp_workspace/<your-workspace>/plan/run_300k_fraction_series_yihou.sh
```

The copy in `scripts/run_300k_fraction_series_yihou.sh` is the exact script that produced this
data. Edit `REPO` / `WS` / `CONT` / `IMG` at the top for your paths. It writes results into
`$WS/iterations/<name>/` and tenancy captures into `$WS/logs/smi_*_yihou.txt`.

**Do not reorder the series into 0 → 100.** The order `50, 0, 100, 25, 75, 12.5` is the experiment
design, not an accident: run in ascending order, a drift trend would be collinear with the effect
and the result would be uninterpretable. The script says so in a comment; keep it.

**Do not reflow `COMMON` onto several lines.** It is interpolated into a double-quoted `ssh "…"`
command, so embedded newlines become command separators on the remote shell. That silently dropped
four flags on the first attempt (`notes.md`).

### What each point is

`local_batch_size = 8` at C=64 / dp=8, so the share lands on whole requests. The spec describes one
attention-DP rank; every rank receives the same multiset.

| share | spec |
|---|---|
| 0 % | `--input-len 70000` |
| 12.5 % | `--input-len-spec list:300000,70000,70000,70000,70000,70000,70000,70000` |
| 25 % | `--input-len-spec list:300000,300000,70000,70000,70000,70000,70000,70000` |
| 50 % | `--input-len-spec list:300000,300000,300000,300000,70000,70000,70000,70000` |
| 75 % | `--input-len-spec list:300000,300000,300000,300000,300000,300000,70000,70000` |
| 100 % | `--input-len 300000` |

Common to all six:

```
--tp-size 8 --ep-size 1 --enable-dp-attention --batch-size 64 --max-running-requests 64
--output-len 10000 --accept-length 3.61 --warmup-steps 10
--enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope --mem-fraction-static 0.85
```

`--max-running-requests 64` is **mandatory, not tuning**. At dp=8 the default of 48 sizes the
per-worker `ReqToTokenPool` to 6 slots and `alloc_req_slots` fails with a message about KV bytes
that is really about slot count.

## 4. Check each run before believing it

The driver does all three automatically and prints them; verify by hand if you run points
individually.

**a. The arguments reached the binary.**

```bash
python3 - <<'PY'
import json
c = json.load(open("<point>/config_yihou.json"))
b, cli = c["benchmark"], " ".join(c["server_cli"])
assert b["warmup_steps"] == 10, b["warmup_steps"]
assert b["batch_size"] == 64
for f in ("--enable-aiter-allreduce-fusion", "--enable-fused-qk-norm-rope"):
    assert f in cli, f
assert "--mem-fraction-static 0.85" in cli
print("ARG CHECK ok")
PY
```

**b. The acceptance gate.** Every point must give exactly:

```
realized_accept_length = 3.6134393063583814
verify_iterations      = 2768
complete               = True
```

A point that misses this is measuring something else and its TPOT is not comparable. Note this gate
is **unaffected by ISL** — acceptance is one scalar per iteration broadcast across the batch, and
all requests share `output_len`, so they finish on the same iteration whatever their prefix.

**c. Which kernel ran.**

```bash
grep -aiE 'Set DSA backends|declined|Loading tilelang' <point>/runtime.log | sort | uniq -c
```

Expected:

```
      1 Set DSA backends for fp8_e4m3 KV Cache: prefill=flydsl, decode=flydsl.
      8 FlyDSL sparse MLA decode declined: q shape (48, 64, 576), need (seq, 8 or 16, 576)
      8 Loading tilelang libs from dev root: /opt/tilelang/build
```

**TileLang is what executes.** FlyDSL is requested and declines, because DP attention gives every
rank all 64 q heads against a gate of 8 or 16. A log that does not say this is a different
experiment.

## 5. Reproduce the table

```bash
python3 - <<'PY'
import json
pts = [("0%",   "s2_pct0_yihou",   2), ("12.5%", "s6_pct12_yihou",  6),
       ("25%",  "s4_pct25_yihou",  4), ("50%",   "s1_pct50_yihou",  1),
       ("75%",  "s5_pct75_yihou",  5), ("100%",  "s3_pct100_yihou", 3)]
rows = []
for lab, n, order in pts:
    r = json.load(open(f"<iterations>/{n}/result_yihou.json"))
    rows.append((lab, order, r["effective_token_latency_ms_per_user"],
                 r["realized_accept_length"], r["verify_iterations"], r["complete"]))
base = rows[0][2]
for lab, order, t, acc, it, comp in rows:
    gate = abs(acc - 3.6134393063583814) < 1e-12
    print(f"{lab:>6} order={order} TPOT={t:8.4f} vs0={100*(t-base)/base:+6.2f}% gate={gate} iters={it}")
print("monotone:", all(rows[i][2] < rows[i+1][2] for i in range(5)))
PY
```

Expected: the table in `README.md`, and `monotone: True`.

## 6. Tenancy audit over the whole series

```bash
for f in <ws>/logs/smi_s*_yihou.txt; do
  grep -q 'No KFD PIDs' "$f" && echo "$(basename $f): clean" || echo "$(basename $f): *** OTHER PIDS ***"
done
```

All twelve must be clean. A point measured while someone else held a GPU is not comparable and
should be rerun, not corrected.
