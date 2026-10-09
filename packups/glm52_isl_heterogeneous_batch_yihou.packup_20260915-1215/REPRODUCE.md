# Reproduce

Two independent halves. **The first needs no GPU, no container and no cluster access** — do it
first, because if it fails nothing downstream is worth running.

---

## Part 1 — the CPU half (2 minutes, any laptop)

```bash
cd <repo>/sglang_decode_internal_bench_and_profiling
python3 -m pytest tests -q
```

Expected:

```
60 passed, 40 subtests passed
```

Reference python was 3.13.13. No venv, no GPU, no sglang install. `bench/isl_spec.py` is stdlib-only
and `tests/test_isl_spec_yihou.py` enforces that by importing it in a subprocess and asserting
`torch` / `numpy` / `sglang` are absent from `sys.modules`.

### Part 1b — the four modes through the real CLI, still no GPU

`resolve_input_lens()` runs in the parent before `mp.spawn`, so the whole spec path is exercisable
without touching a GPU:

```bash
cd <repo>/sglang_decode_internal_bench_and_profiling
python3 - <<'PY'
import sys; sys.path.insert(0, "bench")
from profile_decode import make_parser, validate_args, resolve_input_lens
from collections import Counter

base = ["--model-path","/x","--result-dir","/tmp/yihou-x","--tp-size","8","--ep-size","1",
        "--enable-dp-attention","--batch-size","256"]

for name, extra in [
    ("uniform (back-compat)", ["--input-len","70000"]),
    ("bimodal 10%",           ["--input-len-spec","bimodal:8192,70000,0.1"]),
    ("bimodal 90%",           ["--input-len-spec","bimodal:8192,70000,0.9"]),
    ("normal",                ["--input-len-spec","normal:40000,8000,2,70000"]),
    ("list",                  ["--input-len-spec","list:"+",".join(["4096"]*16+["70000"]*16)]),
]:
    args, _ = make_parser().parse_known_args(base+extra)
    validate_args(args); resolve_input_lens(args)
    c = Counter(args.input_lens)
    print(f"{name:24s} n={len(args.input_lens):3d} distinct={len(c)} {sorted(c.items())[:3]}")
PY
```

Expected — note both ceil directions, 4/28 and 28/4:

```
uniform (back-compat)    n= 32 distinct=1 [(70000, 32)]
bimodal 10%              n= 32 distinct=2 [(8192, 4), (70000, 28)]
bimodal 90%              n= 32 distinct=2 [(8192, 28), (70000, 4)]
normal                   n= 32 distinct=32 [(29208, 1), (31337, 1), (32558, 1)]
list                     n= 32 distinct=2 [(4096, 16), (70000, 16)]
```

### Part 1c — acceptance is ISL-independent, proven on CPU

```bash
python3 - <<'PY'
import sys, random; sys.path.insert(0, "bench")
from batch_state import DecodeAccounting
BS, OSL = 32, 500
# One scalar accept length per iteration, broadcast to the whole batch -- exactly what
# upstream spec_utils.sample_simulated_acc_len does. A private RNG so the sequence is fixed.
coins = random.Random(1234)
seq = [coins.choice([3,4]) for _ in range(4000)]

def run(input_lens):
    acc = DecodeAccounting(BS, input_lens, OSL)
    for a in seq:
        if acc.complete: break
        acc.record([a]*BS, 0.001)
    s = acc.summary(1.0)
    return s["verify_iterations"], s["realized_accept_length"], s["useful_output_tokens"]

print("uniform ", run([70000]*BS))
print("bimodal ", run([8192]*8 + [70000]*24))
print("wide    ", run([2 + 1000*i for i in range(BS)]))
PY
```

All three lines must be identical: `(146, 3.4246575342465753, 16000)`.

---

## Part 2 — the GPU half

### Prerequisites

- SSH to `smci355-ccs-aus-n10-29` (or another MI355X node with the model mounted).
- The container from the previous task. Exact image digest and provenance in `environment.md`.
  If it no longer exists, rebuild from
  `packups/glm52_tp8_ep1_c256_pr51_50_54_yihou.packup_20260915-1019/`.
- Model at `/perf_apps/data/models/GLM-5.2-MXFP4`.
- **Confirm the GPUs are idle before believing any number:**
  ```bash
  rocm-smi --showpids          # want: "No KFD PIDs currently running"
  squeue -w $(hostname -s) -t R -o "%.10i %.12u %.22j %N"
  ```
  Holding a Slurm allocation on this host does **not** mean the GPUs are free — containers here
  ignore the scheduler. See `notes.md`.

### Common setup

```bash
REPO=<repo>
TOOL=$REPO/sglang_decode_internal_bench_and_profiling
WS=$REPO/temp_workspace/<your-workspace>       # outputs land in $WS/iterations
IMG=sha256:bdd783512f3db4d046fcf6e76e7039da054fe2634ad1d831a16a97da65dad656
CONT=yihou-glm52-tp8ep1-pr50-51
```

Every runner takes `--dry-run` as its first argument. **Use it before spending a startup.**

### Run A — the back-compat regression (~6 min). Run this one first.

```bash
cd "$WS" && CONTAINER=$CONT IMAGE=$IMG bash $TOOL/scripts/run_decode.sh \
  regression_uniform70k_c256_yihou \
  --tp-size 8 --ep-size 1 --enable-dp-attention \
  --batch-size 256 --max-running-requests 256 \
  --input-len 70000 --output-len 10000 --accept-length 3.61 --warmup-steps 10 \
  --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope --mem-fraction-static 0.85
```

Gate — this must match exactly, or the change is broken regardless of how the timings look:

```
realized_accept_length = 3.6134393063583814
verify_iterations      = 2768
complete               = True
```

Check it with the tool rather than by eye:

```bash
python3 $TOOL/scripts/verify_point.py --expect-ep 1 "$WS/iterations/regression_uniform70k_c256_yihou"
# verdict: pass, problems: []
```

### Run B — a ragged batch and its uniform control (~2.5 min each)

```bash
ARGS="--tp-size 8 --ep-size 1 --enable-dp-attention \
      --batch-size 64 --max-running-requests 64 --output-len 128 --accept-length 3.61 \
      --warmup-steps 2 --max-steps 20 \
      --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope --mem-fraction-static 0.85"

cd "$WS" && CONTAINER=$CONT IMAGE=$IMG bash $TOOL/scripts/run_decode.sh \
  smokeA2_bimodal_c64_yihou --input-len-spec bimodal:2048,8192,0.25 $ARGS

cd "$WS" && CONTAINER=$CONT IMAGE=$IMG bash $TOOL/scripts/run_decode.sh \
  smokeB_uniform8192_c64_yihou --input-len 8192 $ARGS
```

Then compare. Every acceptance field must be identical across the two:

```bash
python3 - <<PY
import json
A=json.load(open("$WS/iterations/smokeA2_bimodal_c64_yihou/result_yihou.json"))
B=json.load(open("$WS/iterations/smokeB_uniform8192_c64_yihou/result_yihou.json"))
for k in ("verify_iterations","raw_accept_tokens","realized_accept_length",
          "useful_output_tokens","accept_histogram","num_correct_drafts",
          "target_graph_iterations"):
    print(f"{k:26s} {A[k]!s:>22s} {B[k]!s:>22s}  {'same' if A[k]==B[k] else 'DIFFER'}")
print("A input_len/uniform", A["input_len"], A["input_len_uniform"])
print("B input_len/uniform", B["input_len"], B["input_len_uniform"])
PY
```

Expected: all seven `same`; A reports `None / False`, B reports `8192 / True`.

### Assert which kernel actually ran — not optional

```bash
grep -aiE 'Set DSA backends|declined|Loading tilelang' \
  "$WS/iterations/smokeA2_bimodal_c64_yihou/runtime.log" | sort | uniq -c
```

Expected, and matching the published baseline:

```
      1 Set DSA backends for fp8_e4m3 KV Cache: prefill=flydsl, decode=flydsl.
      8 FlyDSL sparse MLA decode declined: q shape (48, 64, 576), need (seq, 8 or 16, 576)
      8 Loading tilelang libs from dev root: /opt/tilelang/build
```

**TileLang is what executes.** A run whose log does not say this is measuring something else.

---

## Applying the change to a different checkout

`patches/modified_files_yihou.diff` is the diff of the seven modified files.
`sources/isl_spec.py` and `sources/test_isl_spec_yihou.py` are **new files not in that diff** —
copy them to `bench/` and `tests/` respectively.

```bash
cd <other-checkout>
git apply --3way <this-packup>/patches/modified_files_yihou.diff
cp <this-packup>/sources/isl_spec.py            sglang_decode_internal_bench_and_profiling/bench/
cp <this-packup>/sources/test_isl_spec_yihou.py sglang_decode_internal_bench_and_profiling/tests/
```

`sources/` also holds the final version of every modified file, so you can diff against them if the
patch does not apply cleanly.
