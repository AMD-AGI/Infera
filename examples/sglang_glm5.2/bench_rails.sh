#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# A/B the Mooncake rail choice: one shared rail (IB_DEVICE=rdma0) vs Mooncake's
# per-GPU NUMA-affine auto-discovery (IB_DEVICE=""). Run this once per config,
# tagging each run, then diff the two with `compare` below.
#
#   TAG=rdma0 bash bench_rails.sh run        # with the legs on the pinned rail
#   ... restart both legs with IB_DEVICE="" ...
#   TAG=auto  bash bench_rails.sh run
#   bash bench_rails.sh compare rdma0 auto
#
# The load is skewed towards prefill (long input, short output) on purpose: that
# is the regime where the KV hand-off shows up in TTFT instead of being hidden
# behind decode time.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/wekafs/models/GLM-5.2-FP8}"
ISL="${ISL:-8192}"
OSL="${OSL:-128}"
CONC="${CONC:-16 64 256}"
TAG="${TAG:-rdma0}"
OUT="${OUT:-$HERE/bench_rail_${TAG}}"

PREFILL_LOG="${PREFILL_LOG:-$HERE/infera_2_sglang_prefill.log}"
DECODE_LOG="${DECODE_LOG:-$HERE/infera_3_sglang_decode.log}"
# Mooncake reports an unroutable QP as a plain errno 110 from the RTR transition,
# usually only once concurrency is high enough to open QPs on every rail pair.
ERR_RE='errno 110|Connection timed out|Failed to modify QP|ibv_modify_qp|failed to (connect|transfer)|TransferEngine.*(error|fail)'

run() {
    mkdir -p "$OUT"
    # Watermark the logs so the error scan only counts this run's failures.
    wc -l <"$PREFILL_LOG" >"$OUT/.prefill_mark"
    wc -l <"$DECODE_LOG" >"$OUT/.decode_mark" 2>/dev/null || echo 0 >"$OUT/.decode_mark"

    for c in $CONC; do
        name="isl${ISL}_osl${OSL}_c${c}"
        echo "[rails:$TAG] $name -> $OUT/$name.json"
        python3 -m sglang.bench_serving \
            --backend sglang-oai --host "$HOST" --port "$PORT" \
            --model "$MODEL" --tokenizer "$MODEL" \
            --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
            --random-range-ratio 1.0 \
            --num-prompts "$((c * 5))" --max-concurrency "$c" --request-rate inf \
            --output-file "$OUT/$name.json" 2>&1 | tee "$OUT/$name.log"
    done

    echo "[rails:$TAG] scanning legs for RDMA transfer failures"
    tail -n "+$(($(cat "$OUT/.prefill_mark") + 1))" "$PREFILL_LOG" \
        | grep -Ei "$ERR_RE" >"$OUT/rdma_errors_prefill.txt" || true
    tail -n "+$(($(cat "$OUT/.decode_mark") + 1))" "$DECODE_LOG" \
        | grep -Ei "$ERR_RE" >"$OUT/rdma_errors_decode.txt" || true
    printf '[rails:%s] prefill errors: %s, decode errors: %s\n' "$TAG" \
        "$(wc -l <"$OUT/rdma_errors_prefill.txt")" \
        "$(wc -l <"$OUT/rdma_errors_decode.txt")"
    echo "[rails:$TAG] results in $OUT"
}

compare() {
    python3 - "$HERE" "$@" <<'PY'
import json, pathlib, sys

here, tags = pathlib.Path(sys.argv[1]), sys.argv[2:]
runs = {}
for tag in tags:
    d = here / f"bench_rail_{tag}"
    for f in sorted(d.glob("isl*.json")):
        runs.setdefault(f.stem, {})[tag] = json.loads(f.read_text().splitlines()[-1])
    errs = sum(
        len((d / n).read_text().splitlines())
        for n in ("rdma_errors_prefill.txt", "rdma_errors_decode.txt")
        if (d / n).exists()
    )
    print(f"{tag}: {errs} RDMA transfer errors during bench")

cols = [("mean_ttft_ms", "TTFT mean"), ("p99_ttft_ms", "TTFT p99"),
        ("mean_tpot_ms", "TPOT mean"), ("output_throughput", "out tok/s")]
for name, byTag in runs.items():
    print(f"\n{name}")
    print(f"  {'metric':<12}" + "".join(f"{t:>14}" for t in tags) + f"{'delta':>12}")
    for key, label in cols:
        vals = [byTag.get(t, {}).get(key) for t in tags]
        row = "".join(f"{v:>14.1f}" if isinstance(v, (int, float)) else f"{'-':>14}" for v in vals)
        delta = ""
        if len(vals) == 2 and all(isinstance(v, (int, float)) for v in vals) and vals[0]:
            delta = f"{(vals[1] - vals[0]) / vals[0] * 100:+11.1f}%"
        print(f"  {label:<12}{row}{delta:>12}")
PY
}

case "${1:-run}" in
    run) run ;;
    compare) shift; compare "$@" ;;
    *) echo "usage: [TAG=...] bash $0 run | bash $0 compare <tagA> <tagB>" >&2; exit 2 ;;
esac
