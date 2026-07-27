#!/usr/bin/env bash
# Build a first-N-layer copy of a DSv4 model dir (config-only truncation).
#
# Why this works: SGLang builds exactly `num_hidden_layers` decoder layers via
# make_layers(); the weight loader auto-skips any checkpoint tensor whose
# layer_id >= end_layer, and the MTP/nextn layer (id == num_hidden_layers) too.
# So a full 61-layer checkpoint loads cleanly into an N-layer model with NO
# SGLang source change. Only `num_hidden_layers` matters. Do NOT touch
# `compress_ratios` (len = 61 + 1 MTP): it is indexed by layer_id, so leaving it
# full is correct — indices 0..N-1 are used, the rest ignored. No length assert.
#
# Run this ON the node (chi2879) where /mnt/vast is mounted.
# Usage: bash make_first_n_config.sh [N] [SRC_DIR] [DST_DIR]
set -u
N="${1:-4}"
SRC="${2:-/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro}"
DST="${3:-/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-${N}L}"

python3 - "$N" "$SRC" "$DST" <<'PY'
import json, os, sys
N, SRC, DST = int(sys.argv[1]), sys.argv[2], sys.argv[3]
os.makedirs(DST, exist_ok=True)
# symlink every entry except config.json (we rewrite that one real file)
for name in os.listdir(SRC):
    if name == "config.json":
        continue
    d = os.path.join(DST, name)
    if os.path.islink(d) or os.path.exists(d):
        continue
    os.symlink(os.path.join(SRC, name), d)
c = json.load(open(os.path.join(SRC, "config.json")))
orig = c["num_hidden_layers"]
c["num_hidden_layers"] = N
json.dump(c, open(os.path.join(DST, "config.json"), "w"), indent=2)
n_link = sum(1 for x in os.listdir(DST) if os.path.islink(os.path.join(DST, x)))
print(f"[make_first_n] {SRC} ({orig} layers) -> {DST} ({N} layers)")
print(f"[make_first_n] symlinks={n_link} config.json=real "
      f"compress_ratios_len={len(c['compress_ratios'])} (left untouched)")
PY
