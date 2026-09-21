#!/usr/bin/env bash
# Purpose: settle the OPEN question "does this shape need
#   --disable-custom-all-reduce?" for one deployed shape (P8D8 or P4D4). Sends
#   ONE fixed prompt 16 times at temperature=0, prints every completion so
#   garbled/degenerate MTP output is visible, and reads per-DP-rank
#   spec_accept_length / spec_accept_rate from the decode engine's /metrics
#   UNDER LOAD (an idle gauge reads 0 and is not a measurement).
# Usage: ./probe_mtp.yihou.sh <out_dir>
#   The deployment under test MUST be launched with simulation OFF
#   (DECODE_SIMULATE_ACC_LEN= on the launch command line); simulated acceptance
#   masks garbling and inflates the acceptance gauge, so it proves nothing about
#   correctness. Override ROUTER / DECODE_HOST / DECODE_PORT / MODEL_NAME if the
#   topology differs from topology.yihou.tsv.
# Artifacts: <out_dir>/req-NN.json (16), metrics-{idle,loaded,post}.txt, verdict
#   printed to stdout.
set -euo pipefail

OUT="${1:?usage: probe_mtp.yihou.sh <out_dir>}"
mkdir -p "$OUT"

# Defaults from scripts/topology.yihou.tsv + config.full.sh ports:
#   prefill/router node 137 = 10.245.153.247, ROUTER_PORT 28000
#   decode node 136 = 10.245.154.168, engine port = ENGINE_PORT_BASE(29001) +
#   decode's topology row index (1) = 29002.
ROUTER="${ROUTER:-http://10.245.153.247:28000}"
DECODE_HOST="${DECODE_HOST:-10.245.154.168}"
DECODE_PORT="${DECODE_PORT:-29002}"
MODEL="${MODEL_NAME:-glm5.2-mxfp4}"
N="${N:-16}"
MAXTOK="${MAXTOK:-256}"
# One fixed, deterministic prompt. At temperature 0 a healthy stack returns the
# same SEMANTIC answer all 16 times. NOTE: they are NOT byte-identical -- under DP
# attention and batching, concurrent requests reduce in different orders, and a
# healthy run here produced 16 DIFFERENT strings. Judge on the content check below
# (first ten primes, in order), on empty count, and on degenerate tails -- NEVER on
# byte-identity, which was measured to be false on a healthy stack.
PROMPT="${PROMPT:-List the first 10 prime numbers, separated by commas, then explain in one sentence why 1 is not prime.}"

say() { printf '\n=== %s ===\n' "$*"; }

ask() {  # ask <outfile>
    curl -sS --max-time 180 "$ROUTER/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],
                  "messages":[{"role":"user","content":sys.argv[2]}],
                  "temperature":0,"max_tokens":int(sys.argv[3])}))' \
            "$MODEL" "$PROMPT" "$MAXTOK")" >"$1"
}

scrape() {  # scrape <outfile>
    curl -sS --max-time 30 "http://$DECODE_HOST:$DECODE_PORT/metrics" >"$1" \
        || echo "  (scrape failed)" >&2
}

# --- 1. idle snapshot (proves the trap: idle ranks read 0) -------------------
say "metrics BEFORE load (expected uninformative)"
scrape "$OUT/metrics-idle.txt"
grep -E 'spec_accept_(length|rate)' "$OUT/metrics-idle.txt" | head -20 || true

# --- 2. fire N identical requests concurrently, scrape mid-flight ------------
say "sending $N identical temperature=0 requests (concurrent = decode under load)"
pids=()
for i in $(seq 1 "$N"); do
    ask "$OUT/req-$(printf '%02d' "$i").json" &
    pids+=("$!")
done
# Let decode get busy, then scrape while requests are still in flight.
sleep 3
say "metrics UNDER load (the measurement)"
scrape "$OUT/metrics-loaded.txt"
grep -E 'spec_accept_(length|rate)' "$OUT/metrics-loaded.txt" || true
# Wait for all requests to finish.
for p in "${pids[@]}"; do wait "$p" || true; done
# A second scrape after completion: the acceptance gauges are running averages,
# so this is a fallback if the mid-flight scrape caught a rank before it served.
scrape "$OUT/metrics-post.txt"

# --- 3. print every completion + coherence verdict ---------------------------
say "completions (temperature=0; judge CONTENT, not byte-identity -- see NOTE)"
python3 - "$OUT" "$N" <<'PY'
import glob, json, os, sys
out, n = sys.argv[1], int(sys.argv[2])
def body(p):
    try:
        d = json.load(open(p))
        m = d["choices"][0]["message"]
        return (m.get("content") or ""), (m.get("reasoning_content") or "")
    except Exception as e:
        return f"<PARSE ERROR: {e}>", ""
texts = []
for p in sorted(glob.glob(os.path.join(out, "req-*.json"))):
    c, r = body(p)
    # JUDGE content + reasoning_content TOGETHER. GLM-5.2 runs with
    # --reasoning-parser glm45, so the model's text goes to reasoning_content
    # first and `content` stays EMPTY until the reasoning block closes. Judging
    # `content` alone made the first probe run report GARBLED against 16
    # perfectly coherent reasoning traces -- a checker looking at the wrong
    # field, which is worse than no checker. If the answer must appear in
    # `content`, raise MAXTOK; do not narrow this back to `content`.
    texts.append((c + "\n" + r).strip())
    tag = os.path.basename(p)
    print(f"--- {tag} ---")
    print(f"  content   = {c!r}")
    if r:
        print(f"  reasoning = {r[:200]!r}{' …' if len(r) > 200 else ''}")
print("\n--- coherence ---")
empty = sum(1 for t in texts if not t.strip())
distinct = sorted(set(texts))
print(f"  requests            : {len(texts)}")
print(f"  empty completions   : {empty}")
print(f"  distinct completions: {len(distinct)}  (temperature=0; see NOTE below)")
# Degenerate-tail signature: a completion that is one character/token repeated.
def degenerate(t):
    s = t.strip()
    return len(s) > 4 and len(set(s.replace(' ', ''))) <= 1
deg = sum(1 for t in texts if degenerate(t))
print(f"  degenerate-tail     : {deg}")

# CONTENT check -- the primary discriminator, added by the leader.
# Rationale: the observed GLM-5.2 MTP garbled signature is "first token correct,
# every later token degenerate" producing MIXED garbage such as
#   '1engaalog8effi...curaereziTlutuga8'
# which the single-repeated-character test above does NOT catch. The prompt asks
# for the first 10 primes, so a healthy completion must contain that sequence in
# order; a corrupted speculative path cannot fake it. This is specific to the
# PROMPT above -- if you change the prompt, change this check with it.
PRIMES = ["2", "3", "5", "7", "11", "13", "17", "19", "23", "29"]
def has_primes(t):
    pos, i = -1, 0
    for p in PRIMES:
        j = t.find(p, pos + 1)
        if j < 0:
            return False
        pos = j
    return True
good = sum(1 for t in texts if has_primes(t))
print(f"  contain 1st-10-primes in order: {good}/{len(texts)}   <-- PRIMARY CHECK")

verdict = "COHERENT" if (empty == 0 and deg == 0 and good == len(texts)) else \
          "GARBLED / SUSPECT -- inspect the completions printed above"
print(f"  VERDICT             : {verdict}")
print("  NOTE: 'distinct completions > 1' alone is NOT evidence of garbling.")
print("  With DP attention and batching, concurrent requests can land in")
print("  different batch compositions and reduce in different orders, so a")
print("  healthy stack may still differ across the 16. Judge on the content")
print("  check, the empty count, and your own reading of the text above.")
PY

# --- 4. per-DP-rank acceptance, read the right way ---------------------------
say "per-rank spec_accept_length / spec_accept_rate (under-load scrape)"
python3 - "$OUT" <<'PY'
import os, re, sys
out = sys.argv[1]
def parse(path):
    ranks = {}
    if not os.path.exists(path):
        return ranks
    for line in open(path):
        m = re.match(r'sglang:spec_accept_(length|rate)\{([^}]*)\}\s+([\d.eE+-]+)', line)
        if not m:
            continue
        metric, labels, val = m.group(1), m.group(2), float(m.group(3))
        # DP rank appears as a label (e.g. dp_rank / engine ... ); key on the
        # whole label set so ranks stay distinct regardless of label name.
        ranks.setdefault(labels, {})[metric] = val
    return ranks
loaded = parse(os.path.join(out, "metrics-loaded.txt"))
post   = parse(os.path.join(out, "metrics-post.txt"))
# Take, per rank per metric, the LARGER of the two scrapes.
#
# The previous form was `{**post, **loaded}`, which lets `loaded` overwrite
# `post` wholesale. These gauges are only refreshed when a rank finishes a
# batch, so a scrape taken 3 s in reads 0.0 on every rank -- and those zeros
# then clobbered a perfectly good post-run scrape. The first probe run
# reported "no rank reported non-zero acceptance" while metrics-post.txt on
# disk held 3.81-4.42 on all eight ranks. Max() cannot lose a real reading to
# a not-yet-populated one, in either direction.
merged = {}
for src in (post, loaded):
    for labels, d in src.items():
        tgt = merged.setdefault(labels, {})
        for k, v in d.items():
            if v > tgt.get(k, 0.0):
                tgt[k] = v
active = {k: v for k, v in merged.items()
          if v.get("length", 0) > 0 or v.get("rate", 0) > 0}
if not active:
    print("  no rank reported non-zero acceptance — INCONCLUSIVE, not a zero.")
    print("  (check the deployment is up, simulation is OFF, and MTP is on.)")
    raise SystemExit(0)
lengths = []
for labels, v in sorted(active.items()):
    ln, rt = v.get("length"), v.get("rate")
    if ln is not None:
        lengths.append(ln)
    print(f"  [{labels}]  accept_length={ln}  accept_rate={rt}")
if lengths:
    lo = min(lengths)
    print(f"\n  active ranks: {len(active)}   min accept_length: {lo}")
    print("  " + ("PASS (>= 2.0)" if lo >= 2.0 else "LOW (< 2.0) — MTP barely accepting"))
    print("  NOTE: acceptance is a throughput signal; the coherence verdict")
    print("  above is what decides the --disable-custom-all-reduce question.")
PY
