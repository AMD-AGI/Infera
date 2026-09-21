#!/usr/bin/env bash
# Purpose: continuously sample MTP acceptance from a decode engine and append a
#   tidy CSV time series, so acceptance can be correlated with what the service
#   was doing at the time instead of being read once at the end.
# Usage: ./mtp_sampler.yihou.sh <out_csv> [interval_s]
#   Stop it with: kill <pid>  (it also stops when <out_csv>.stop appears)
# Artifacts: <out_csv> with columns
#   ts_utc,epoch,dp_rank,accept_length,accept_rate,verify_calls,running_req,label
# Notes:
#   * `spec_accept_length` is a GAUGE reporting a recent mean, not a lifetime
#     one -- it moves as traffic changes, which is exactly why a single
#     end-of-run reading can mislead and a series is worth having.
#   * An idle rank keeps reporting its last value; it does NOT decay to zero.
#     Correlate with verify_calls to know whether a sample is fresh.
set -uo pipefail

OUT="${1:?usage: mtp_sampler.yihou.sh <out_csv> [interval_s]}"
INTERVAL="${2:-15}"
DECODE_HOST="${DECODE_HOST:-10.245.154.168}"
DECODE_PORT="${DECODE_PORT:-29002}"
LABEL="${LABEL:-}"

[[ -s "$OUT" ]] || echo "ts_utc,epoch,dp_rank,accept_length,accept_rate,verify_calls,running_req,label" > "$OUT"

while [[ ! -e "$OUT.stop" ]]; do
    body="$(curl -s -m 10 "http://$DECODE_HOST:$DECODE_PORT/metrics" 2>/dev/null)"
    if [[ -n "$body" ]]; then
        printf '%s' "$body" | LABEL="$LABEL" python3 -c '
import os, re, sys, time
ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
ep = int(time.time())
label = os.environ.get("LABEL", "")
acc, rate, running = {}, {}, {}
verify = ""
for line in sys.stdin:
    m = re.match(r"sglang:spec_accept_length\{([^}]*)\}\s+([\d.eE+-]+)", line)
    if m:
        r = re.search(r"dp_rank=\"(\d+)\"", m.group(1))
        if r: acc[r.group(1)] = m.group(2)
        continue
    m = re.match(r"sglang:spec_accept_rate\{([^}]*)\}\s+([\d.eE+-]+)", line)
    if m:
        r = re.search(r"dp_rank=\"(\d+)\"", m.group(1))
        if r: rate[r.group(1)] = m.group(2)
        continue
    m = re.match(r"sglang:spec_verify_calls_total\{[^}]*\}\s+([\d.eE+-]+)", line)
    if m:
        verify = m.group(1); continue
    m = re.match(r"sglang:num_running_reqs\{([^}]*)\}\s+([\d.eE+-]+)", line)
    if m:
        r = re.search(r"dp_rank=\"(\d+)\"", m.group(1))
        if r: running[r.group(1)] = m.group(2)
for k in sorted(acc, key=int):
    # Values pulled out first on purpose: nested quotes inside an f-string
    # expression do not survive being embedded in a single-quoted shell
    # string, and the resulting SyntaxError made the sampler emit a
    # header-only CSV while looking like it was running.
    rk = rate.get(k, "")
    nk = running.get(k, "")
    print(",".join([ts, str(ep), k, acc[k], rk, verify, nk, label]))
' >> "$OUT"
    fi
    sleep "$INTERVAL"
done
echo "sampler stopped (saw $OUT.stop)"
