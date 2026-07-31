#!/usr/bin/env bash
# The three acceptance tests, run inside the prefill container against the router.
#   T1  correctness  — temp=0 factual probe (4 cases)
#   T2  conc=32      — 128 requests, every output captured and classified
#   T3  conc=128     — 512 requests, same
#
# Pass criteria (per the operator's call on 2026-07-31):
#   T1  >= 3/4 correct
#   T2/T3  throughput test — no HTTP errors, no hangs, no KV corruption.
#     `finish=length` from the stop/EOS path is NOT counted as a failure here:
#     these prompts run without a chat template and with EOS suppressed for
#     throughput, so a run-on completion is expected and is not a defect of
#     kvaware/kvd. CORRUPT_REASONING is the verdict that would indicate real KV
#     damage, and it must be 0.
set -u
BASE="${BASE:?BASE=http://<router-ip>:8100}"
MODEL="${MODEL:-glm5.2-mxfp4}"
OUT="${OUT:-/tmp/final_results}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
mkdir -p "$OUT"

kvdstats() {  # $1 = label
  python3 -c "
import asyncio, json, sys
from infera.kvd.client import KvdClient
async def m():
    c = KvdClient('$KVD_SOCK', client_id='stats-$1')
    await c.connect(); s = await c.stats(); await c.close()
    print(json.dumps(s if isinstance(s, dict) else str(s)))
asyncio.run(m())
" 2>/dev/null || echo '{"error":"kvd unreachable"}'
}

echo "===== kvd counters BEFORE ====="
kvdstats before | tee "$OUT/kvd_before.json"

echo
echo "===== T1: correctness (temp=0 factual probe) ====="
python3 /tmp/probe.py "$BASE" "$MODEL" 2>&1 | tee "$OUT/t1_correctness.txt"
T1=${PIPESTATUS[0]}
echo "T1 rc=$T1" | tee -a "$OUT/t1_correctness.txt"

# stress_capture.py always exits 0 — it is a capture tool, not a judge. The
# verdict lives in the JSON it writes, so grade that explicitly.
grade() {  # $1 = json path, $2 = label
  python3 -c "
import json, sys, collections
d = json.load(open('$1'))
rows = d['rows']
c = collections.Counter(r['verdict'] for r in rows)
err  = c.get('ERROR', 0)                      # HTTP/transport failure  -> hard fail
corr = c.get('CORRUPT_REASONING', 0)          # KV damage               -> hard fail
wrong= c.get('WRONG', 0)                      # right shape, wrong needle
loop = c.get('DIGIT_LOOP', 0) + c.get('TAIL_REPEAT', 0)   # EOS suppressed: expected
print('$2', 'n=%d dur=%.1fs' % (len(rows), d['duration_s']))
for k, v in sorted(c.items()):
    print('   %-18s %d' % (k, v))
tp = len(rows) / d['duration_s']
print('   throughput        %.1f req/s' % tp)
print('   -- gate: ERROR=%d (want 0)  CORRUPT_REASONING=%d (want 0)  WRONG=%d' % (err, corr, wrong))
print('   -- informational: stop/EOS run-on (DIGIT_LOOP+TAIL_REPEAT)=%d' % loop)
sys.exit(0 if (err == 0 and corr == 0) else 1)
"
}

echo
echo "===== T2: conc=32, 128 requests, ISL/OSL 1024 ====="
python3 /tmp/stress_capture.py "$BASE" "$MODEL" 32 128 1024 1024 \
  "$OUT/t2_conc32.json" 20260731 2>&1 | tee "$OUT/t2_conc32.txt"
grade "$OUT/t2_conc32.json" "T2/conc=32" 2>&1 | tee "$OUT/t2_grade.txt"
T2=${PIPESTATUS[0]}

echo
echo "===== kvd counters AFTER conc=32 ====="
kvdstats mid | tee "$OUT/kvd_mid.json"

echo
echo "===== T3: conc=128, 512 requests, ISL/OSL 1024 ====="
python3 /tmp/stress_capture.py "$BASE" "$MODEL" 128 512 1024 1024 \
  "$OUT/t3_conc128.json" 20260731 2>&1 | tee "$OUT/t3_conc128.txt"
grade "$OUT/t3_conc128.json" "T3/conc=128" 2>&1 | tee "$OUT/t3_grade.txt"
T3=${PIPESTATUS[0]}

echo
echo "===== kvd counters AFTER ====="
kvdstats after | tee "$OUT/kvd_after.json"

echo
echo "===== SUMMARY ====="
echo "T1 correctness rc=$T1  T2 conc=32 rc=$T2  T3 conc=128 rc=$T3"
echo "(rc=0 is pass for each)"
echo "artifacts in $OUT"
