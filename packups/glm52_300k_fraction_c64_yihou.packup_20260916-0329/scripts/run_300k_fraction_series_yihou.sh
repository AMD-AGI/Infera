#!/usr/bin/env bash
# 300K-fraction series: what does raising the share of 300K requests do to a C=64 batch?
#
# Six points at local_batch_size = 8 (C=64, dp=8), so the fraction lands on whole requests:
#   0 %  -> 0/8 at 300K      12.5 % -> 1/8      25 % -> 2/8
#   50 % -> 4/8              75 %   -> 6/8      100 % -> 8/8
#
# RUN ORDER IS DELIBERATELY NOT MONOTONE: 50, 0, 100, 25, 75, 12.5.
# Thermal/clock drift within a session is real -- the previous task measured 0.35-0.61 % within a
# single run -- and if the series were run 0->100 in order, a drift trend would be perfectly
# collinear with the ISL trend and no amount of analysis afterwards could separate them. Shuffled,
# a monotone result is evidence; in order, it would be an artifact we could not rule out.
#
# EVERY run captures rocm-smi --showpids before AND after. The earlier series in this workspace
# captured --showtemp --showpower --showclocks --showuse, which says nothing about whether anyone
# else was on the GPUs -- and a colleague's container did appear mid-session, leaving one
# measurement impossible to adjudicate. Tenancy has to be in the record, not in memory.
#
# Creates only. Deletes nothing.
set -euo pipefail

NODE=smci355-ccs-aus-n10-29
REPO=/home/yihou/dev/git.16-19/infera.dev.yihou.sglang.bench.fast.script
WS=$REPO/temp_workspace/isl_hetero_batch_yihou_20260915-1114
TOOL=$REPO/sglang_decode_internal_bench_and_profiling
CONT=yihou-glm52-tp8ep1-pr50-51
IMG=sha256:bdd783512f3db4d046fcf6e76e7039da054fe2634ad1d831a16a97da65dad656

L=300000
S=70000
# name:spec-args   (the 0 % and 100 % points are uniform, so they use --input-len)
SERIES=(
  "s1_pct50_yihou:--input-len-spec list:$L,$L,$L,$L,$S,$S,$S,$S"
  "s2_pct0_yihou:--input-len $S"
  "s3_pct100_yihou:--input-len $L"
  "s4_pct25_yihou:--input-len-spec list:$L,$L,$S,$S,$S,$S,$S,$S"
  "s5_pct75_yihou:--input-len-spec list:$L,$L,$L,$L,$L,$L,$S,$S"
  "s6_pct12_yihou:--input-len-spec list:$L,$S,$S,$S,$S,$S,$S,$S"
)

# MUST STAY ON ONE LINE. This string is interpolated into a double-quoted ssh command, so it is
# NOT word-split locally -- it is handed to the remote shell verbatim, and a newline inside it is a
# COMMAND SEPARATOR there, not whitespace. Written across three lines on the first attempt, lines 2
# and 3 were executed as separate remote commands: --warmup-steps 10, --mem-fraction-static 0.85,
# --enable-aiter-allreduce-fusion and --enable-fused-qk-norm-rope never reached the benchmark, which
# silently ran at warmup=3 with the fusion optimisations OFF. It produced complete, gate-passing,
# internally consistent results that were not comparable to anything else we have measured.
COMMON="--tp-size 8 --ep-size 1 --enable-dp-attention --batch-size 64 --max-running-requests 64 --output-len 10000 --accept-length 3.61 --warmup-steps 10 --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope --mem-fraction-static 0.85"

# Writes to logs/, never into the iteration directory: run_decode.sh refuses to start if that
# directory already exists, so pre-creating it to hold a capture kills the run before it begins.
# That cost a run at 07:24 on 2026-09-15 in the previous task.
smi () {  # $1 = run name, $2 = before|after
    ssh -o BatchMode=yes "$NODE" \
      'date -u "+%Y-%m-%dT%H:%M:%SZ"; rocm-smi --showpids 2>&1; echo "--- temp/power ---"; rocm-smi --showtemp --showpower 2>&1' \
      > "$WS/logs/smi_$1_$2_yihou.txt" 2>&1 || true
}

printf '===== 300K-fraction series, order: 50 0 100 25 75 12.5 =====\n'
for i in "${!SERIES[@]}"; do
    entry=${SERIES[$i]}
    NAME=${entry%%:*}
    SPEC=${entry#*:}

    printf '\n===== [%d/6] %s  %s =====\n' "$((i+1))" "$NAME" "$SPEC"
    date -u '+start %Y-%m-%dT%H:%M:%SZ'
    smi "$NAME" before

    ssh -o BatchMode=yes "$NODE" \
      "cd '$WS' && CONTAINER=$CONT IMAGE=$IMG bash $TOOL/scripts/run_decode.sh $NAME $SPEC $COMMON" \
      || printf '!! run %s exited non-zero\n' "$NAME"

    smi "$NAME" after
    date -u '+done  %Y-%m-%dT%H:%M:%SZ'

    # Assert the arguments actually LANDED, do not assume they did. The first attempt at this
    # series lost four flags to remote-shell line splitting and still produced complete,
    # gate-passing results; nothing in the output said anything was wrong.
    python3 -c "
import json, sys
try:
    c = json.load(open('$WS/iterations/$NAME/config_yihou.json'))
    b, cli = c['benchmark'], ' '.join(c['server_cli'])
    bad = []
    if b.get('warmup_steps') != 10: bad.append('warmup_steps=%r' % b.get('warmup_steps'))
    if b.get('batch_size') != 64:   bad.append('batch_size=%r' % b.get('batch_size'))
    for flag in ('--enable-aiter-allreduce-fusion', '--enable-fused-qk-norm-rope'):
        if flag not in cli: bad.append('missing ' + flag)
    if '--mem-fraction-static 0.85' not in cli: bad.append('mem-fraction-static not 0.85')
    print('  ARG CHECK: ' + ('FAILED -> ' + '; '.join(bad) if bad else 'ok'))
    d = json.load(open('$WS/iterations/$NAME/result_yihou.json'))
    print('  TPOT=%.4f  accept=%s  iters=%s  complete=%s' % (
        d['effective_token_latency_ms_per_user'], d['realized_accept_length'],
        d['verify_iterations'], d['complete']))
except Exception as e:
    print('  no result:', e)
"
    if (( i < ${#SERIES[@]} - 1 )); then echo "  --- 60 s gap ---"; sleep 60; fi
done

printf '\n===== series complete =====\n'
