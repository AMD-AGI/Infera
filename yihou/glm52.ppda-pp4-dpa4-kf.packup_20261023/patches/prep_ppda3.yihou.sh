#!/usr/bin/env bash
# prep_ppda3 (task gap 2026-10-23, ON 135): gap workflow wf_5e8048c1-c0d: ppda1 loses its 3.6 s TTFT in decode_bootstrap
# (prealloc queue on one full DP rank while others have room; router split requests exactly equally 2379 x 4); llying runs
# router R2 (INFERA_R2_DECODE_DEMAND=on, image router supports it). Adds ROUTER_EXTRA_ENV to run.v3x + DCPX_R2 knob,
# env.ppda3 = env.ppda1 + DCPX_R2=on, dryrun + tokdiff, queued after ppda2.
set -uo pipefail
W=/shared_nfs/yihou_glm52/g4tune_20261019; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; D=/shared_nfs/yihou_glm52/dcpx_20261022/ppdc
F=$H/run.v3x.yihou.sh
grep -q ROUTER_EXTRA_ENV $F || { cp -p $F $D/run.v3x.yihou.sh.before-r2; python3 -I - $F <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = 'router_cmd=(docker run -d --init --name "$CONTAINER_PREFIX-router" --network host\n  -e "INFERA_PD_DP_RANK_AFFINITY=$AFF"\n'
assert s.count(a) == 1
b = ('# ppda3 (10-23): ROUTER_EXTRA_ENV = space separated KEY=VALUE pairs for the router container (e.g. INFERA_R2_DECODE_DEMAND=on).\n'
     '_rx_env=(); for _kv in ${ROUTER_EXTRA_ENV:-}; do _rx_env+=(-e "$_kv"); done\n'
     + a + '  "${_rx_env[@]}"\n')
s = s.replace(a, b); open(p, "w").write(s)
PY
}
bash -n $F && echo RUNV3X_OK
C=$H/config.yihou.dcpx.sh
grep -q DCPX_R2 $C || { cp -p $C $D/config.yihou.dcpx.sh.before-r2; python3 -I - $C <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = '# --- KV rails:'
assert s.count(a) == 1
s = s.replace(a, '''# DCPX_R2=off|shadow|on (ppda3): router R2 - pick the decode DP rank by in-flight input/KV demand (llying kit --decode-input-demand on).
: "${DCPX_R2:=}"
if [[ -n "$DCPX_R2" ]]; then
    [[ "$DCPX_R2" =~ ^(off|shadow|on)$ ]] || { echo "config.yihou.dcpx.sh: DCPX_R2 must be off|shadow|on" >&2; return 2 2>/dev/null || exit 2; }
    export ROUTER_EXTRA_ENV="${ROUTER_EXTRA_ENV:+$ROUTER_EXTRA_ENV }INFERA_R2_DECODE_DEMAND=$DCPX_R2"
fi

''' + a)
s = s.replace('-kf${DCPX_KF:-0}', '-kf${DCPX_KF:-0}${DCPX_R2:+-r2$DCPX_R2}')
open(p, "w").write(s)
PY
}
bash -n $C && echo CONFIG_OK
E=$W/env.ppda3-135x136.yihou.sh
[[ -e $E ]] || { echo "# env.ppda3-135x136.yihou.sh (2026-10-23) = env.ppda1 + DCPX_R2=on ONLY (router R2 decode-rank choice by demand; gap analysis: ppda1 decode_bootstrap 3.6-7.4 s vs llying 1.0-1.7 s)."; grep -v "^# env.ppda1" $W/env.ppda1-135x136.yihou.sh | sed -e "s/YIHOU_ROUND=ppda1/YIHOU_ROUND=ppda3/" -e "s/ DCPX_DECODE_EC=0 / DCPX_DECODE_EC=0 DCPX_R2=on /"; } > $E
grep -c "DCPX_R2=on" $E
( source $E; cd $H; bash $H/run.v3x.yihou.sh dryrun ) > $W/dryrun.ppda3-135x136.yihou.txt 2>&1 || { echo DRYRUN_FAIL; tail -4 $W/dryrun.ppda3-135x136.yihou.txt; }
echo "== tokdiff ppda1 -> ppda3"; diff <(tr ' ' '\n' < $W/dryrun.ppda1-135x136.yihou.txt | sed 's/ppda1/ROUND/g') <(tr ' ' '\n' < $W/dryrun.ppda3-135x136.yihou.txt | sed 's/ppda3/ROUND/g') | grep '^[<>]' | grep -v '^[<>] +$' | tr '\n' ' '; echo
cat > $D/queue_ppda3.yihou.sh <<'EOS'
#!/usr/bin/env bash
# queue_ppda3 (task gap 2026-10-23, ON 135): after orch16 ppda2 is done, run orch16 ppda3 80 80 (A1 only).
D=/shared_nfs/yihou_glm52/dcpx_20261022/ppdc
for i in $(seq 1 600); do grep -q "orch16 done" $D/orch16.ppda2.yihou.log && break; sleep 30; done
grep -q "orch16 done" $D/orch16.ppda2.yihou.log || { echo "ppda2 orch not done after 5 h"; exit 1; }
bash $D/orch16.yihou.sh ppda3 80 80
EOS
cd /; setsid nohup bash $D/queue_ppda3.yihou.sh > $D/orch16.ppda3.yihou.log 2>&1 < /dev/null &
sleep 2; pgrep -af queue_ppda3 | head -1 | cut -c1-100
