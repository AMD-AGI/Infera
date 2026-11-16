#!/usr/bin/env bash
# One step of the reproduction, run by the hold job's queue runner on NODE0 (hold2.sbatch / q.sh).
# Crusoe copy of llying/dp-graph-variant-fix-aus-20261103/opt/scripts/opt_item5.sh with the AUS-only parts dropped
# (fenic IPs, driver-pilot exclusion, holder requeue): prefill = NODE0, decode = NODE1 (MODE=1node: both on NODE0,
# prefill GPUs 0-3, decode GPUs 4-7).
#   score ARM CONC   pd_perf_chain.sh (fresh deployment, AgentX CONC lanes, DURATION s) + score_summary.sh
#   gate  ARM        verify_suite.sh with PROBE_N / LONGCTX_SPECS / EVAL from the environment (real acceptance)
# Usage: [MODE=1node] [EXTRA_ENV=file] run_item.sh NAME score|gate ARM [CONC]
# EXTRA_ENV: sourced after the arm (site-specific additions such as HSA_ENABLE_SDMA=0), see ../config-crusoe/.
set -uo pipefail
name="$1" kind="$2" arm="$3" conc="${4:-104}"
R=/home/liyingli/bench_agentx/pd-repro-20261116
export PKG=$R/pkg
export SHIM_JOB="${JOB:?run inside the hold job}" PATH="$PKG/tools/sshshim:$PATH"
set -a; . "$PKG/config/site.local.env"; set +a
out=$LOG_ROOT/opt.out
mkdir -p "$LOG_ROOT"
say() { echo "$(date -u +%T) $name $*" | tee -a "$out"; }
p="$NODE0"
if [[ "${MODE:-2node}" == 1node ]]; then
    d="$p"; export PREFILL0_GPUS=0,1,2,3 DECODE_GPUS=4,5,6,7
else
    d="$NODE1"
fi
targets=("$p"); [[ "$d" == "$p" ]] || targets+=("$d")
on() { if [[ "$1" == "$p" ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
src_ip='ip -o -4 route get 1.1.1.1 | awk "{for (i = 1; i < NF; i++) if (\$i == \"src\") {print \$(i + 1); exit}}"'
export PREFILL_NODE_IP="$(bash -c "$src_ip" | tr -d '[:space:]')" DECODE_SSH="$d"
if [[ "$d" == "$p" ]]; then export DECODE_NODE_IP="$PREFILL_NODE_IP"; else export DECODE_NODE_IP="$(on "$d" "$src_ip" | tail -1 | tr -d '[:space:]')"; fi
[[ "$PREFILL_NODE_IP" =~ ^[0-9.]+$ && "$DECODE_NODE_IP" =~ ^[0-9.]+$ ]] || { say "no node IPs ($PREFILL_NODE_IP / $DECODE_NODE_IP)"; exit 1; }
# The arm, then site additions, are what the kit scripts source; write the combined file so pd_perf_chain.sh /
# verify_suite.sh (which source exactly one file) see both.
armf="$PKG/config/$arm"
if [[ -n "${EXTRA_ENV:-}" ]]; then
    c="$LOG_ROOT/arms/${arm%.env}+$(basename "$EXTRA_ENV" .env).env"
    mkdir -p "$(dirname "$c")"
    printf '. "%s"\n. "%s"\n' "$armf" "$EXTRA_ENV" > "$c"
    armf="$c"
fi
# A node whose containerd hangs answers the docker socket's _ping but times out on images/load (v3-058, 11-16):
# refuse it up front instead of waiting on a docker load that never ends.
for n in "${targets[@]}"; do
    hc="$(on "$n" "curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json" 2>/dev/null | tail -1)"
    [[ "$hc" == 200 ]] || { say "docker on $n does not answer images/json (http $hc): containerd hung? stopping"; exit 1; }
done
# Leftovers of an earlier item: save their logs (current_run) and remove the containers.
(set -a; . "$PKG/config/recipe-e10.env"; set +a; export DECODE_SSH="$d"; cd "$PKG/kit" && bash engine/down.sh) > "$LOG_ROOT/$name.predown.log" 2>&1
# Image on every node (load from the tar where missing) and the 11 base files.
b="{ docker image inspect $IMAGE >/dev/null 2>&1 || timeout 1200 docker load -i $IMAGE_TAR | tail -1; }; INFERA_REPO=$INFERA_REPO bash $PKG/scripts/build_image.sh"
pids=()
for n in "${targets[@]}"; do on "$n" "$b" > "$LOG_ROOT/build-$n.log" 2>&1 & pids+=($!); done
rc=0; for x in "${pids[@]}"; do wait "$x" || rc=1; done
((rc == 0)) || { say "image step failed: $(tail -2 "$LOG_ROOT"/build-*.log | tr '\n' ' ')"; exit 1; }
# The #45800 backport (arm-C-prod-ptpcfix mounts $PKG/../ptpc-fix/glm4_moe.py): the image's file + the PR diff.
PF=$R/ptpc-fix/glm4_moe.py
if [[ ! -s "$PF" ]]; then
    mkdir -p "$R/ptpc-fix"
    docker run --rm --network none --entrypoint cat "$IMAGE" /sgl-workspace/sglang/python/sglang/srt/models/glm4_moe.py > "$PF.orig"
    cp "$PF.orig" "$PF.tmp" && patch "$PF.tmp" < "$R/opt/glm4_moe.pr45800.diff" > "$R/ptpc-fix/patch.log" 2>&1 && mv "$PF.tmp" "$PF" ||
        { say "ptpc-fix: patch failed ($(tail -2 "$R/ptpc-fix/patch.log" | tr '\n' ' '))"; exit 1; }
    say "ptpc-fix: $(md5sum < "$PF" | cut -c1-12) ($(grep -c . "$R/ptpc-fix/patch.log") patch lines: $(tr '\n' ' ' < "$R/ptpc-fix/patch.log"))"
fi
# Wait until every GPU of the nodes is below 2 % VRAM (a removed engine's VRAM can take minutes to come back), max 30 min.
vram_busy() {
    local c="rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {if (\$NF + 0 > 2) b = 1} END {print b + 0}'" n busy=""
    for n in "${targets[@]}"; do [[ "$(on "$n" "$c" 2>/dev/null | tail -1)" == 1 ]] && busy="$busy $n"; done
    echo "$busy"
}
t0=$(date +%s)
while busy="$(vram_busy)"; [[ -n "$busy" ]]; do
    if (( $(date +%s) - t0 > 1800 )); then say "VRAM still in use on$busy after 30 min; starting anyway"; break; fi
    say "waiting for VRAM to be freed on$busy"; sleep 60
done
say "start: $kind $(basename "$armf") conc=$conc on $p ($PREFILL_NODE_IP) -> $d ($DECODE_NODE_IP), job $JOB"
case "$kind" in
score)
    lab="opt-$name-c$conc"
    bash "$PKG/scripts/pd_perf_chain.sh" "$PKG/kit" "$armf" "$LOG_ROOT/perf/$lab" "$conc" >> "$LOG_ROOT/$name.chain.log" 2>&1
    bash "$PKG/scripts/score_summary.sh" "$LOG_ROOT/perf/$lab/c$conc.log" > "$LOG_ROOT/score-$name.txt" 2>&1
    say "score: $(grep -E '^\| C[0-9]+' "$LOG_ROOT/score-$name.txt" || sed -n 3p "$LOG_ROOT/score-$name.txt")" ;;
gate)
    bash "$PKG/scripts/verify_suite.sh" "$armf" >> "$out" 2>&1 || say "verify_suite failed"
    r="$(ls -td "$WORK_DIR"/runs/"$(basename "$armf" .env)"-* 2>/dev/null | head -1)"
    say "accept $(basename "${r:-none}"): $(grep -aoh 'accept len: [0-9.]*' "$r"/logs/*decode*.log 2>/dev/null |
        awk '{s += $3; n++} END {if (n) printf "mean %.2f over %d decode log lines", s / n, n; else print "no lines"}')" ;;
*) say "unknown kind $kind"; exit 1 ;;
esac
say "end"
