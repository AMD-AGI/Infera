#!/usr/bin/env bash
# One optimisation step as a holder work item, on the holder's two nodes (prefill = the batch host, decode = the other),
# with the package copy ../pkg (= the verification package + router hooks in kit/engine, arms in pkg/config).
#   score ARM CONC   pd_perf_chain.sh (fresh deployment, AgentX CONC lanes, 3600 s) + score_summary.sh
#   gate  ARM        verify_suite.sh with PROBE_N / LONGCTX_SPECS / EVAL from the environment (real acceptance)
#   trace ARM CONC   decode_profile_chain.sh (AgentX CONC lanes 1500 s, two 40-step decode profiles from 600 s after
#                    warmup, SIM 2.99) + ana/dgaps.py per trace (GPU idle per decode step by position)
# A step that would not finish inside the allocation (EST_MIN against squeue's time left) is not started: the item
# sleeps until the job ends, so the next holder requeues it from running/ and runs it from the start.
# Summary lines: $LOG_ROOT/opt.out (LOG_ROOT=/perf_apps/liyingli/vfix-logs/opt).
# opt_item5.sh: opt_item4.sh, deferring politely: with ITEM=<item script> set, an item that does not fit goes back to the
# queue (for a holder with more time left) instead of holding running/ until the job ends.
# opt_item4.sh: opt_item3.sh on a one-node allocation too (prefill GPUs 0-3 and decode GPUs 4-7 on the batch host).
# opt_item3.sh: opt_item2.sh + sshshim fallback, a down.sh of leftovers first, a container guard step per node.
# opt_item2.sh: + real MTP acceptance after a gate. Usage: opt_item2.sh NAME EST_MIN score|gate ARM [CONC]
set -uo pipefail
name="$1" est="$2" kind="$3" arm="$4" conc="${5:-104}"
O=/perf_apps/liyingli/bench_agentx/opt-aus-20261103
export PKG=$O/pkg
set -a; . "$PKG/config/site.local.env"; set +a
export LOG_ROOT=/perf_apps/liyingli/vfix-logs/opt
mkdir -p "$LOG_ROOT"
out=$LOG_ROOT/opt.out
say() { echo "$(date -u +%T) opt $name $*" >> "$out"; }
left_min() {  # squeue %L: [D-]HH:MM:SS or MM:SS
    local l d=0 h=0 m=0
    l="$(squeue -h -j "${SLURM_JOB_ID:?}" -o %L 2>/dev/null)"
    [[ "$l" == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
    IFS=: read -ra p <<<"$l"
    if ((${#p[@]} == 3)); then h=${p[0]} m=${p[1]}; elif ((${#p[@]} == 2)); then m=${p[0]}; fi
    echo $((10#$d * 1440 + 10#$h * 60 + 10#$m))
}
left="$(left_min)"
if ((left < est + 5)); then
    if [[ -n "${ITEM:-}" && -f "$ITEM" ]]; then
        cp "$ITEM" "$(dirname "$(dirname "$ITEM")")/queue/"
        say "deferred: needs ~$est min, job $SLURM_JOB_ID has $left min left; back to the queue"
        sleep 600; exit 0
    fi
    say "deferred: needs ~$est min, job $SLURM_JOB_ID has $left min left; waiting for the next allocation"
    exec sleep 100000
fi
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
p="$(hostname -s)"
if ((${#nodes[@]} == 1)); then
    d="$p"; export PREFILL0_GPUS=0,1,2,3 DECODE_GPUS=4,5,6,7
else
    d="${nodes[0]}"; [[ "${nodes[0]%%.*}" == "$p" ]] && d="${nodes[1]}"
fi
targets=("$p"); [[ "$d" == "$p" ]] || targets+=("$d")
# Once the job's extern step ends, pam_slurm_adopt refuses ssh to the job's nodes (AUS, 11-04 11:42); srun into the job
# still works, so decode-side commands then go through tools/sshshim (srun --overlap).
if ! timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$d" true 2>/dev/null; then
    export SHIM_JOB="$SLURM_JOB_ID" PATH="$PKG/tools/sshshim:$PATH"
    say "ssh to $d refused: decode commands go through tools/sshshim (srun --overlap)"
fi
# Leftovers of an item that could not reach the decode (e.g. ssh refused mid-run): save their logs into that run's
# logs/ (current_run still points at it) and remove the containers before this item starts.
(set -a; . "$PKG/config/recipe-e10.env"; set +a; export DECODE_SSH="$d"; cd "$PKG/kit" && bash engine/down.sh) > "$LOG_ROOT/$name.predown.log" 2>&1
fenic='ip -4 -o addr show dev fenic | awk "{print \$4}" | cut -d/ -f1'
export PREFILL_NODE_IP="$(bash -c "$fenic")" DECODE_SSH="$d"
if [[ "$d" == "$p" ]]; then export DECODE_NODE_IP="$PREFILL_NODE_IP"; else export DECODE_NODE_IP="$(ssh -o BatchMode=yes "$d" "$fenic")"; fi
[[ -n "$PREFILL_NODE_IP" && -n "$DECODE_NODE_IP" ]] || { say "no fenic address on $p / $d"; exit 1; }
# Image (loaded from TAR, else built, where missing; 11-file check everywhere), InferenceX checkout (a fresh GitHub
# clone takes ~15 min on AUS). A fresh node lacks the base image layers, and an anonymous pull of them is slow.
TAR=/perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar
b="{ docker image inspect $IMAGE >/dev/null 2>&1 || { [[ -s $TAR ]] && docker load -i $TAR | tail -1; }; };"
b="$b DOCKER_CONFIG=/tmp/liyingli-docker-anon INFERA_REPO=$INFERA_REPO CTX=/scratch/$USER/ctx-fdbafb60 bash $PKG/scripts/build_image.sh"
anon='mkdir -p /tmp/liyingli-docker-anon && { [[ -s /tmp/liyingli-docker-anon/config.json ]] || echo "{}" > /tmp/liyingli-docker-anon/config.json; }'
rd=0
if [[ "$d" != "$p" ]]; then ssh -o BatchMode=yes "$d" "$anon; $b" > "$LOG_ROOT/build-$d.log" 2>&1 & fi
bash -c "$anon; $b" > "$LOG_ROOT/build-$p.log" 2>&1; rp=$?
if [[ "$d" != "$p" ]]; then wait $!; rd=$?; fi
((rp == 0 && rd == 0)) || { say "image step failed on $p ($rp) / $d ($rd)"; exit 1; }
IXC=/perf_apps/liyingli/vfix-logs/cache-InferenceX-918524ff
if [[ ! -d "$WORK_DIR/cache/InferenceX/.git" && -d "$IXC/.git" ]]; then
    mkdir -p "$WORK_DIR/cache" && cp -a "$IXC" "$WORK_DIR/cache/InferenceX"
fi
# Docker containers outlive the job: a guard step per node removes ours when Slurm signals the steps (preemption, end of
# the allocation). Stopped after down.sh at the end of the item, so down.sh still saves the container logs first.
guards=()
for n in "${targets[@]}"; do
    srun --jobid="$SLURM_JOB_ID" --overlap -N1 -n1 -w "$n" --gres=gpu:8 --export=NONE \
        /bin/bash "$O/scripts/guard_containers.sh" glm52_2p1d- >> "$LOG_ROOT/guard.log" 2>&1 &
    guards+=($!)
done
cleanup() {
    (set -a; . "$PKG/config/recipe-e10.env"; set +a; cd "$PKG/kit" && bash engine/down.sh) >/dev/null 2>&1
    kill "${guards[@]}" 2>/dev/null; wait "${guards[@]}" 2>/dev/null
}
# Preemption first sends SIGTERM to every step and kills the job after the grace time (~5 min on AUS: 00:02:26 ->
# 00:07:30, 11-05). Stop here instead of going on with a run whose engines the guards just removed: the item stays in
# running/, and the next allocation of the holder runs it again from the start.
preempted() { cleanup; trap - EXIT; say "SIGTERM (preemption notice): engines removed, waiting for the job to end"; exec sleep 100000; }
trap cleanup EXIT
trap preempted TERM INT
say "start: $kind $(basename "$arm") conc=$conc on $p ($PREFILL_NODE_IP) -> $d ($DECODE_NODE_IP), job $SLURM_JOB_ID, $left min left"
case "$kind" in
score)
    lab="opt-$name-c$conc"
    bash "$PKG/scripts/pd_perf_chain.sh" "$PKG/kit" "$PKG/config/$arm" "$LOG_ROOT/perf/$lab" "$conc" >> "$LOG_ROOT/$name.chain.log" 2>&1
    bash "$PKG/scripts/score_summary.sh" "$LOG_ROOT/perf/$lab/c$conc.log" > "$LOG_ROOT/score-$name.txt" 2>&1
    say "score: $(grep -E '^\| C[0-9]+' "$LOG_ROOT/score-$name.txt" || sed -n 3p "$LOG_ROOT/score-$name.txt")" ;;
trace)
    lab="opt-$name-c$conc"
    SIM=2.99 bash "$O/scripts/decode_profile_chain.sh" "$PKG/kit" "$PKG/config/$arm" "$LOG_ROOT/trace/$lab" "$conc" 1500 600 40 300 \
        >> "$LOG_ROOT/$name.chain.log" 2>&1
    while read -r t; do
        say "dgaps ${t#"$LOG_ROOT/trace/$lab/traces/"}: $(python3 "$O/scripts/ana/dgaps.py" "$t" 2>&1 | head -1)"
    done < <(find "$LOG_ROOT/trace/$lab/traces" -name '*.json.gz' | sort) ;;
gate)
    bash "$PKG/scripts/verify_suite.sh" "$PKG/config/$arm" >> "$out" 2>&1 || say "verify_suite failed"
    # Real MTP acceptance over the deployment (decode log lines, every DP rank; smoke + probe load).
    r="$(ls -td "$WORK_DIR"/runs/"$(basename "$arm" .env)"-* 2>/dev/null | head -1)"
    say "accept $(basename "${r:-none}"): $(grep -aoh 'accept len: [0-9.]*' "$r"/logs/*decode*.log 2>/dev/null |
        awk '{s += $3; n++} END {if (n) printf "mean %.2f over %d decode log lines", s / n, n; else print "no lines"}')" ;;
*) say "unknown kind $kind"; exit 1 ;;
esac
say "end"
