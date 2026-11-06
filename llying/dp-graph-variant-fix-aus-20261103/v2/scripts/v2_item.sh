#!/usr/bin/env bash
# One holder work item on the Infera v2 kit (examples/sglang_pd_glm5.2_agentx of feat/sglang-pd-glm5.2-agentx, run from
# the worktree ~/code/bench_agentx/Infera-v2) with the v2 image (Infera main 44a294f8, 01-build-main-image).
#   gate  [SHAPE]        up.sh (image check) -> smoke -> GSM8K probe (PROBE_N) -> long context (LONGCTX_SPECS) -> eval.sh
#                        (EVAL=1) -> down.sh; real MTP acceptance from the decode log
#   sweep SHAPE CONC...  engine/agentx/sweep.sh (SIM_ACCEPT_LEN=2.99, a fresh deployment per point)
# SHAPE: 1node (4+4 on this node: prefill GPUs 0-3, decode 4-7), 1nodeswap (prefill 4-7, decode 0-3), 4p4 (prefill 4-7 here, decode 4-7 on the other node),
# 2p1d (prefills 0-3 / 4-7 here, decode 0-7 on the other node), 1p2d (prefill 0-3 here, decodes 0-3 / 4-7 on the other node). Summary lines: $LOG_ROOT/v2.out.
# Usage: v2_item.sh NAME gate|sweep SHAPE [CONC...]
set -uo pipefail
name="$1" kind="$2" shape="${3:-1node}"
shift 3 || shift $#
V=/perf_apps/liyingli/bench_agentx/v2-aus
O=/perf_apps/liyingli/bench_agentx/opt-aus-20261103
P=/perf_apps/liyingli/bench_agentx/vfix-20261103/scripts
export LOG_ROOT=/perf_apps/liyingli/vfix-logs/v2
mkdir -p "$LOG_ROOT"
out=$LOG_ROOT/v2.out
say() { echo "$(date -u +%T) v2 $name $*" >> "$out"; }
# A step that would not finish inside the allocation goes back to the queue (ITEM=<item script>, for a holder with more
# time left), as in opt_item5.sh. A sweep point takes ~80 min (deployment + warmup + 3600 s window + teardown).
left_min() {  # squeue %L: [D-]HH:MM:SS or MM:SS
    local l d=0 h=0 m=0
    l="$(squeue -h -j "${SLURM_JOB_ID:?}" -o %L 2>/dev/null)"
    [[ "$l" == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
    IFS=: read -ra t <<<"$l"
    if ((${#t[@]} == 3)); then h=${t[0]} m=${t[1]}; elif ((${#t[@]} == 2)); then m=${t[0]}; fi
    echo $((10#$d * 1440 + 10#$h * 60 + 10#$m))
}
case "$kind" in sweep) est=$((85 * $# + 5)) ;; *) est=45 ;; esac
est=${EST_MIN:-$est}  # a shorter window (DURATION) needs less
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
# dockerd (root) cannot read /home (root-squashed NFS), so the kit runs from a copy on /perf_apps, at the same path on
# every node.
SRC=/home/liyingli/code/bench_agentx/Infera-v2/examples/sglang_pd_glm5.2_agentx
export KIT=$V/kit
rsync -a --delete "$SRC/" "$KIT/" || { say "cannot copy the kit from $SRC"; exit 1; }
rev="$(git -C "$SRC" rev-parse --short HEAD)$(git -C "$SRC" diff --quiet HEAD -- . || echo -dirty)"
echo "$rev" > "$KIT/.infera-rev"
REV=44a294f8
export IMAGE=infera-sglang:v0522-main-$REV
TAR=/perf_apps/liyingli/images/infera-sglang-v0522-main-$REV.tar
export MODEL=/perf_apps/data/models/GLM-5.2-MXFP4 WORK_DIR=/scratch/$USER/v2 MOONCAKE_DISABLE_HIP_DMABUF=1
export RDMA_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 MC_GID_INDEX=1
mkdir -p "$WORK_DIR"
fenic='ip -4 -o addr show dev fenic | awk "{print \$4}" | cut -d/ -f1'
p="$(hostname -s)"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
case "$shape" in
1node) d="$p"; export PREFILL_WORKERS=1 PREFILL0_GPUS=0,1,2,3 DECODE_GPUS=4,5,6,7 ;;
1nodeswap) d="$p"; export PREFILL_WORKERS=1 PREFILL0_GPUS=4,5,6,7 DECODE_GPUS=0,1,2,3 ;;
4p4) d="${nodes[0]}"; [[ "${nodes[0]%%.*}" == "$p" ]] && d="${nodes[1]}"; export PREFILL_WORKERS=1 PREFILL0_GPUS=4,5,6,7 DECODE_GPUS=4,5,6,7 ;;
2p1d) d="${nodes[0]}"; [[ "${nodes[0]%%.*}" == "$p" ]] && d="${nodes[1]}"; export PREFILL_WORKERS=2 PREFILL0_GPUS=0,1,2,3 PREFILL1_GPUS=4,5,6,7 DECODE_GPUS=0,1,2,3,4,5,6,7 ;;
1p2d) d="${nodes[0]}"; [[ "${nodes[0]%%.*}" == "$p" ]] && d="${nodes[1]}"; export PREFILL_WORKERS=1 PREFILL0_GPUS=0,1,2,3 DECODE_WORKERS=2 DECODE_GPUS=0,1,2,3 DECODE1_GPUS=4,5,6,7 ;;
*) say "unknown shape $shape"; exit 1 ;;
esac
# pam_slurm_adopt refuses ssh once the job's extern step ends (AUS 11-04); srun into the job still works.
if ! timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$d" true 2>/dev/null; then
    export SHIM_JOB="$SLURM_JOB_ID" PATH="$O/pkg/tools/sshshim:$PATH"
    say "ssh to $d refused: decode commands go through tools/sshshim (srun --overlap)"
fi
export DECODE_SSH="$d" PREFILL_NODE_IP="$(bash -c "$fenic")"
if [[ "$d" == "$p" ]]; then export DECODE_NODE_IP="$PREFILL_NODE_IP"; else export DECODE_NODE_IP="$(ssh -o BatchMode=yes "$d" "$fenic")"; fi
[[ -n "$PREFILL_NODE_IP" && -n "$DECODE_NODE_IP" ]] || { say "no fenic address on $p / $d"; exit 1; }
targets=("$p"); [[ "$d" == "$p" ]] || targets+=("$d")
# The v2 image on every node of the shape (from the shared tar), and the InferenceX checkout (a GitHub clone is slow).
for n in "${targets[@]}"; do
    c="docker image inspect $IMAGE >/dev/null 2>&1 || docker load -i $TAR | tail -1"
    if [[ "$n" == "$p" ]]; then bash -c "$c"; else ssh -o BatchMode=yes "$n" "$c"; fi
done > "$LOG_ROOT/$name.image.log" 2>&1 || { say "image load failed (see $name.image.log)"; exit 1; }
IXC=/perf_apps/liyingli/vfix-logs/cache-InferenceX-918524ff
[[ -d "$WORK_DIR/cache/InferenceX/.git" || ! -d "$IXC/.git" ]] || { mkdir -p "$WORK_DIR/cache" && cp -a "$IXC" "$WORK_DIR/cache/InferenceX"; }
for n in "${targets[@]}"; do
    c="bash $O/scripts/clear_foreign_containers.sh"
    r="$(if [[ "$n" == "$p" ]]; then bash -c "$c"; else ssh -o BatchMode=yes "$n" "$c"; fi 2>&1)"
    [[ -z "$r" ]] || say "foreign containers: $r"
done
(cd "$KIT" && bash engine/down.sh) > "$LOG_ROOT/$name.predown.log" 2>&1
# Docker containers outlive the job: a guard step per node removes ours when Slurm signals the steps.
guards=()
for n in "${targets[@]}"; do
    srun --jobid="$SLURM_JOB_ID" --overlap -N1 -n1 -w "$n" --gres=gpu:8 --export=NONE \
        /bin/bash "$O/scripts/guard_containers.sh" glm52_agentx- >> "$LOG_ROOT/guard.log" 2>&1 &
    guards+=($!)
done
# VRAM_SAMPLE=<s>: every <s> seconds each node's per-GPU VRAM use goes to $LOG_ROOT/$name.vram-<node>.log (K5 C88
# crash, 11-06: K5 leaves 2.1 GB free per decode GPU after graph capture, K3 6.0 GB).
samplers=()
if [[ -n "${VRAM_SAMPLE:-}" ]]; then
    for n in "${targets[@]}"; do
        c="while sleep $VRAM_SAMPLE; do echo \"# \$(date -u +%T)\"; rocm-smi --showmeminfo vram --csv; done"
        if [[ "$n" == "$p" ]]; then
            bash -c "$c" >> "$LOG_ROOT/$name.vram-$n.log" 2>&1 &
        else
            ssh -o BatchMode=yes "$n" "$c" >> "$LOG_ROOT/$name.vram-$n.log" 2>&1 &
        fi
        samplers+=($!)
    done
fi
cleanup() {
    (cd "$KIT" && bash engine/down.sh) >/dev/null 2>&1
    kill "${guards[@]}" "${samplers[@]}" 2>/dev/null; wait "${guards[@]}" "${samplers[@]}" 2>/dev/null
}
# Preemption first sends SIGTERM to every step and kills the job after the grace time (~5 min on AUS: 00:02:26 ->
# 00:07:30, 11-05). Stop here instead of going on with a run whose engines the guards just removed: the item stays in
# running/, and the next allocation of the holder runs it again from the start.
preempted() { cleanup; trap - EXIT; say "SIGTERM (preemption notice): engines removed, waiting for the job to end"; exec sleep 100000; }
trap cleanup EXIT
trap preempted TERM INT
say "start: kit $rev, $kind $shape on $p ($PREFILL_NODE_IP) -> $d ($DECODE_NODE_IP), job $SLURM_JOB_ID, KF_KERNELS=${KF_KERNELS:-1} DRAFT_MOE_FP8=${DRAFT_MOE_FP8:-1} $*"
case "$kind" in
gate)
    label="$name-$(date -u +%m%dT%H%M)" log="$LOG_ROOT/$label.log"
    prompts="$LOG_ROOT/gsm8k_probe_prompts.jsonl"
    [[ -s "$prompts" ]] || gunzip -c "$P/../data/gsm8k_probe_prompts.jsonl.gz" > "$prompts"
    if ! (cd "$KIT" && RUN_ID="$label" bash engine/up.sh) > "$log" 2>&1; then
        say "$label up failed: $(tail -3 "$log" | tr '\n' ' ')"; exit 1
    fi
    run="$(cat "$WORK_DIR/current_run")" url="http://$PREFILL_NODE_IP:28300"
    say "$label env: $(ssh -o BatchMode=yes "$d" "docker exec glm52_agentx-decode env" | grep -E '^(SGLANG_USE_ROCM700A|SGLANG_GLM_NEXTN_MOE_PTPC|YIHOU_KF_(SPARSE|INDEXER|MOE)|SGLANG_USE_AITER)=' | sort | tr '\n' ' ')"
    (cd "$KIT" && bash engine/smoke.sh) >> "$log" 2>&1
    say "$label smoke: $(grep -a 'checks passed' "$log" | tail -1)"
    if [[ "${PROBE_N:-200}" != 0 ]]; then
        timeout 5400 python3 "$P/gsm8k_probe.py" "$prompts" "$url" glm5.2-mxfp4 "$LOG_ROOT/$label.probe.jsonl" \
            --limit "${PROBE_N:-200}" --concurrency 64 >> "$log" 2>&1
        say "$label probe: $(python3 "$P/probe_stats.py" "$LOG_ROOT/$label.probe.jsonl" --review "$LOG_ROOT/$label.review.txt")"
    fi
    k=0
    for spec in ${LONGCTX_SPECS-16000:64}; do
        k=$((k + 1))
        timeout 5400 python3 "$P/longctx_probe.py" "$prompts" "$url" glm5.2-mxfp4 "$LOG_ROOT/$label.longctx-$k-${spec%%:*}.jsonl" \
            --ctx-tokens "${spec%%:*}" --n "${spec##*:}" --concurrency 32 >> "$log" 2>&1
        say "$label longctx $spec: $(grep -aE '^(needle|gsm8k): n=' "$log" | tail -2 | tr '\n' ' ')"
    done
    if [[ "${EVAL:-0}" == 1 ]]; then
        (cd "$KIT" && bash engine/eval.sh) >> "$log" 2>&1
        res="$(find "$run/gsm8k" -name 'results*.json' 2>/dev/null | head -1)"
        say "$label lm-eval: $( [[ -n "$res" ]] && python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))['results']; t = next(iter(r.values()))
print('strict', t.get('exact_match,strict-match'), 'flexible', t.get('exact_match,flexible-extract'))" "$res" || echo missing)"
    fi
    (cd "$KIT" && bash engine/down.sh) >> "$log" 2>&1
    say "$label accept: $(grep -aoh 'accept len: [0-9.]*' "$run"/logs/*decode*.log 2>/dev/null |
        awk '{s += $3; n++} END {if (n) printf "mean %.2f over %d decode log lines", s / n, n; else print "no lines"}')"
    mkdir -p "$LOG_ROOT/runs" && cp -a "$run" "$LOG_ROOT/runs/" ;;
sweep)
    sid="$name-$(date -u +%m%dT%H%M)"
    (cd "$KIT" && SWEEP_ID="$sid" bash engine/agentx/sweep.sh "$@") > "$LOG_ROOT/$sid.sweep.log" 2>&1
    s="$WORK_DIR/sweeps/$sid"
    mkdir -p "$LOG_ROOT/sweeps" && cp -a "$s" "$LOG_ROOT/sweeps/"
    while read -r r; do mkdir -p "$LOG_ROOT/runs" && cp -a "$r" "$LOG_ROOT/runs/"; done < "$s/runs"
    say "sweep $sid: $(grep -E '^\| C' "$s/summary.md" | tr '\n' ' ')" ;;
*) say "unknown kind $kind"; exit 1 ;;
esac
say "end"
