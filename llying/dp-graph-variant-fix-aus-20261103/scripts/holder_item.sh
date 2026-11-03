#!/usr/bin/env bash
# Holder work item (/apps/liyingli/holder/queue): the verification of ../../vfix-20261103 (fde1fb75's
# llying/dp-graph-variant-fix-20261103, unmodified apart from config/site.local.env) inside the holder's allocation,
# through the package's own scripts/slurm_verify.sbatch. Only the deployments not yet completed in verify.out run, in the
# REPRODUCE order, so a requeued or new holder (which runs this item from the start) resumes where it stopped.
# A deployment counts as completed once its "<label> down" line is in verify.out; an interrupted one is run again.
# preflight always runs (it records which image this allocation uses); before it, scripts/build_image.sh on both nodes
# builds the image where it is missing (a no-op plus the 11-file check where it is present).
# Node IPs: the fenic address on each node (getent there gives 127.0.1.1 for the node itself and another subnet for
# the .cs-aus.dcgpu names). VFIX_STEPS overrides the step list (same names as run_verify.sh).
set -uo pipefail
PKG=/perf_apps/liyingli/bench_agentx/vfix-20261103
set -a; . "$PKG/config/site.local.env"; set +a
V=$LOG_ROOT/verify.out
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
p="$(hostname -s)"
d="${nodes[0]}"; [[ "${nodes[0]%%.*}" == "$p" ]] && d="${nodes[1]}"
fenic='ip -4 -o addr show dev fenic | awk "{print \$4}" | cut -d/ -f1'
P_IP="$(bash -c "$fenic")" D_IP="$(ssh -o BatchMode=yes "$d" "$fenic")"
[[ -n "$P_IP" && -n "$D_IP" ]] || { echo "no fenic address: prefill $p '$P_IP', decode $d '$D_IP'"; exit 1; }
ALL="${VFIX_STEPS:-preflight unit B1 N1a A1 B2 A2 B3 C0 C1 C2 N0 N1 Bprod Cprod scoreB scoreC}"
declare -A arm=([A1]=A-gate [A2]=A-gate [B1]=B-gate [B2]=B-gate [B3]=B-gate [C1]=C-gate [C2]=C-gate [C0]=C0-e11
    [N0]=N0-nospec [N1]=N1-nospec-fix [N1a]=N1a-nospec-audit [Bprod]=B-prod [Cprod]=C-prod)
declare -A nth
todo=()
for s in $ALL; do
    case "$s" in
    preflight) ;;
    unit) grep -qaE '^[0-9:]{8} unit: ([0-9]+)/\1 passed$' "$V" 2>/dev/null && continue ;;
    scoreB | scoreC) grep -qaE "^[0-9:]{8} score ${s#score}: " "$V" 2>/dev/null && continue ;;
    *)
        a="${arm[$s]:?unknown step $s}"
        nth[$a]=$((${nth[$a]:-0} + 1))
        done_n="$(grep -caE "^[0-9:]{8} arm-$a-[0-9]{4}T[0-9]{4} down$" "$V" 2>/dev/null)"
        ((${done_n:-0} >= nth[$a])) && continue ;;
    esac
    todo+=("$s")
done
echo "$(date -u +%FT%TZ) holder item: job ${SLURM_JOB_ID:-?}: prefill $p ($P_IP), decode $d ($D_IP); remaining steps: ${todo[*]}"
[[ "${todo[*]}" != preflight ]] || exit 0
b="DOCKER_CONFIG=/tmp/liyingli-docker-anon INFERA_REPO=$INFERA_REPO CTX=/scratch/$USER/ctx-fdbafb60 bash $PKG/scripts/build_image.sh"
mkdir -p /tmp/liyingli-docker-anon && [[ -s /tmp/liyingli-docker-anon/config.json ]] || echo '{}' > /tmp/liyingli-docker-anon/config.json
ssh -o BatchMode=yes "$d" "mkdir -p /tmp/liyingli-docker-anon && { [[ -s /tmp/liyingli-docker-anon/config.json ]] || echo '{}' > /tmp/liyingli-docker-anon/config.json; }; $b" \
    > "$LOG_ROOT/build-$d.log" 2>&1 &
bash -c "$b" > "$LOG_ROOT/build-$p.log" 2>&1; rp=$?
wait $!; rd=$?
echo "$(date -u +%FT%TZ) holder item: image $p: $(tail -1 "$LOG_ROOT/build-$p.log"); $d: $(tail -1 "$LOG_ROOT/build-$d.log")"
((rp == 0 && rd == 0)) || { echo "image step failed (rc $rp / $rd)"; exit 1; }
export PREFILL_NODE_IP_OVERRIDE="$P_IP" DECODE_NODE_IP_OVERRIDE="$D_IP" PREFILL_NODE_IP="$P_IP" DECODE_NODE_IP="$D_IP" DECODE_SSH="$d"
exec bash "$PKG/scripts/slurm_verify.sbatch" "${todo[@]}"
