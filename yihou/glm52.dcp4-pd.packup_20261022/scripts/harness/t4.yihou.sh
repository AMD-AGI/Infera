#!/usr/bin/env bash
# T4 launcher (pdrun): run design/pd_dcp_dsa.md section 7 E2E checks against a LIVE dcp_pd stack.
# Login host:  source $H/env.<round>-<nk>.yihou.sh; PD_JOB=<job> bash t4.yihou.sh STEP [STEP...]
# STEPS: a = math + passkey 8K/34K/96K (mixrun/check.py copy; same prompts/seed as the mixed runs)
#        b = GSM8K sgl-eval 0.1.3, 200 ex, GLM-5.2 preset, enable_thinking=true (same cmd as mixrun/run_gsm8k.sh)
#        c = accept length, real draft (refuses if the decode container has SGLANG_SIMULATE_ACC_LEN)
#        d = teacher-forced logprob agreement: NOT implemented here (needs the mixed DCP4 outputs) -> runbook.
# Clients run in detached --rm containers named yihou-dcp-t4<step>-<round> on the node (CPU only), writing to
# $WS/runs/<round>/t4/. Once analyst writes design/t4_runbook.md, steps/args follow it (diff this header vs runbook).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${CONFIG:?source an env.<round>-<nk>.yihou.sh first}"
read -r -a OVR <<< "${YIHOU_OVERRIDES:-}"; for kv in "${OVR[@]}"; do export "$kv"; done
set -a; source "$CONFIG" || exit 2; set +a
IP=$YIHOU_IP; RD="$WS/runs/$YIHOU_ROUND/t4"; mkdir -p "$RD"
R="http://$IP:$ROUTER_PORT"; D="http://$IP:$((ENGINE_PORT_BASE + ENGINE_PORT_STRIDE))"
T4=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/tools/t4
V=/shared_nfs/yihou_glm52/dcp_pd_20261021/mixrun/venv-sgleval     # read-only use of mixrun's sgl-eval venv
CIMG=lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
q(){ local o="" a; for a in "$@"; do printf -v a '%q' "$a"; o+="${o:+ }$a"; done; echo "$o"; }
node(){ "${SSH[@]}" -n "$YIHOU_NODE" "$(q "$@")"; }
client(){ local name=$1; shift
  node docker run -d --rm --name "yihou-dcp-t4$name-$YIHOU_ROUND" --network host -v "$WS:$WS" -w /tmp \
    -e "T4_MODEL=$SERVED_MODEL" -e "HOME=$RD/home" --entrypoint bash "$CIMG" -c "$*"; }
for step in "$@"; do
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  case "$step" in
    a) client a "python3 -I $T4/check.py $R $RD/check.$ts.json > $RD/check.$ts.txt 2>&1"
       echo "T4a -> $RD/check.$ts.txt" ;;
    b) mkdir -p "$RD/gsm8k.$ts"
       client b "$V/bin/sgl-eval run gsm8k --base-url $R/v1 --model $SERVED_MODEL --load-preset-from-model-id zai-org/GLM-5.2-FP8 --chat-template-kwarg enable_thinking=true --num-examples ${T4_GSM8K_N:-200} --out-dir $RD/gsm8k.$ts > $RD/gsm8k.$ts/run.log 2>&1"
       echo "T4b -> $RD/gsm8k.$ts/run.log" ;;
    c) node docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER_PREFIX-decode-0" | grep -q '^SGLANG_SIMULATE_ACC_LEN=.' &&
         { echo "T4c refused: decode runs with SGLANG_SIMULATE_ACC_LEN (simulated acceptance)" >&2; exit 2; }
       client c "python3 -I $T4/accept.py $R $D $RD/accept.$ts.json > $RD/accept.$ts.txt 2>&1"
       echo "T4c -> $RD/accept.$ts.txt" ;;
    d) echo "T4d: not implemented in the launcher (see design/t4_runbook.md)" >&2; exit 2 ;;
    *) sed -n '2,12p' "$0"; exit 2 ;;
  esac
done
