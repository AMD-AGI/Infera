#!/usr/bin/env bash
# Run the two arms of the digit-loop reproduction against the live router.
#   arm1: conc=1   n=32  (baseline -- same prompts, no concurrency pressure)
#   arm2: conc=128 n=512 (the reported failing point; exp07's 4x conc prompt count)
# Same prompt generator + same salt => identical prompt CONTENT across arms, so any verdict
# delta is caused by concurrency alone.
set -euo pipefail
PREFILL_HOST="${PREFILL_HOST:-chi2867}"; PREFILL_IP="${PREFILL_IP:-10.2.122.44}"
PORT="${ROUTER_PORT:-8002}"
CTR=pd_uni
KIT=/mnt/vast/c_huggingface/glm52_longctx_pd
MODEL=glm5.2-mxfp4
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@149.28.124.225 "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 \"$2\""; }

ARM="${1:-both}"
run(){ local conc="$1" n="$2" tag="$3"
  J "$PREFILL_HOST" "docker exec -d $CTR bash -c 'python3 $KIT/stress_capture.py http://$PREFILL_IP:$PORT $MODEL $conc $n 1024 1024 $KIT/cap_$tag.json 0 > $KIT/cap_$tag.log 2>&1'"
  echo "launched arm $tag (conc=$conc n=$n) -> $KIT/cap_$tag.{json,log}"
}
case "$ARM" in
  base) run 1 32 base ;;
  c128) run 128 512 c128 ;;
  both) run 1 32 base ;;
esac
