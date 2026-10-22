#!/usr/bin/env bash
# Render the docker run argv (one arg per line) of prefill 0 and decode for a stage, docker/ip stubbed.
L=$(cd "$(dirname "$0")/.." && pwd); s=$1
export NK=999 NODE_IP=10.0.0.9 RENDER_IP=10.0.0.9
source "$L/stages/S$s.env"
export WORK_DIR=$L/render/work PATH=$L/render/stub:$PATH RENDER_OUT=$L/render/S$s.argv.txt
: > "$RENDER_OUT"
bash "$L/kit/engine/worker.sh" prefill 0 && bash "$L/kit/engine/worker.sh" decode
