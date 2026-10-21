#!/usr/bin/env bash
# Queue step: make both nodes ready for the PR 210 kit. Image present (built from pr210/src-aab4cc36 if missing),
# the kit's ports free, every GPU's VRAM below 2%, and on the prefill node the kit's WORK_DIR seeded with the
# InferenceX checkout at 918524ff and the AgentX dataset (the kit would otherwise clone / download them).
# Usage: kit_prep.sh
set -uo pipefail
source "$(dirname "$0")/kit_env.sh"
IX_SRC=/perf_apps/liyingli/bench_agentx/glm52-c80-20260922/scripts/bench-harness/.cache/InferenceX
HF_SRC=/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/cache/agentx/hf/hub
WD=/scratch/liyingli_2p1d/pr210/work
PORTS="24701 24702 28300 27201 27301 28398 27757 27758 27759 27760 29051 $(seq -s ' ' 27434 27444) $(seq -s ' ' 27534 27548)"

for node in "$P_NODE" "$D_NODE"; do
    ssh "$node" "docker image inspect infera-sglang:pr210-aab4cc36 >/dev/null 2>&1" || {
        say "$node: no image, building"
        ssh "$node" "bash $R/scripts/pr210/build_image.sh $R/pr210/src-aab4cc36 infera-sglang:pr210-aab4cc36" ||
            die "$node: image build"
    }
    busy="$(ssh "$node" "ss -lntH | awk '{print \$4}' | sed 's/.*://' | sort -u" | grep -xF -f <(tr ' ' '\n' <<< "$PORTS") | tr '\n' ' ')"
    [[ -z "$busy" ]] || die "$node: kit ports in use: $busy"
    vram="$(ssh "$node" "rocm-smi --showmemuse --csv" | awk -F, '/^card/ && $2+0 >= 2 {printf "%s:%s%% ", $1, $2}')"
    [[ -z "$vram" ]] || die "$node: VRAM in use: $vram"
    say "$node: image ok, ports free, VRAM free"
done
ssh "$P_NODE" "mkdir -p $WD/cache/agentx/hf && \
    { [[ \"\$(git -C $WD/cache/InferenceX rev-parse HEAD 2>/dev/null)\" == 918524ff94045b3f091115f1051c22a8588edf2b ]] || \
      { rm -rf $WD/cache/InferenceX && cp -a $IX_SRC $WD/cache/InferenceX; }; } && \
    cp -rn $HF_SRC $WD/cache/agentx/hf/ && git -C $WD/cache/InferenceX rev-parse HEAD && du -sh $WD/cache/agentx/hf" ||
    die "seeding $WD on $P_NODE"
say "done"
