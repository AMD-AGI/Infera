#!/usr/bin/env bash
# Shared by run_decode.sh / run_profile.sh / equivalence scripts. Sourced, never executed. Deletes nothing.
# Host-side layout: KIT_DIR (kit copy, mounted ro at /kit), WORK_DIR (mounted at /out). Outputs: $WORK_DIR/iterations/NAME.
TOOL_IN=/kit/vllm_inproc/tool3_spec_step
CONTAINER=${CONTAINER:-yihou-inproc-v3}
PINNED_DIGEST=${PINNED_DIGEST:-sha256:51ceb17daa7d00c40a9978fd7c98337ffc3cb538719422fcdd9c085c1329db37}
MODEL=${MODEL:-/models/DeepSeek-V4.1-Flash}
KIT_DIR=${KIT_DIR:-/data/yihou/inproc_bench_20261125/kit_v3}
WORK_DIR=${WORK_DIR:-/data/yihou/inproc_bench_20261125/work_v3}

v3_die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

# v3_expand_args FILE... -> one arg per line (comments skipped)
v3_expand_args() {
    local f
    for f in "$@"; do
        [ -f "$f" ] || v3_die "args file $f missing"
        grep -v '^[[:space:]]*#' "$f" | tr -s ' \t' '\n\n' | grep -v '^$'
    done
}

# v3_check_container: pinned image + our owner label, before anything is measured.
v3_check_container() {
    [[ "$CONTAINER" =~ ^yihou-inproc-v3[A-Za-z0-9_-]*$ ]] || v3_die "container must be yihou-inproc-v3*"
    test "$(docker inspect -f '{{.Image}}' "$CONTAINER")" = "$PINNED_DIGEST" || v3_die "$CONTAINER is not the pinned image"
    test "$(docker inspect -f '{{index .Config.Labels "owner"}}' "$CONTAINER")" = yihou || v3_die "$CONTAINER is not ours"
}

# v3_snapshot OUT_HOST OUT_CTR CMD_TEXT : provenance written before launch (GLM-5.2 run_decode.sh set + kit provenance).
v3_snapshot() {
    local out=$1 ctr_out=$2 cmd=$3
    printf '%s\n' "$cmd" > "$out/command.txt"
    cp "$KIT_DIR/GIT_HEAD" "$out/git_head.txt" 2>/dev/null || echo unknown > "$out/git_head.txt"
    cp "$KIT_DIR/CODE_DIFF" "$out/code.diff" 2>/dev/null || : > "$out/code.diff"
    (cd "$KIT_DIR/vllm_inproc/tool3_spec_step" && sha256sum bench/*.py scripts/*.sh scripts/*.py) > "$out/code_hashes.sha256"
    cp -a "$KIT_DIR/vllm_inproc/tool3_spec_step/bench" "$out/bench_snapshot"
    docker inspect "$CONTAINER" | docker exec -i "$CONTAINER" python3 /kit/common/redact.py --docker-inspect > "$out/container-inspect.json"
    docker exec "$CONTAINER" python3 /kit/common/provenance.py "$ctr_out" --engine vllm $cmd > /dev/null
}
