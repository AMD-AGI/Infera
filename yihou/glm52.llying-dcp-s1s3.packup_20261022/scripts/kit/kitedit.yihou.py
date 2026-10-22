#!/usr/bin/env python3
"""Apply the dcp_pd kit edits (llying_dcp_plan.md section 1) to a copy of Infera
examples/sglang_2p1d_glm5.2 @ d395c3b3 + kit-v6-pp4.diff. Each edit asserts its anchor once.
Usage: kitedit.yihou.py KIT_DIR"""
import sys

K = sys.argv[1]


def rep(path, old, new, count=1):
    p = f"{K}/{path}"
    s = open(p).read()
    assert s.count(old) == count, (path, old[:70], s.count(old))
    open(p, "w").write(s.replace(old, new))


# 1 common.sh: DECODE_SSH=local runs the decode leg on this host
rep("engine/common.sh",
    'on_decode() { ssh -o BatchMode=yes "$DECODE_SSH" "$1"; }',
    '# yihou dcp_pd: DECODE_SSH=local runs the decode leg on this node (same-node P4D4).\n'
    'on_decode() { if [[ "$DECODE_SSH" == local ]]; then bash -c "$1"; '
    'else ssh -o BatchMode=yes "$DECODE_SSH" "$1"; fi; }')
# recorded / forwarded settings: add the yihou knobs
rep("engine/common.sh",
    '    KV_EVENTS_PORT KV_SNAPSHOT_PORT ETCD_IMAGE CTR)',
    '    KV_EVENTS_PORT KV_SNAPSHOT_PORT ETCD_IMAGE CTR\n'
    '    PP4_CHUNK PP4_HICACHE MC_TE_FILTERS_OVERRIDE DISAGG_IB_DEVICE INDEX_SHARE SGLANG_OVERLAY DECODE_LAYOUT\n'
    '    DECODE_ENV PREFILL_ENV DECODE_PID_BURN)')

W = "engine/worker.sh"
# 2 Mooncake filter override (same-node HIP IPC: a filter that matches no HCA)
rep(W, '"MC_TE_FILTERS=$RDMA_DEVICES"', '"MC_TE_FILTERS=${MC_TE_FILTERS_OVERRIDE:-$RDMA_DEVICES}"')
# 3 per-GPU ib-device map override
rep(W, '--disaggregation-ib-device "$RDMA_DEVICES"',
    '--disaggregation-ib-device "${DISAGG_IB_DEVICE:-$RDMA_DEVICES}"')
# 4 custom AR on (user 10:12Z)
rep(W, '    --disable-custom-all-reduce\n',
    '    # yihou dcp_pd: custom all-reduce ON (user directive 2026-10-22 10:12Z).\n')
# 5 index_share knob, default true (user 10:09Z)
rep(W, "--json-model-override-args '{\"index_share_for_mtp_iteration\":false}'",
    "--json-model-override-args \"{\\\"index_share_for_mtp_iteration\\\":${INDEX_SHARE:-true}}\"")
# 6 decode layout: dpa4 (llying) | dcp4 (yihou r6b knobs)
rep(W,
    '''    args+=(
        --tp-size "$n_gpus" --dp-size "$n_gpus" --enable-dp-attention
        --mem-fraction-static 0.92 --cuda-graph-max-bs-decode 256 --tokenizer-worker-num 4
        --no-enable-kv-events --kv-events off
    )''',
    '''    case "${DECODE_LAYOUT:-dpa4}" in
    dpa4)   # llying: TP4 + DP4 attention, EP1
        args+=(
            --tp-size "$n_gpus" --dp-size "$n_gpus" --enable-dp-attention
            --mem-fraction-static 0.92 --cuda-graph-max-bs-decode 256 --tokenizer-worker-num 4
            --no-enable-kv-events --kv-events off
        ) ;;
    dcp4)   # yihou r6b decode: TP4 EP4 + DCP4, no DP attention (attn_tp % dcp == 0). Graph bs = max running = 96
            # (r6b: 64/64; raised to admit C80). --ep-size / --max-running-requests override the common ones (last wins).
        env+=(SGLANG_DP_USE_GATHERV=0)
        args+=(
            --tp-size "$n_gpus" --ep-size "$n_gpus" --dcp-size "$n_gpus"
            --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 96 --max-running-requests 96
            --tokenizer-worker-num 4 --no-enable-kv-events --kv-events off
        ) ;;
    *) die "DECODE_LAYOUT must be dpa4|dcp4" ;;
    esac''')
# 7 per-leg extra env (space-separated KEY=VALUE)
rep(W, 'args+=("${extra[@]}")\nenv_args=()',
    '''args+=("${extra[@]}")
# yihou dcp_pd: per-leg extra container env, space-separated KEY=VALUE (e.g. the DCP4 knobs).
if [[ "$role" == prefill ]]; then read -ra env_extra <<<"${PREFILL_ENV:-}"; else read -ra env_extra <<<"${DECODE_ENV:-}"; fi
env+=("${env_extra[@]}")
env_args=()''')
# 8 PID burn also for the decode (same node: RCCL names shm after pids)
rep(W, 'if [[ "$role" == prefill && "$idx" == 1 ]]; then',
    'if [[ "$role" == prefill && "$idx" == 1 ]] || [[ "$role" == decode && "${DECODE_PID_BURN:-1}" == 1 ]]; then')
# 9 drop llying's deepseek_v2.py mount; our branch overlay on both legs
rep(W,
    '''# v0.5.22 reads embed_tokens.embedding_dim on every PP stage, but the middle
# stages hold a PPMissingLayer there; the patched copy reads config.hidden_size.
[[ "$role" != prefill ]] || mounts+=(-v "$KIT/patches/deepseek_v2.py:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py:ro")''',
    '''# yihou dcp_pd: llying's deepseek_v2.py file mount is replaced by our branch overlay (the PP fix is a commit
# there), mounted read-only over the image's sglang on BOTH legs.
[[ -z "${SGLANG_OVERLAY:-}" ]] || mounts+=(-v "$SGLANG_OVERLAY:/sgl-workspace/sglang/python/sglang:ro")''')
# 10 --init on engine containers (the job-bound guard signals PID 1)
rep(W, 'docker run -d --name "$name" --network host', 'docker run -d --init --name "$name" --network host')

# down.sh: stop + save logs; never remove containers or files
D = "engine/down.sh"
rep(D, '''    docker rm -f "$c" >/dev/null || { echo "could not remove $c yet; run down.sh again"; continue; }
    echo "removed $c (log: $out/$c.log)"''',
    '''    echo "stopped $c (log: $out/$c.log); container kept (yihou rule: docker stop, never rm)"''')
rep(D, '''    if on_decode "docker rm -f $c" >/dev/null; then
        echo "removed $c on $DECODE_SSH (log: $out/$c.log)"
    else
        echo "could not remove $c on $DECODE_SSH yet; run down.sh again"
    fi''', '''    echo "stopped $c on $DECODE_SSH (log: $out/$c.log); container kept"''')
rep(D, '|| { ' + 'rm' + ' -f "$out/$c.inspect.json"; continue; }', '|| continue')
rep(D, 'else\n    ' + 'rm' + ' -f "$out/$c.inspect.json"\nfi', 'fi')

# up.sh: router with --init
rep("engine/up.sh", 'docker run -d --name "$CTR-router" --network host',
    'docker run -d --init --name "$CTR-router" --network host')
# agentx.sh: metadata for the PP4 TP1 prefill; client with --init
rep("engine/agentx/agentx.sh", 'PREFILL_TP=4\n', 'PREFILL_TP=1\nPREFILL_PP=4\n')
rep("engine/agentx/agentx.sh", 'docker run --rm --name "$CTR-agentx"', 'docker run --rm --init --name "$CTR-agentx"')
print("kit edits applied")
