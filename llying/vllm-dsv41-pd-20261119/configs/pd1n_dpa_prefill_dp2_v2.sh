# pd1n_dpa_prefill_dp2.sh with knobs from B300's prefill (dsv41flash b300-fp4 agentx disagg-variants.yaml), all off by
# default: P_UTIL (--gpu-memory-utilization, B300 0.95), P_LONG_PREFILL (--long-prefill-token-threshold, B300 8192),
# P_MAXSEQS (--max-num-seqs, B300 64), P_EXTRA_ARGS (more words); SPEC_K via spec_k.sh (block rejection on the prefill).
source "$W/configs/pd1n_dpa_prefill_dp2.sh"
source "$W/configs/argtools.sh"
[ -z "${P_UTIL:-}" ] || args_set --gpu-memory-utilization "$P_UTIL"
[ -z "${P_LONG_PREFILL:-}" ] || args_set --long-prefill-token-threshold "$P_LONG_PREFILL"
[ -z "${P_MAXSEQS:-}" ] || args_set --max-num-seqs "$P_MAXSEQS"
unset GOLDEN_AL
source "$W/configs/spec_k.sh"
# shellcheck disable=SC2206
[ -z "${P_EXTRA_ARGS:-}" ] || VLLM_ARGS+=($P_EXTRA_ARGS)
