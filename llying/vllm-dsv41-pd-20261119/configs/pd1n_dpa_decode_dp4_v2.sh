# pd1n_dpa_decode_dp4.sh with knobs, all off by default: D_UTIL (--gpu-memory-utilization), D_MAXSEQS (--max-num-seqs per
# DP rank; B300 caps 18-168 by concurrency), D_EXTRA_ARGS (more words); SPEC_K / GOLDEN_AL via spec_k.sh (GOLDEN_AL set:
# synthetic acceptance, InferenceX scoring; unset with SPEC_GOLDEN unset: block rejection for correctness probes);
# STACK_DUMP_DIR: vLLM's stall watchdog.
source "$W/configs/pd1n_dpa_decode_dp4.sh"
source "$W/configs/argtools.sh"
[ -z "${D_UTIL:-}" ] || args_set --gpu-memory-utilization "$D_UTIL"
[ -z "${D_MAXSEQS:-}" ] || args_set --max-num-seqs "$D_MAXSEQS"
source "$W/configs/spec_k.sh"
[ -z "${STACK_DUMP_DIR:-}" ] || VLLM_ARGS+=(--watchdog-config "{\"dump_dir\":\"$STACK_DUMP_DIR\",\"timeout\":60}")
# shellcheck disable=SC2206
[ -z "${D_EXTRA_ARGS:-}" ] || VLLM_ARGS+=($D_EXTRA_ARGS)
