# pd1n_dpa_decode_dp4.sh plus vLLM's stall watchdog: EngineCore and every worker dump all thread stacks to
# $STACK_DUMP_DIR when they stop feeding it for 60 s (diagnostics only; off by default in vLLM).
source "$W/configs/pd1n_dpa_decode_dp4.sh"
VLLM_ARGS+=(--watchdog-config "{\"dump_dir\":\"${STACK_DUMP_DIR:?STACK_DUMP_DIR}\",\"timeout\":60}")
