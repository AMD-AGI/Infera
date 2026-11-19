# Source after a worker config (needs argtools.sh). SPEC_K: DSpark num_speculative_tokens (unset: keep the config's 5).
# GOLDEN_AL (generation role only): verify with InferenceX's golden acceptance length for that K instead of block
# rejection (infx/golden_al_distribution/deepseek-v4.1-flash__dspark__...: thinking_on K=3 2.90, K=4 3.26, K=5 3.51).
if [ -n "${SPEC_K:-}" ] || [ -n "${GOLDEN_AL:-}" ]; then
    _k=${SPEC_K:-5}
    if [ -n "${GOLDEN_AL:-}" ]; then
        args_set --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$_k,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$GOLDEN_AL,\"enable_adaptive_verification\":false}"
    else
        args_set --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$_k,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}"
    fi
fi
