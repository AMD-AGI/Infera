# agg_tp2.sh with InferenceX's AgentX throughput acceptance: DSpark verifies with the golden acceptance length
# instead of block rejection (infx/srt_slurm/synthetic_acceptance.py; deepseek-v4.1-flash thinking_on, 5 drafts:
# 3.51 in infx/golden_al_distribution/deepseek-v4.1-flash__dspark__native__probabilistic-block.yaml).
source "$W/configs/agg_tp2.sh"
source "$W/configs/argtools.sh"
args_set --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
