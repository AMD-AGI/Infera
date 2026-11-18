import argparse, shlex, sys
from sglang.srt.server_args import ServerArgs
p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p)
argv = shlex.split(open(sys.argv[1]).read())
a = p.parse_args(argv)
for k in ["tp_size","dp_size","enable_dp_attention","kv_cache_dtype","dsa_prefill_backend","dsa_decode_backend","cuda_graph_max_bs_decode","speculative_algorithm","speculative_num_steps","speculative_eagle_topk","speculative_num_draft_tokens","disable_custom_all_reduce","mem_fraction_static","max_running_requests","watchdog_timeout"]:
    print(k, getattr(a, k, "<NO FIELD>"))
