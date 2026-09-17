# Key runtime.log lines — extracted, not the whole logs

Full runtime.log files are multi-megabyte (they contain single-line tqdm bars) and are NOT
in this packup, by the same decision taken for the previous packup in this project. They
remain untouched on disk at:
  /home/yihou/dev/git.16-19/infera.dev.yihou.sglang.bench.fast.script/temp_workspace/isl_hetero_batch_yihou_20260915-1114/iterations/<point>/runtime.log

An extract cannot prove the absence of something. What follows is what was searched for and
found; if you need to establish that something did NOT appear, go to the originals.

================ s2_pct0_yihou ================
--- backend selection (which kernel actually ran) ---
      8 gfx950 fused DSA indexer enabled
      8 [dense-decode] DSA dual-graph enabled: capturing dense (k-only) + sparse (full indexer) decode graphs; dispatch on max_kv_len vs index_topk=
      8 Loading tilelang libs from dev root: /opt/tilelang/build
      8 FlyDSL sparse MLA decode declined: q shape (48, 64, 576), need (seq, 8 or 16, 576)
      7 FlyDSL sparse MLA decode declined: q shape (8, 64, 576), need (seq, 8 or 16, 576)
      1 Set DSA backends for fp8_e4m3 KV Cache: prefill=flydsl, decode=flydsl.
      1   0%|          | 0/1 [00:00<?, ?it/s]Capturing batches (bs=8 avail_mem=41.52 GB):   0%|          | 0/1 [00:00<?, ?it/s][2026-09-15 12:57:1
--- progress sample (context span) ---
[INTERNAL-DECODE TP0] iteration=1 context_min=70004 context_max=70004 useful=32 target_graph=True
[INTERNAL-DECODE TP0] iteration=2 context_min=70008 context_max=70008 useful=64 target_graph=True
[INTERNAL-DECODE TP0] iteration=2700 context_min=79758 context_max=79758 useful=78064 target_graph=True
--- topology / batch line ---
[INTERNAL-DECODE TP0] topology={'tp_rank': 0, 'tp_size': 8, 'pp_rank': 0, 'pp_size': 1, 'dp_rank': 0, 'dp_size': 8, 'attn_tp_rank': 0, 'attn_tp_size': 1, 'attn_cp_rank': 0, 'attn_cp_size': 1, 'attn_dc

================ s3_pct100_yihou ================
--- backend selection (which kernel actually ran) ---
      8 gfx950 fused DSA indexer enabled
      8 [dense-decode] DSA dual-graph enabled: capturing dense (k-only) + sparse (full indexer) decode graphs; dispatch on max_kv_len vs index_topk=
      8 Loading tilelang libs from dev root: /opt/tilelang/build
      8 FlyDSL sparse MLA decode declined: q shape (48, 64, 576), need (seq, 8 or 16, 576)
      7 FlyDSL sparse MLA decode declined: q shape (8, 64, 576), need (seq, 8 or 16, 576)
      1 Set DSA backends for fp8_e4m3 KV Cache: prefill=flydsl, decode=flydsl.
      1   0%|          | 0/1 [00:00<?, ?it/s]Capturing batches (bs=8 avail_mem=41.52 GB):   0%|          | 0/1 [00:00<?, ?it/s][2026-09-15 13:01:5
--- progress sample (context span) ---
[INTERNAL-DECODE TP0] iteration=1 context_min=300004 context_max=300004 useful=32 target_graph=True
[INTERNAL-DECODE TP0] iteration=2 context_min=300008 context_max=300008 useful=64 target_graph=True
[INTERNAL-DECODE TP0] iteration=2700 context_min=309758 context_max=309758 useful=78064 target_graph=True
--- topology / batch line ---
[INTERNAL-DECODE TP1] topology={'tp_rank': 1, 'tp_size': 8, 'pp_rank': 0, 'pp_size': 1, 'dp_rank': 1, 'dp_size': 8, 'attn_tp_rank': 0, 'attn_tp_size': 1, 'attn_cp_rank': 0, 'attn_cp_size': 1, 'attn_dc

