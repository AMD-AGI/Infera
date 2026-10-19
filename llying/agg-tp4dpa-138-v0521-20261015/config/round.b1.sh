# b1 (crsuse2-m2m-140): a3's C48 settings -- write_through 270 GB, mem 0.85, delayer 60
# passes, 1 tokenizer -- plus the delayer piggyback: while any rank continues a chunked
# prefill (an extend step that runs the other ranks idle), ranks with admissible requests
# prefill in the same step instead of waiting for the delayer
# (patches/runtime-mounts/*piggyback*, SGLANG_PREFILL_DELAYER_PIGGYBACK=1).
AGG_NODE=crsuse2-m2m-140
source "$AGG/config/round.a3.sh"
YIHOU_MOUNTS="$AGG/config/mounts.piggyback.tsv"
MIXED_EXTRA_ENV="SGLANG_PREFILL_DELAYER_PIGGYBACK=1"
