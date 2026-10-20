# b5 (crsuse2-m2m-140): b0r plus --disable-custom-all-reduce, the correctness fix (DP attention +
# aiter custom all-reduce in decode graphs garbles outputs; llying/mix-v0522-correctness-20261019).
# Same node, image and settings as b0r; measures what the fix costs at C48.
source "$AGG/config/round.b0r.sh"
MIXED_EXTRA_ARGS="--disable-custom-all-reduce"
