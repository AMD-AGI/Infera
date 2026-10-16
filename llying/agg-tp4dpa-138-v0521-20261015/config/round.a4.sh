# a4: write_back + mem-fraction 0.92 together (user's call, 10-16 03:05: both are capacity levers).
# a3 C64 lost 3.5 points of prefix hit to a full host pool (write_through keeps a host copy of every
# GPU page, so distinct cache ~= host pool, 5.93M tokens per rank) and ran the GPU pool at 0.69 mean /
# 14% of the time above 0.95. write_back keeps only evicted pages on host (distinct cache ~= GPU + host);
# 0.92 adds ~20 GB of KV per GPU (~37 GB was still free after graph capture at 0.85).
MIXED_HICACHE_WRITE_POLICY=write_back
MIXED_MEM_FRACTION=0.92
