# a3: scored runs, HiCache 270 GB per rank with write_through (every computed page also copied to
# host at once; the GPU pool can then drop pages without a copy).
MIXED_HICACHE_WRITE_POLICY=write_through
