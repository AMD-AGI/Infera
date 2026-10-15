# a2: pipeline bring-up without HiCache (its 4 x 270 GB host pool takes long to pin and to
# release on every restart). Not a scored configuration.
MIXED_HICACHE=0 MIXED_HICACHE_SIZE=0
HICACHE_HOST_REGISTER_CHUNK_GB=""
AGENTX_WARMUP_REQUESTS_PER_LANE=2
