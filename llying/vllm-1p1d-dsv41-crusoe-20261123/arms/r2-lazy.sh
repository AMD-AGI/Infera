# Round 2 candidate: round 1 plus lazy UMBP offload on the prefill (vllm-project/vllm#61797, in this image). The eager
# store re-writes ~every GPU-hit and computed token to host DRAM (5.95 TB/h at AUS c768, ~14x the newly computed KV);
# lazy mode stores only GPU blocks about to be evicted, as the B300 recipe's SimpleCPUOffloadConnector does.
source /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r1-capacity.sh
UMBP_EXTRA_JSON='"lazy_offload":true'
