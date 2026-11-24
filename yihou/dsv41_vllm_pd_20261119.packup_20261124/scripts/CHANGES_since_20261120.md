# scripts/kit changes since pack-up 20261120

Reference: ../dsv41_vllm_pd_20261119.packup_20261120/scripts/kit vs workspace dsv41_vllm_pd_20261119/scripts/kit
(diff -rq, 2026-11-24). All other kit files are byte-identical to the 20261120 pack-up.

## Modified

- serve.sh
  - New opt-in `MC_STOPPED_FIX=1`: the container entry patches vLLM's mooncake_connector.py in-container so KV is also sent
    when the prefill leg ends FINISHED_STOPPED (upstream only sends on FINISHED_LENGTH_CAPPED, so the decode pull waits until
    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT). Source per comment: llying-001/vllm#3; comment says vLLM main 7a62ca0 still has the bug.
    Refuses to start (exit 12) if the patch target is not matched exactly once.
  - Docker RDMA flags split into `DOCKER_RDMA_DEV` (--device /dev/infiniband) and `DOCKER_RDMA_LOCK` (IPC_LOCK + memlock),
    both default 1 = previous behaviour. Comment: debug split for the 2026-11-20 Triton "Code 209" hunt.
- serve_pd.sh
  - Records `MC_STOPPED_FIX`, `DOCKER_RDMA_DEV`, `DOCKER_RDMA_LOCK` in the run's recorded env/versions block (3 echo lines).

## New (crusoe launch helpers; paths hard-coded to /shared_nfs/yihou/dsv41_vllm_pd_20261119)

- cru_pd_chain.sh: run PD points back to back on one crusoe node (serve_pd.sh --wait -> run_point.sh official -> stop.sh
  --prefix per point) so GPUs do not idle between points; logs to $DSV41_NFS_ROOT/launch/<tag>.{serve,point}.log.
- cru_launch_point.sh: launch one colocated point (serve.sh + run_point.sh official, background, with cru_env.sh); prints tag.
- cru_swap_after.sh: wait for CHAIN_DONE of an old tag, print its ledger row, stop it, then launch a new colocated point.
- cru_seq_launch.sh: launch several points one at a time, starting the next only after the previous server logs "ready after"
  (or an error), max 20 min each.
- cru_seq_swap.sh: sequential swap queue (wait old CHAIN_DONE up to 3 h, stop, launch next) with MIN_GAP (default 480 s)
  between launches; comment: concurrent server starts stalled crusoe nodes 3x on 2026-11-21/22.

## Present in the 20261120 pack-up but no longer in the workspace kit

- cru_launch.sh, cru_pd.sh (not present in dsv41_vllm_pd_20261119/scripts/kit on 2026-11-24; why: unknown).
