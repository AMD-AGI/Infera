# Environment (delta, 2026-11-20 05:30Z – 11-24)

Unchanged items (dataset, client venv, InferenceX commit, model paths, NFS/node-local roots, crusoe access):
`../dsv41_vllm_pd_20261119.packup_20261120/environment.md`.

## Software (from the copied `results/runs_*/<tag>/versions.txt`; identical on every run of this window)
- Serving image: `vllm/vllm-openai-rocm:nightly-60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d@sha256:67a3a0fb79893fc5a0fcff5f9b739ba8fec7e920875a12e0574c2e9f50cd814c`
  (all 114 crusoe ledger rows and all openai rows record this image).
- In-image: vllm 0.34.0rc1.dev571+g60bd6dd2e.rocm100, amd-aiter 0.1.29, amd_mori 1.2.3.post1,
  mooncake-transfer-engine-rocm 0.3.14 (tag v0.3.14 = e1ea5d31), nixl-rocm 1.5.0.
- In-container patch (PD configs with `MC_STOPPED_FIX=1`): vLLM `mooncake_connector.py` request_finished also sends KV on
  FINISHED_STOPPED (serve.sh; logged as "MC_STOPPED_FIX applied" in each role's entry.log).
- Router (PD): vllm-router 0.1.14 (wheel installed in the container). Client: agentx-harness 1.0.14 venv; InferenceX @ db3fc796.
- Repo: infera `dev/yihou/dsv41` at 26c0c82b6 when this pack-up was written.

## Nodes and jobs used in this window
| cluster | node | job | held (UTC) | used for | how it ended |
|---|---|---|---|---|---|
| crusoe | crsuse2-m2m-v3-186 | 300344 | 11-20 05:13 – 05:48 | c1-r1 c128, e7s PD smoke, e7r ref | cut for idleness (no score) |
| crusoe | crsuse2-m2m-v3-129 | 300599 | 11-20 10:15 – ~12:05 | e1-r1cru c128, e7r c32, e7s* Code-209 split | cut for idleness |
| openai | smci355-ccs-aus-n06-25 | 36733 | 11-21 ~03:01 – ~08:13 | e2-r2rep, e3-r2-c512, e4b/e4c-rep, e7/e7h PD, e7s* smokes | preempted (requeued) |
| openai | smci355-ccs-aus-n01-29 | 36884 | 11-21 07:39 – 08:13 | lent to the kf teammate | preempted |
| openai | smci355-ccs-aus-n02-25 | 36884 (requeued) | 11-21 12:00 – 12:04:47 | e7h c384b / e7p c256 (loading only) | cancelled by pool daemon (docker_free < 200 G) |
| crusoe | crsuse2-m2m-v3-200 | 302960 | 11-21 ~13:45 – 11-22 01:44 | e7th/e7tp/e7u PD (TCP), R1/E4 curves, R2 attempts | 12 h job limit |
| crusoe | crsuse2-m2m-v3-014 | 303579 | 11-22 01:45 – ~13:45 | R1/E4 curves | job limit |
| crusoe | crsuse2-m2m-v3-014 | 304125 | 11-22 13:46 – 11-23 01:34 | R1/E4 curves | released at end |
| crusoe | crsuse2-m2m-v3-014 | 305057 | 11-23 01:38 – 12:35 | R1/E4 curves (GPU4-7 idle from 04:32 by rule) | released after final points |

Job/node per crusoe run is inferred from launch time vs grant windows in `process/team-polls.md` (run files carry no job id).
openai NFS root `/perf_apps/yihou/dsv41_vllm_pd_20261119`; on n02-25 the node-local root was `/scratch/yihou/dsv41_vllm_pd_20261119`
(`/data` root-only there). crusoe roots unchanged (`/shared_nfs/yihou/…`, `/mnt/m2m_nobackup/yihou/…`), kit at `$R/ws/scripts/kit`.
Topology fact (rocm-smi --showhw on v3-014): GPU0-3 = PCIe domain 0002, GPU4-7 = domain 0003. crusoe nodes have no ib_peer_mem.

## Full raw outputs (not in git)
- openai: `/perf_apps/yihou/dsv41_vllm_pd_20261119/runs/<tag>/` (server/router/client logs, aiperf artifacts), ledger `results/ledger.csv`.
- crusoe: `/shared_nfs/yihou/dsv41_vllm_pd_20261119/runs/<tag>/` and `launch/<tag>.{serve,point,stop}.log`, ledger `results/ledger.csv`.

## Secrets needed (names only; values not stored)
- openai: `ssh openai-login` (slurm), node ssh only while holding a job; pool manager session books nodes.
- crusoe: `ssh spur005` (or spur006) login; `spur exec <job> -- …` to reach a held node.
- Docker Hub pull (anonymous OK). HuggingFace token not needed at run time (offline cache). No API keys in configs (grep KEY|TOKEN|
  SECRET|PASSWORD matched only TOKENIZER= / max-num-batched-tokens / tokenizer-mode).
