# DSv4.1-Flash vLLM kit: how to run E0, R4, R1, R2 on a granted openai node

Kit dir `K=/home/yihou/dev/git.16-19/infera.dsv41/yihou/dsv41_vllm_pd_20261119/scripts/kit` (HOME NFS, visible on nodes).
Run artefacts: NFS `/perf_apps/yihou/dsv41_vllm_pd_20261119/{preflight,runs/<tag>}`; node-local runtime dirs
`/data/yihou/dsv41_vllm_pd_20261119/<tag>/`. Containers `yihou-dsv41-<tag>` (+ `-client`). Review findings: `REVIEW.md`.

Rules that apply to every step: start only on a pool "granted" message; one fresh server + fresh tag per scored point;
never touch containers not named `yihou-dsv41-*`; delete only named paths containing `/yihou/`; English in files.

## 0. Get the node
1. Ask the pool-manager session `machines-and-clusters-b1` for an openai node; wait for "granted" with node + job id.
2. `ssh smci355-ccs-aus-nXX-YY` from a16-19 (only works while the job runs; fallback
   `ssh openai-login srun --jobid <JOB> --overlap -N1 -n1 <cmd>`).
3. Note `JOB`, node, lease end (UTC). Export on the node for every step below:
   `export KIT_STOP_BY=<lease end ISO-8601 UTC>`.

## 1. E0: node facts and images (~15-40 min, mostly image pull)
```bash
K=/home/yihou/dev/git.16-19/infera.dsv41/yihou/dsv41_vllm_pd_20261119/scripts/kit
bash $K/pull_images.sh --list                 # what will be pulled
bash $K/pull_images.sh r1-dp2-c128 --router-wheel   # 60bd image (r1/r2 share it) + vllm-router 0.1.14 wheel to NFS
bash $K/pull_images.sh r4-tp2-c32             # EXPECTED TO FAIL: tag 404 on Docker Hub (REVIEW.md risk 1); see step 2
bash $K/preflight.sh --in-image-all           # -> /perf_apps/yihou/dsv41_vllm_pd_20261119/preflight/<host>-<ts>/facts.{json,md}
bash $K/client_selftest.sh                    # AgentX client in the server image: must print SELFTEST PASS
```
Read `facts.md` before serving. Go / no-go:
- foreign containers on `/dev/kfd` or any GPU VRAM > 10 % -> do not use those GPUs; tell the pool manager.
- `docker info` root dir free >= 200 GB; `/data/yihou` writable.
- host DRAM free: each r2 server needs 449 GB (UMBP pool). Run at most `floor(free / 500 GB)` r2 servers, and never more
  than 2 per node until a real measurement says otherwise.
- in-image list: `pip list | grep -i -E "vllm|aiter|nixl|mooncake|mori"` per image (G2 in SYNTHESIS).
- Run the HIP-vs-rocm-smi index command that preflight printed into its notes; GPU lists below are HIP indices.

## 2. R4 (TP2 c32, reference 107,989 tok/s/GPU, CI job 114309765517) -- E1 part 1
Image `vllm/vllm-openai-rocm:nightly-rocm100-7d0b4e57...` is not on Docker Hub any more. Before starting:
`docker image inspect vllm/vllm-openai-rocm:nightly-rocm100-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe` on the node. If absent,
stop and ask the user (options: a colleague's local copy / `docker save`; or skip R4). Do not substitute another image
under the R4 name.
```bash
T=e1-r4-c32-$(date -u +%m%d%H%M)
bash $K/serve.sh r4-tp2-c32 0,1 $T --dry-run | less        # check argv once
bash $K/serve.sh r4-tp2-c32 0,1 $T --wait                  # ready can take ~15-30 min (graph capture); do not restart early
setsid nohup bash $K/run_point.sh $T http://127.0.0.1:8000 32 2 official >/dev/null 2>&1 &
```
Use `r4-tp2-c32` (the CI recipe). `r4-tp2-c32-s8` (seqs 8 / 8192) was never run in CI on this image: experiment only.

## 3. R1 (TP1xDP2 c128, reference 271,329, CI row 448240) -- E1 part 2
```bash
T=e1-r1-c128-$(date -u +%m%d%H%M)
bash $K/serve.sh r1-dp2-c128 2,3 $T --port-offset 10 --wait     # router :8010, engine :6110
setsid nohup bash $K/run_point.sh $T http://127.0.0.1:8010 128 2 official >/dev/null 2>&1 &
```
With `--port-offset 0` the endpoints are router `:8000`, engine `:6100`. Every server on one node needs its own offset
(r4 uses 8000 too) and disjoint GPUs; serve.sh refuses overlaps with our own containers and busy ports.
A nonzero offset on a DP config appends `--data-parallel-rpc-port 29550+N` (not in CI); run the DP server with offset 0
when it is the only DP server on the node.

## 4. R2 (TP1xDP2 + UMBP host-DRAM tier; c384 reference 427,451 row 448242, c256 401,896 row 448241) -- E2
```bash
T=e2-r2-c384-$(date -u +%m%d%H%M)
bash $K/serve.sh r2-dp2-umbp-c384 4,5 $T --port-offset 20 --wait   # UMBP pool fault-in first (CI allows 1800 s), then vLLM
setsid nohup bash $K/run_point.sh $T http://127.0.0.1:8020 384 2 official >/dev/null 2>&1 &
```
c256: same with `r2-dp2-umbp-c256` and concurrency 256. Each r2 server takes 449 GB host DRAM.

## 5. While it runs (watchdog teammate, every 10 min)
From a16-19: `KIT_NODE=<node> KIT_JOBID=<JOB> bash $K/watchdog.sh <tag> [<tag> ...]` (exit 0 OK, 1 alert, 3 unreachable).
Logs: `/perf_apps/yihou/dsv41_vllm_pd_20261119/runs/<tag>/{entry,server,umbp,router}.log`, bench in `runs/<tag>/bench/`.
If the job is requeued, the old node keeps our containers (no job binding in the kit): report it to the pool manager and
stop them as soon as any access to that node exists (`stop.sh <tag>` there).

## 6. After each point
1. `run_point.sh` runs `summarize.py` and appends to `<ws>/results/ledger.csv`; read `runs/<tag>/bench/run_point.log`.
   Manual: `python3 $K/summarize.py /perf_apps/yihou/dsv41_vllm_pd_20261119/runs/<tag> --mode official`.
2. Check the contract line (duration 3600, warmup 10/lane, seed 42, error ratio), AL ~3.51, prefix hit, KV usage.
   E1 pass band: within ~2-3 % of the reference; otherwise debug the harness before anything else.
3. Stop the server: `bash $K/stop.sh <tag>` (never `docker rm`/`pkill`).
4. Clean node-local runtime dirs of that tag only, by literal path:
   `rm -r /data/yihou/dsv41_vllm_pd_20261119/<literal-tag>` (typed out, not from a variable). NFS run dirs are kept.

## 7. Known gaps (details in REVIEW.md)
R4 image availability; no slurm-job binding of containers; host DRAM headroom for UMBP; harness code CI 94b81ec4 vs our
db3fc796; router `/health`, root squash on NFS, PyPI reachability, UMBP-in-one-container, port-offset sufficiency: unverified.
