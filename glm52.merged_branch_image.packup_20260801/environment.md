# Environment

Captured **2026-08-01 04:23 UTC**, after the run, by `scripts/envsnap.sh` on each
node. Raw output: `env/env_chi2879.txt`, `env/env_chi2867.txt`.

## Nodes

| | chi2879 | chi2867 |
|---|---|---|
| role | prefill + etcd + router + kvd | decode + kvd |
| data-plane IP | 10.2.122.10 | 10.2.122.44 |
| GPU | 8× AMD Instinct MI355X, **gfx950** | same |
| CPU | 2× AMD EPYC 9575F (256 threads) | same |
| RAM | 3023 GB | same |
| kernel | `6.8.0-124-generic` | `6.8.0-107-generic` |
| amdgpu | 6.16.13 | 6.16.13 |

**The kernels differ between the two nodes** (`-124` vs `-107`). It did not
matter for this run — both carry the same amdgpu and ionic_rdma — but it is
recorded because a future RDMA-level difference would want it.

## RDMA fabric

Identical on both nodes:

- `ionic_rdma` **26.03.3.001**
- 8 rails `ionic_0` … `ionic_7`, **all `PORT_ACTIVE`**
- Host libionic: `/usr/lib/x86_64-linux-gnu/libionic.so.1` →
  `libionic.so.1.1.54.0-187`

The host library is **bind-mounted into the container** at
`/host-libionic/libionic.so` and the image entrypoint
(`infera-inject-host-ionic`) swaps it in. It must match the host's `ionic_rdma`
kmod. If the injection fails, mooncake silently drops to TCP and the run
"succeeds" while measuring nothing — which is why `reset_merged.sh` asserts 8
`PORT_ACTIVE` *inside* the container before proceeding.

## Images

**Base** (pinned, must stay pinned):

    lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x
    RepoDigest sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d
    local Id   sha256:b958aa54ef56e47db315f5a1f424f3b7e5004d8ce5391ed1e027b4f5628627fb

The DSA diffs apply at `--fuzz=0` against sglang
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (release/v0.5.15), so a base bump
fails the build at the patch step rather than mis-applying silently. Intended.

**Built** — `infera/engine-sglang:merged`, built on each node from branch source:

| node | image id |
|---|---|
| chi2879 | `sha256:1f7cf6964ceefed995dad3ea759ad9b8fb84508753289e8d8a34032b80082002` |
| chi2867 | `sha256:0d478433f1b368483a4721ae5f40f276ca33fb50dd2a72112d562b9da64b396a` |

**They differ and that is expected** — independent builds, so Rust router objects
and layer timestamps differ. Content equivalence is what matters and was checked:
18 bytecode/source assertions plus a behavioural smoke test, identical on both.
See `notes.md` §8.

Inside the built image:

- sglang **0.5.15.post1**
- torch **2.9.1+rocm7.2.0.git7e1940d4**
- infera: the branch source, installed by `pip install .[sglang]`. **One** copy
  at `/opt/infera/infera` (the patched base image carried two — `notes.md` §3).

## Repository

- Remote: `git@github.com:AMD-AGI/Infera.git`
- Base: `main` @ `8692fb4ff4b24f20b213a5d166f0b867d20d2bea`
- Branch: `yihou.dev.glm52.merged.experiment` @ `330da16`, **25 commits**
- Worktree used: `~/dev/git.16-19/infera.glm52.merged`

Cherry-pick sources:

| group | branch | range |
|---|---|---|
| A | `worktree-dsa-hip-dp-rows-fix.rebase` (PR #58) | `1ac6222^..78cf750` |
| B | `yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` (PR #59) | `17e34e4^..52d7119` |
| C | `origin/llying/glm5p2_fp8_fixes` (PR #56) | `0360af5`, `6121189`, `01b0534` — **reduced**, see `branch/MERGE_BRANCH.md` |

## External dependencies (not in any repo)

| what | path | notes |
|---|---|---|
| model weights | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | shared VAST NFS (`10.2.123.177:/aac-8634674/…`), mounted `/mnt/vast` on both nodes, 501 T / 75% used |
| model sampling config | `…/GLM-5.2-MXFP4/generation_config.json` | **load-bearing** — temperature 1.0, top_p 0.95. See `notes.md` §1 |
| chat template | `…/GLM-5.2-MXFP4/chat_template.jinja` | applied by the OpenAI-compatible endpoint |
| host libionic | `/usr/lib/x86_64-linux-gnu/libionic.so.1` | bind-mounted; must match the host kmod |
| on-node staging | `/mnt/vast/c_huggingface/merge_20260731/` | logs + scripts from both runs; shared, so both nodes see it |

## Run configuration

Both legs, from `scripts/glm52_leg.sh`:

    --tp-size 8 --dp-size 8 --enable-dp-attention --ep-size 8
    --nsa-prefill-backend tilelang --nsa-decode-backend tilelang
    --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85
    --context-length 32768 --chunked-prefill-size 65536   (= 8192 per DP rank)
    --cuda-graph-max-bs 128 --max-running-requests 2048
    --disaggregation-transfer-backend mooncake
    --disaggregation-ib-device ionic_0,…,ionic_7
    --hicache-size 16
    --kv-events-config '{"publisher":"zmq",...}'

Decode leg with `MTP=1` adds:

    --speculative-algorithm EAGLE --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
    --num-reserved-decode-tokens 256 --disable-custom-all-reduce

`--disable-custom-all-reduce` is required: the aiter custom all-reduce kernel
deadlocks on gfx950 during EAGLE verify.

Router: `python -m infera.server … --router-policy kv-aware --router-backend
python` (the default backend). **The Rust backend was never used** — and it still
carries the bigram bug. See `notes.md` §9.

## Disk

| node | filesystem | size | used |
|---|---|---|---|
| both | `/dev/md0` (`/`) | 838 G | 74% |
| both | `/mnt/vast` | 501 T | 75% |

~30 GB free per node is needed for the built image.

## Secrets

**None are required for the run itself.** Recorded for completeness — names and
sources only, never values:

| what | needed for | where it comes from |
|---|---|---|
| cluster SSH | reaching `chi287x` via `root@149.28.124.225` | your own key; arrange with the cluster owner |
| GitHub SSH | cloning `AMD-AGI/Infera` | your own key |
| docker registry | **not needed** — the base image is public on Docker Hub and the merged image is built locally |
| etcd / router auth | **not needed** — neither is authenticated on this cluster |

No credential value appears in any script, log, or result file in this kit.

## Gaps

- The environment was snapshotted **after** the run, not during. Nothing changed
  on the nodes between the run and the snapshot (no reboots, no driver updates),
  but it is a snapshot of the end state rather than a per-round capture.
- Ambient load was not recorded. Other tenants' containers (`mlperf_gptoss2`,
  `primus_train`, `mtt_pd`) were running throughout on both nodes. They do not
  contend for our 8 GPUs, but they do share host CPU and the NFS mount, so the
  latency figures in `results/raw/` are not clean-machine numbers. Every gate in
  this kit is a correctness or count criterion, not a latency one.
