# Environment

Everything pinned. Where something floats, it is called out.

## Time

| | |
|---|---|
| image build | 2026-08-03 ~11:40 UTC |
| smoke (discarded) | 11:41 – 11:53 UTC |
| **c8 measurement** | **11:59:53 – 12:16:22 UTC** (profiling 12:00:49 – 12:15:49) |
| **c16 measurement** | **12:16:22 – 12:33:03 UTC** (profiling 12:17:30 – 12:32:30) |
| node snapshots | 12:42 UTC — **after** the run, both legs still live, so the recorded cmdlines are the ones that served it |

## Hardware

Two MI355X (gfx950) 8-GPU nodes on the vultr cluster, plus the jump host as the
load generator.

| | chi2835 (prefill) | chi2879 (decode) | jump host |
|---|---|---|---|
| IP | 10.2.122.78 | 10.2.122.10 | 149.28.124.225 |
| kernel | 6.8.0-107-generic | 6.8.0-124-generic | — |
| ROCm driver | **6.16.13** | **6.16.13** | n/a |
| GPUs | 8× gfx950 (MI355X) | 8× gfx950 (MI355X) | not used |
| **RDMA rails `PORT_ACTIVE`** | **8** | **7** | n/a |
| CPU | AMD EPYC 9575F 64-Core, 256 threads | AMD EPYC 9575F 64-Core, 256 threads | 128 cores |
| RAM | 3,023 GB | 3,023 GB | 3,023 GB |
| RDMA fabric | ionic RoCE, `ionic_0..7` | ionic RoCE, `ionic_0..7` less `ionic_5` | n/a |

> **chi2879 runs 7 rails, not 8** — `ionic_5` is `PORT_DOWN` and the leg
> launcher enumerates only active devices. This is inherited from the par8 run,
> is **not controlled for**, and its effect on KV-transfer time is unmeasured.

The load generator runs on the **jump host**, not on a GPU node — the benchmark
is pure HTTP against the router, so it does not contend for GPU.

## Software — the deployment under test

Unchanged from `../par8.glm52.dpaoff.packup_20260803`; **this kit did not restart
or reconfigure anything.**

| | |
|---|---|
| image | `infera/engine-sglang:merged-e` |
| image digest | `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a` |
| repo branch | `yihou.dev.glm52.merged.experiment` |
| **repo commit** | **`ccd68181d3bb76614d942f430dc00da52f8b58ab`** (working-tree state at pack-up; the *image* is what actually ran — see caveat) |
| sglang | v0.5.15.post1 (in-image) |
| model | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST) |
| router | infera Rust router, `10.2.122.78:8100` |
| etcd | `10.2.122.78:2379` |

> **The repo commit above did not build the image.** This kit only sent HTTP
> traffic at an already-running deployment; the engine code that served it is
> whatever is baked into `merged-e` (digest above), built during the par8 work.
> The commit is recorded so the docs/scripts in *this* kit are pinned, not as a
> claim about the engine build. For the engine's provenance see
> `../par8.glm52.dpaoff.packup_20260803/environment.md`.

Full resolved engine command lines for both legs: `env/node_snapshots.txt`.
Summary of the knobs that matter:

| | prefill (chi2835) | decode (chi2879) |
|---|---|---|
| `--tp-size` | 8 | 8 |
| `--context-length` | 262144 | 262144 |
| `--chunked-prefill-size` | **16384** | 65536 |
| `--enable-dp-attention` | **absent (DPA off)** | **present**, `--dp-size 8` |
| `--mem-fraction-static` | 0.80 | 0.85 |
| spec decode | — | `EAGLE`, steps 3, topk 1, **4 draft tokens** |
| `--enable-cache-report` | yes | yes |
| hicache / kvd | `--hicache-size 16`, `infera-kvd` backend | — (image skips kvd on a PD decode leg) |
| transfer | mooncake, ionic ×8 | mooncake, ionic ×7 |

## Software — the benchmark

| | |
|---|---|
| customer kit | ROCm/MAD **PR #173**, commit `158d3c928b8aa2de52de75fb03e0b95d0ef1e410` |
| `replay_caseA.sh` md5 | `7cde1afc627c7e4868eac0fd13741baa` (**unmodified**) |
| aiperf | `github.com/SemiAnalysisAI/aiperf` @ `cquil11/aiperf-agentx-v1.0`, commit **`6cd93f01e6fa9bf7170454a2bb6134aeb25b064b`** |
| aiperf version string | `0.8.0` |
| our aiperf image | `aiperf-agentx:v1.0`, digest `sha256:9ebc838842c76ab3cc2e31420d7bdad261bbe04626ef2b00031e09ea5741f0c7` |
| base image | `python:3.13-slim-bookworm` |
| corpus | frozen `caseA_conformance_corpus.tar.gz` from the PR — 200 sessions / 1,778 requests, seed 42 |

> **The aiperf install is pinned by branch, not by SHA, in
> `scripts/build_aiperf_img.sh`.** The commit above is what *we* got on
> 2026-08-03. If the fork's branch moves, a rebuild will not reproduce this
> image. To pin hard, change the `pip install` target to
> `git+https://github.com/SemiAnalysisAI/aiperf.git@6cd93f01e6fa9bf7170454a2bb6134aeb25b064b`.

> **`transformers` is installed from git main** by aiperf's own dependency spec
> (`transformers @ git+https://github.com/huggingface/transformers.git`). This is
> upstream's choice, it floats, and we did not pin it. Only the tokenizer is
> exercised — torch is absent from the image and aiperf does not need it.

## The measurement configuration

Every value passed to the customer's script, and why:

| var | value | default | reason for the change |
|---|---|---|---|
| `URL` | `http://10.2.122.78:8100` | *(required)* | our router |
| `SERVED` | `glm5.2-mxfp4` | `GLM-5.2-MXFP4` | our `--served-model-name` |
| `TOK` | `$HERE/tokenizer` | `/models/GLM-5.2-MXFP4` | the container mounts only `$HERE`, `/models`, `/shared_nfs` — not `/mnt/vast` |
| `CONCS` | `8 16` | `1 2 4 8 16` | operator decision; conc=1 is unsupported by the scenario |
| `DUR` | `900` | `300` | **scenario-enforced minimum** (`inferencex_agentx_mvp.py:33`) |
| `IMG` | `aiperf-agentx:v1.0` | `rocm/atom-dev:latest` | the default image does not ship the aiperf agentx fork |
| `OUT` | `/root/agentx_20260803/results` | `$HERE/results` | **this was a mistake** — see `notes.md` |

Scenario-forced settings we did not choose (`inferencex_agentx_mvp.py`):
`ignore_eos`, `streaming`, `use_end_to_start_delays`, no input truncation,
`cache_bust = FIRST_TURN_PREFIX`.

Fixed aiperf flags from the customer script: `--trajectory-start-min-ratio 0.90
--trajectory-start-max-ratio 0.98 --use-server-token-count --no-gpu-telemetry
--max-context-length 262144 --tokenizer-trust-remote-code --unsafe-override`.

## Dependency paths outside this kit

| what | absolute path | note |
|---|---|---|
| model + tokenizer | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | shared VAST (`10.2.123.177:/aac-8634674/...`), 501 T, mounted on jump host and both nodes |
| workspace on jump host | `/root/agentx_20260803/` | scripts, corpus, rescued artifacts |
| par8 deployment kit | `../par8.glm52.dpaoff.packup_20260803/` | how the legs were brought up |

## Secrets

**None required.** Both source repos are public; the router is unauthenticated on
the internal network; no registry login is used (`python:3.13-slim` is
anonymously pullable). The only credential involved is the operator's existing
SSH access to the jump host.
