# Environment

Everything pinned. Where something floats or was not captured, it is called out.

## Time

| | |
|---|---|
| pre-change evidence captured | 2026-08-04 04:00 UTC |
| `net.py` patch applied to the container | 04:22 UTC |
| chi2835 `sysctl` change | 04:33 UTC |
| prefill leg r3 (CHUNK 131072) ready | 04:36:32 UTC |
| router → round-robin | 04:38 UTC |
| **round-robin c8 profiling** | **04:44:21 – 04:59:21 UTC** |
| **prefill DP6 aborted (HSA OOM)** | **05:04 UTC** — during the round-robin c16 leg |
| round-robin c16 (short window, degraded) | 04:59:45 – 05:15:55 UTC |
| prefill leg r4 (CHUNK 65536) ready | 05:30:55 UTC |
| router → kv-aware pw=2 dw=2 | 05:31:19 UTC |
| decode leg r4d relaunched (re-bootstrap) | 05:42:54 UTC |
| **kv-aware pw=2 c8 profiling** | **05:47:51 – 06:02:51 UTC** |
| kvd after-state captured | 06:07 UTC |

## Hardware

Two MI355X (gfx950) 8-GPU nodes on the vultr cluster, plus the jump host as
load generator.

| | chi2835 (prefill) | chi2879 (decode) | jump host |
|---|---|---|---|
| IP (data plane) | 10.2.122.78 | 10.2.122.10 | 149.28.124.225 |
| kernel | 6.8.0-107-generic | 6.8.0-124-generic | — |
| ROCm | **7.2.0** | 7.2.0 | n/a |
| GPUs | 8× gfx950 (MI355X), 309 GB VRAM/card | 8× gfx950 (MI355X) | not used |
| CPU | AMD EPYC 9575F 64-Core, 256 threads | AMD EPYC 9575F 64-Core, 256 threads | 128 cores |
| RAM | 3,023 GB | 3,023 GB | 3,023 GB |
| **RDMA rails ACTIVE** | **8** (`ionic_0..7`) | **7** (`ionic_5` down) | n/a |
| `ip_local_port_range` | **1024 65535 → 32768 60999** (changed, see below) | 32768 60999 | — |

> **chi2879 runs 7 rails, not 8.** Inherited from par8 and the 20260803 kit;
> the leg launcher enumerates only ACTIVE devices. **Not controlled for**, effect
> on KV-transfer time **unmeasured**.

> **chi2835's `ip_local_port_range` was changed by this run.** It read
> `1024 65535` while all three other cluster nodes read the kernel default
> `32768 60999`. Reset to the default at 04:33 UTC. **Runtime only** — nothing
> under `/etc/sysctl.conf` or `/etc/sysctl.d/` sets it, so a reboot restores the
> old value. Full rationale and the revert command: `env/sysctl_change_chi2835.txt`.
> Why it mattered: `patches/README.md`.

The load generator runs on the **jump host**, not on a GPU node — the benchmark
is pure HTTP against the router, so it does not contend for GPU.

## Software — the deployment under test

| | |
|---|---|
| image | `infera/engine-sglang:merged-e` |
| **image digest** | **`sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a`** |
| repo branch | `yihou.dev.glm52.merged.experiment` |
| **repo commit at pack-up** | **`2a01e3192a2efb71cd440a171a42198c8fc7e131`** |
| sglang | v0.5.15.post1 (in-image) |
| model | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST) |
| router | infera Rust router, `10.2.122.78:8100` |
| etcd | `10.2.122.78:2379` |

> **The repo commit did not build the image.** The engine that ran is whatever
> is baked into `merged-e` (digest above), built during the par8 work. **One
> exception applies to this run:** `infera/common/net.py` was patched *inside
> the running container* (`patches/`), so the live engine is `merged-e` **plus
> that one file**. Reproducing from the image alone will hit the port bug again
> unless the patch is re-applied or the node's `ip_local_port_range` already
> excludes the low range.

### Resolved engine knobs

Full command lines: `env/cmdlines_after.txt` (and `cmdlines_before.txt` for the
20260803 posture, captured before anything was touched).

| | prefill (chi2835) | decode (chi2879) |
|---|---|---|
| `--tp-size` | 8 | 8 |
| `--context-length` | 262144 | 262144 |
| `--dp-size` / `--enable-dp-attention` | **8 / present** | 8 / present |
| `--ep-size` | 8 | 8 |
| `--chunked-prefill-size` **passed** | **65536** | 65536 |
| `chunked_prefill_size` **resolved** | **8192** (`//dp_size`, `server_args.py:4902`) | 8192 |
| `--mem-fraction-static` | **0.70** | 0.85 |
| `max_total_num_tokens` | **2,291,584** | — |
| spec decode | — | `EAGLE`, steps 3, topk 1, 4 draft tokens |
| prefill delayer | **present** (`--enable-prefill-delayer`, 5000 ms) | n/a |
| `--enable-cache-report` | yes | yes |
| hicache / kvd | `--hicache-size 16`, `infera-kvd` | — |
| transfer | mooncake, ionic ×8 | mooncake, ionic ×7 |

The **prefill delayer returns automatically with DPA** —
`glm52_leg.sh:151-156` scopes it to the DPA branch. It was absent in the
20260803 run and present here. **Not controlled for.**

### Router configurations used

| leg | resolved argv (from `ps`, not from the script) |
|---|---|
| round-robin | `--router-policy round-robin` — and **no** `--kv-tokenizer-path`, **no** `--kv-*-overlap-weight`; `launch_rust.py:87` gates those behind `policy == "kv-aware"` |
| kv-aware | `--router-policy kv-aware --kv-tokenizer-path <model> --kv-overlap-weight 1.0 --kv-prefill-overlap-weight 2.0 --kv-decode-overlap-weight 2.0` |

Recorded verbatim in `env/router_cmdline_after.txt`. Note the two invocations
differ by more than one token: the kv-aware-only flags vanish entirely under
round-robin. That is upstream's gating, not a scripting slip.

## Software — the benchmark

Unchanged from the 20260803 kit; the same container image and the same corpus
directory on the jump host were reused.

| | |
|---|---|
| customer kit | ROCm/MAD **PR #173**, commit `158d3c928b8aa2de52de75fb03e0b95d0ef1e410` |
| `replay_caseA.sh` md5 | **`7cde1afc627c7e4868eac0fd13741baa`** — **unmodified**, verified in the run log |
| aiperf | `github.com/SemiAnalysisAI/aiperf` @ `cquil11/aiperf-agentx-v1.0`, commit `6cd93f01e6fa9bf7170454a2bb6134aeb25b064b` |
| aiperf image | `aiperf-agentx:v1.0`, digest `sha256:9ebc838842c76ab3cc2e31420d7bdad261bbe04626ef2b00031e09ea5741f0c7` |
| corpus | frozen `caseA_conformance_corpus.tar.gz` — 200 sessions / 1,778 requests, seed 42 |

## The measurement configuration

| var | value | note |
|---|---|---|
| `URL` | `http://10.2.122.78:8100` | our router |
| `SERVED` | `glm5.2-mxfp4` | our `--served-model-name` |
| `TOK` | `/root/agentx_20260803/bench/tokenizer` | the container mounts only `$HERE` |
| `CONCS` | **`8`** (kv-aware leg) / `8 16` (round-robin leg) | operator decision; c16 dropped after the first leg |
| `DUR` | `900` | scenario-enforced minimum (`inferencex_agentx_mvp.py:33`) |
| `IMG` | `aiperf-agentx:v1.0` | |
| `OUT` | `/root/agentx_rr_20260804/results` | **outside `$HERE`** — the known upstream defect; see `notes.md` |

Scenario-forced and not chosen by us: `ignore_eos`, `streaming`,
`use_end_to_start_delays`, no input truncation,
`cache_bust = FIRST_TURN_PREFIX`.

## Dependency paths outside this kit

| what | absolute path | note |
|---|---|---|
| model + tokenizer | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | shared VAST, mounted on jump host and both nodes |
| node workspace | `/mnt/vast/c_huggingface/bench_20260801/` | leg scripts + engine logs |
| jump-host workspace | `/root/agentx_rr_20260804/` | this run's scripts and rescued artifacts |
| corpus + tokenizer + aiperf image | `/root/agentx_20260803/bench/` | reused from the previous kit, untouched |
| previous kit | `../agentx.caseA.customer.packup_20260803/` | the comparison baseline |

## Secrets

**None required.** Both source repos are public; the router is unauthenticated
on the internal network; no registry login is used. The only credential is the
operator's existing SSH access to the jump host. No secret value appears in any
file in this kit.
