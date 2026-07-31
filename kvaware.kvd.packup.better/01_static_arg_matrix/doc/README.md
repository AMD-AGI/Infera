# 01 — sglang arg-compatibility matrix (static, no GPU)

**Ran:** 2026-07-30 · **Cost:** ~2 min, one node, no model load, no GPU
**Verdict:** ✅ PASS — 5 OK / 3 FAIL, exactly as expected

## What this experiment answers

Can kvd (sglang HiCache) and kvaware (KV events) coexist with PD disaggregation
and DP-attention? And if not, what exactly forbids it?

Answered by driving **sglang's own `ServerArgs.from_cli_args`** over 8 flag
combinations. No GPU time — this is the cheapest possible way to map the
constraint surface, and it should be the first thing anyone runs.

## Result

| # | Combination | Verdict |
|---|---|---|
| 1 | mix baseline | OK |
| 2 | mix + hicache | OK |
| 3 | mix + DPA + hicache | OK |
| 4 | PD-prefill + DPA + hicache | OK |
| 5 | PD-decode + DPA + hicache | **FAIL** |
| 6 | #5 + `--disaggregation-decode-enable-radix-cache` | OK |
| 7 | #4 + `--disable-radix-cache` | FAIL (same conflict) |
| 8 | #5 + `--disaggregation-decode-enable-offload-kvcache` | FAIL (same conflict) |

Failure message on 5/7/8:

```
ValueError: The arguments enable-hierarchical-cache and disable-radix-cache are
mutually exclusive and cannot be used at the same time.
```

## The finding that matters

A **decode leg sets `disable_radix_cache=True` by itself** ("KV cache is forced
as chunk cache for decode server"), and sglang forbids hicache alongside it. So
**kvd is illegal on a PD decode leg by default.**

What legalises it is `--disaggregation-decode-enable-radix-cache`, and infera
auto-appends that flag *only when kv-events are enabled* and the transfer
backend is mooncake (`infera/engine/sglang/args.py:251-263`).

> **Therefore: turning kvaware OFF silently disables kvd on the decode leg.**

Direct probe of the mechanism (in `results/arg_matrix_verdicts.txt`):

```
[decode leg, NO hicache]          disable_radix_cache=True
[decode leg + decode-radix flag]  disable_radix_cache=False
[prefill leg, NO hicache]         disable_radix_cache=False
```

## How to reproduce

```bash
bash scripts/run.sh
```

Writes `results/arg_matrix_verdicts.observed.txt`; compare with the committed
`results/arg_matrix_verdicts.txt`. The script uses a uniquely-named throwaway
container and removes it on exit (including on failure).

Prerequisite: the model path must be **mounted** (`-v /mnt/vast:/mnt/vast`) —
`from_cli_args` touches the tokenizer, so an unmounted path fails with
`OSError: Repo id must be in the form 'repo_name'...` rather than a real verdict.

## Gotchas specific to this experiment

- **Don't read `[EXIT]` as a pass.** `argcheck.py` distinguishes `SystemExit`
  (argparse rejected a flag outright) from a `ValueError` raised by sglang's
  semantic validation. Only the latter is the constraint we care about.
- The exact set of OK/FAIL is **version-sensitive**. It was captured against
  sglang 0.5.15.post1; upstream may relax or tighten the rule. If your numbers
  differ, that is a finding, not a broken script.

## Environment (verbatim in every packup so this folder stands alone)

**Cluster access.** Jump host `root@149.28.124.225`, then `ssh <node>`. Key-based,
no password appears in any script here.

```bash
J(){ ssh -o StrictHostKeyChecking=no root@149.28.124.225 \
       "ssh -o StrictHostKeyChecking=no $1 '$2'"; }
```

**Nodes** (8× AMD Instinct MI355X / gfx950 each, 128 threads, 3023 GB RAM):

| Host | Data-plane IP | amdgpu | Kernel |
|---|---|---|---|
| chi2879 | 10.2.122.10 | 6.16.13 | 6.8.0-124-generic |
| chi2867 | 10.2.122.44 | 6.16.13 | 6.8.0-107-generic |

**Fabric:** ionic RoCE v2, 8 rails/node (`ionic_0`…`ionic_7`), all PORT_ACTIVE.
Module `26.03.3.001`, NIC firmware `1.117.5-a-77`, routable GID at **index 1**
(hence `MC_GID_INDEX=1`). chi2879→chi2867 RTT 0.069 ms.

**Image:** `infera/engine-sglang:pd-unified`
sha256 `f8ec2d627392435b7cf4c97e47b93a3b36588bec43864a1758b7c0dc9405bd18`
(sglang 0.5.15.post1, torch 2.9.1+rocm7.2.0, ROCm 7.2.0). A **local build**, not
on a registry — the Infera PR #19 rebuild that makes mooncake cross-node RDMA
work. Distribute with `docker save ... | ssh <dst> docker load`.

**infera repo:** branch `yihou.dev.glm5.2.mxfp4.experiment`, commit `362192e7`.

**Models (absolute paths on the shared VAST NFS mount `/mnt/vast`):**
- GLM-5.2-MXFP4 — `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (408 GB, 282 shards,
  `GlmMoeDsaForCausalLM`, 78 layers, 256 experts)
- Qwen3-1.7B — `/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

**Kit staging dir:** `/mnt/vast/c_huggingface/glm52_kvexp` — must be on the
shared FS so both nodes' containers see the same copy.

**Host `libionic` injection is mandatory.** Without it RDMA silently degrades to
TCP. Verify **inside** the container: `ibv_devinfo | grep -c PORT_ACTIVE` → `8`.

**Secrets:** cluster SSH only (key-based). No registry login needed (local
image). etcd runs unauthenticated on the data-plane IP.

## Traps that bite every experiment here

**1. `docker exec -d $CTR bash -lc '...'` does not persist.** The detached
login-shell form exits and takes the child with it. Symptom: no process, no log
file, no error. Bit us twice (router, kvd daemon). **Always** stage a script
file and run `docker exec -d $CTR bash /the_script.sh`, or use
`docker exec -d $CTR env VAR=... bash /script`.

**2. Nested ssh quoting silently mangles variables.** In
`ssh jump "ssh node '...$f...'"` the OUTER shell expands `$f`. Stage a script
file instead of fighting the quoting.

**3. Cold start is 6-12 min and looks like a hang.** GLM-5.2 loads 408 GB.
Watch the log growing (`wc -l`); don't kill it.

**4. Three kvaware ports collide when two workers share a host** — unrelated
code paths, fixing one does nothing for the others:

| Port | Default | Failure |
|---|---|---|
| sglang `--kv-events-config` block | from `free_tcp_port_block` | deterministic same base → `ZMQError: Address already in use` (this is **patch 0001**) |
| `--kv-events-bind` | `tcp://0.0.0.0:5557` | identical on every leg; 2nd fails to bind |
| `--kv-snapshot-port` | `8801` | **the nastiest** — leg prints `ready to roll`, *then* dies during etcd registration. Looks healthy; worker never appears in `/v1/workers`. |

**5. `--mem-fraction-static` is TP-dependent.** `0.85` suits TP8 (51 GB/GPU of
weights). At **TP4** weights double to 102 GB/GPU and 0.85 OOMs — use `0.70`.

**6. `--hicache-ratio` sizes the host pool off the KV pool.** Default 2.0 on a
small model tried to allocate **355 GB per DP rank**. Use `--hicache-size <GB>`
(absolute) instead.

**7. Never probe a PD leg directly.** `curl` to a leg's own port just hangs — a
PD leg only serves through the pair. Use the router, and use a differential run
(flip one thing, hold the rest) to isolate.

**8. Shared cluster hygiene.** Don't prune images, don't mount other people's
drives, don't `docker rm` a container you can't prove is yours
(`docker inspect` → Binds/Env/Created).
