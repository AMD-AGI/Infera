# Spec — what this experiment had to prove, and whether it did

The task was given in conversation (2026-07-31), not as a standing spec file, so
it is transcribed here verbatim-in-substance rather than copied from a document.

## The ask

> 基于我们最后打开"所有开关"并成功运行的实验成果。在我们提 pr 的干净分支
> (branch `yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr`, worktree
> `/home/yihou/dev/git.16-19/infera.glm5.2.mxfp4.offical`) 里：
>
> 1. 产出最终的 patch。
> 2. 产出一个正式的 dockerfile。
> 3. 一些说明：哪些是为了实验目的设计的。正式生产环境中如果要打开所有开关的推荐
>    配置。如何快速验证一些开关是否奏效。哪些开关是可以调配数值的（比如 kv
>    aware），建议是什么。
>
> 最后：使用该 dockerfile build docker image，进行最后一轮正式成功的复现实验。
> 要求正确性，conc=32 conc=128 的压测都通过。然后 packup 最终这个干净可交付
> 复现的实验。

## Success criteria, and the result against each

### D1 — Final patch set

**Bar:** the fixes from the investigation, on the clean PR branch.

**Result: PASS.** 5 commits (`patches/0001…0005`), applied on `origin/main`
(`8692fb4`). Both source fixes carry regression tests that **fail on the pre-fix
code** — verified by reverting only the two source files: 3 failed / 48 passed,
vs 51 passed with the fixes. A fresh clone at `8692fb4` + these 5 patches yields
a tree hashing **identical** to the branch HEAD
(`a16d0dce342be853e0369681f8fae7fde84d6b2a`).

### D2 — A formal Dockerfile

**Bar:** a production Dockerfile, and the final run must be built from it.

**Result: PASS.** `deploy/docker/Dockerfile.sglang.kvaware-kvd` (copied to
`dockerfiles/`). Built as `infera/engine-sglang:kvaware-kvd`,
`sha256:c9292019d0c5a36d97fb87e7b2944b3b6382bab2f33410b89ed03103192a0d80`, and
that digest is what ran on both nodes.

Note it deliberately **layers on** `Dockerfile.sglang` rather than duplicating a
78.6 GB build: the existing image already contains engine + kvd daemon + router +
statctl. What the new file adds is the contract — digest-pinnable base, a
build-time self-check, and the operational defaults. The self-check printed
`kvaware+kvd self-check OK` during the build.

### D3 — The written guidance

**Bar:** experiment-only vs production settings; recommended all-on config; how
to quickly verify a switch works; which knobs are numeric and what to set.

**Result: PASS.** `manual/serving/kvaware_kvd_operations.md` (patch 0005) covers
all four, plus a SGLang section in `manual/features/kv_cache_offload.md` (patch
0003). Every log string and flag default cited was checked against source.

### D4 — Correctness

**Bar:** 正确性.

**Result: PASS — 4/4.** temp=0 factual probe through the router, coherent
output. Evidence: `results/raw/t1_correctness.txt`.

### D5 — conc=32 stress

**Bar:** 压测通过.

**Result: PASS.** 128 requests, ISL/OSL 1024. **128/128 needle correct, 0 HTTP
errors, 128/128 `finish=stop`**, 3.1 req/s.
Evidence: `results/raw/t2_conc32.json`.

### D6 — conc=128 stress

**Bar:** 压测通过. Refined by the operator on 2026-07-31:

> 他这个数字循环是正常的，因为我们没带 chat template 而且为了性能取消了 EOS,
> 所以压测吞吐无错就可以。

So the bar is **throughput without errors** — not zero `finish=length`.

**Result: PASS.** 512 requests. **0 HTTP errors, 0 hangs, 502/512 needle
correct**, 6.8 req/s. The 10 non-clean responses are all `finish=length` at
exactly the 1024-token cap (the run-on/EOS mode described above); the split is
clean — 502 good at `finish=stop` median 149 tokens, 10 bad at exactly 1024.
Evidence: `results/raw/t3_conc128.json`.

### D7 — Packed up

**Bar:** packup 最终这个干净可交付复现的实验.

**Result: this folder.**

## Beyond the bar — kvd proven to serve

Not asked for, but the run would have been misleading without it. During the
stress tests kvd wrote 32 GB and read **nothing** (`gets_total=0`): SGLang's
in-GPU radix cache serves repeated prefixes without touching L3, so neither a
passing test nor a latency win proves kvd did anything.

Restart-and-replay (engine restarted → GPU cache empty; kvd daemon and its L3
survive):

| | gets | hits | sets | misses |
|---|---:|---:|---:|---:|
| before restart | 0 | 0 | 18272 | 0 |
| after restart | **102** | **102** | **18272** (unchanged) | 0 |

Reads with **no new writes** on an empty GPU cache. Evidence:
`results/kvd_reuse_before_restart.log` / `results/kvd_reuse_after_restart.log`.

## Explicitly out of scope

- **854ebf70** (long-context mooncake `wait_event` fix) — lives on
  `origin/llying/dev/glm5p2_fp8_exp`, not on `main`. Operator recorded it as a
  **TODO** for a later roll-up, deliberately not merged here.
- GPU-direct (AIC) verification — dropped earlier by the operator.
