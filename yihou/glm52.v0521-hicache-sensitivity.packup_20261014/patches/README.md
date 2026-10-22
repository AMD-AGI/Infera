# Patches — runtime source mounts (no new patch in this experiment)

hc50 introduces **no code change**: it is the e2g recipe with `PREFILL_HICACHE_SIZE=50`. Both hc50 and e2g run the
image `infera-sglang:v0521-yihou-1010-mc4340` (v0.5.21 recipe + Mooncake #4340 HIP-IPC sub-allocation relocation fix,
built per the baseline kit) with the three read-only bind mounts listed in
`scripts/harness/mounts.yihou.v2-regbase.tsv`. The files are copied verbatim from
`/shared_nfs/yihou_glm52/v0521dbg_20261010/harness/mounts-v0521/src/`.

| file (mounted over) | what | why (baseline kit patches/README.md:26-28) |
|---|---|---|
| `kv_events_publisher.yihou-noblock.py` -> `managers/scheduler_components/kv_events_publisher.py` | KV-metrics ZMQ PUSH sends NOBLOCK, drop on EAGAIN | R001: blocking send to an unbound endpoint wedged the prefill scheduler |
| `forward_mla_rocm.yihou-noabsorbcfg.py` -> `models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` | `_absorb_bmm_config` -> None | R003: v0521 hand config at M=256 aborts the triton LLVM backend |
| `conn.yihou-regbase.py` -> `disaggregation/mooncake/conn.py` | logs `YIHOU-REGBASE ... interior=` per registered region | diagnostic (all runs `interior=0`); behaviour-neutral. hc50 census: REGBASE interior=0, 'extends past its allocation'=0 on both legs |

How: bind-mounted `:ro` by `bench-harness/engine.sh` on both legs (see the `-v` items in `results/dryrun/*.txt`).

Full what/why/how of each lives in the baseline kit
`/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/patches/`. Diffs against the image files are there (`runtime-mounts/*.diff`).
