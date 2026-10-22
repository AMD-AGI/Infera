# Decode layouts for PD on 4 GPUs: DP-attention vs DCP (GLM-5.2, ROCm, TP4 MoE, EAGLE 5/1/6)

Code read on `906b0915a9` (python = `ae529c96bf`). **Nothing here is measured** except the numbers
explicitly labelled with their source.

Notation: `B` = running requests on the decode, `T = 6B` tokens per target-verify step (5 steps, 6 draft
tokens). Layouts are named `DPa/DCPd`, with `attn_tp = 4 / a`.

## 1. Support by the code

How the code builds DCP groups (`distributed/parallel_state.py:2711-2733`):
- The groups are contiguous slices of each **TP** group (`[0..3]` for TP4), `dcp_size` wide.
- They are not built from the attention-TP group.
- A DCP group therefore stays inside one DP replica only when `attn_tp_size % dcp_size == 0`. The docs
  require this (`dcp.mdx:184`).
- **No code guard enforces it.** The only assert is for elastic EP (`arg_groups/parallel_hook.py:674`).

| Layout | attn_tp | Supported? | What blocks it / caveats |
|---|---|---|---|
| **DP4/DCP1** (22k baseline) | 1 | yes; this is the 22k run | None. |
| **DP1/DCP4** (ours, pdBe) | 4 | yes; pdBe-r1 passed math and passkey (pdrun, first-hand) | Router affinity must be off (decode DP1). |
| **DP2/DCP2** | 2 | **Structurally consistent, untested.** DCP groups `[0,1]`, `[2,3]` equal the attention-TP groups, so both DCP ranks hold the same DP replica's batch, and idle/padded DP steps are idle on both. | (a) No test or run of DSA DCP with DP-attention exists, and no guard rejects it (research critic: "PP, DP attention: no guard, no test"). (b) PD: decode attn_tp 2 > prefill 1 maps to `target_tp_rank 0`, `required_dst_info_num 2` (`common/conn.py` `_resolve_rank_mapping`, same branch as the tested 1->4 case); the relayout runs per peer with `dst_dcp_rank` 0/1. (c) Router: affinity must be **off**. With it on, `disagg.rs:72-89` pins the decode to the prefill DP rank (0..3) and 503s for ranks 2-3. Off: the decode DP is picked by policy, and the decode body carries `disagg_prefill_dp_rank`. (d) The recipes `dsa_dp_sync` / `dsa_indexer rows` were written for DPA without DCP; their DCP interaction is reviewed only for attn_tp 4. |
| DP4/DCP4 | 1 | **no** | `attn_tp % dcp != 0`. The DCP group `[0..3]` would span four DP replicas with different batches. No launch guard found. What happens at runtime is unverified: the DCP collectives would span ranks with different batches. Do not run. |

## 2. Per decode GPU, per target-verify step: replicated vs split work

Sources:
- `fused_qkv_a_proj_with_mqa` is `ReplicatedLinear`; `q_b_proj` / `kv_b_proj` are column-parallel over
  attn_tp; `o_proj` is row-parallel (`models/deepseek_v2.py:2068-2155`).
- Indexer weights are `ReplicatedLinear` (`dsa_indexer.py:339-363`). The decode indexer is the paged
  path over all of the rank's tokens, never DCP-split (`dsa_indexer.py:2231-2237`).
- DCP attention: Q heads gathered, owned top-k slots only, a2a merge (`forward_mla_rocm.py:772-777, 1018-1031`).

| Work item | DP4/DCP1 | DP2/DCP2 | DP1/DCP4 | Note |
|---|---|---|---|---|
| tokens this GPU's attention block sees | T/4 (own DP batch) | T/2 | T | |
| `fused_qkv_a` (q_a + kv_a latent), replicated over attn_tp | T/4 | T/2 | **T** | grows with dcp |
| q_b / w_kc / w_vc, head-split | T/4 x 64 heads | T/2 x 32 | T x 16 | equal |
| Q all-gather over DCP (target: q-rep inactive, confirmed) | - | T x 16 heads -> 32 | T x 16 -> 64 | DCP only |
| **indexer** (scores the whole context per query token; 21 `full` layers; replicated) | T/4 queries | T/2 | **T** | **grows with dcp**; index-K read per query ≈ context x 132 B per layer |
| sparse attention core: heads x keys | T/4 x 64 x 2048 | T/2 x 64 x 1024 | T x 64 x 512 | equal FLOPs and equal KV bytes read |
| LSE merge (a2a, bf16 out + fp32 lse) | - | 1 a2a over 2 | 1 a2a over 4 | DCP only |
| top-k -> local rows remap (21 per step) | - | yes | yes | DCP only |
| o_proj | T/4 full, no reduce | T/2, reduce over 2 | T, reduce over 4 | equal FLOPs |
| MoE (TP4 or EP4) | all T tokens (DP gather / scatter) | all T | all T | equal |
| DP imbalance | per-DP batches; 22k measured imb 1.41-1.54 (max/mean running per DP, `decode_stats.txt`) | 2 replicas | none (one batch) | DCP removes it |
| KV capacity per replica | 22k: ≈2.76M per DP (derived: tok 978k / usage 0.354, e2gA1) | unmeasured | **9,764,096** (pdBe-r1 log) | |

What follows from the code (directions only):
- **DCP adds per-GPU work that DP-attention does not have:**
  - the indexer and `fused_qkv_a` run on `dcp` times more tokens;
  - plus the Q all-gather, the LSE a2a, and the remap.
- **DCP gains:** no DP imbalance (one batch), and capacity.
- **Attention core, MoE, q_b and o_proj are equal across layouts.**
- **Which effect wins at a given batch is unmeasured.**
  - The indexer term scales with context length.
  - The 22k run averaged about **257K tokens per running request** (derived from measured e2gA1: tok 978k / run 3.81).
  - So the indexer cost per verify step is likely not small. The kernel time is unmeasured; a trace will
    show it, see section 4.
- **Capacity is not binding in the 22k shape at C40.**
  - Measured usage was 0.32-0.35 at about 3.8 running per DP.
  - The PD decode has no radix cache, so KV holds only running requests.
  - Linear extrapolation (derived, not measured): a DPA4 decode reaches 100% at about 10.7 running per DP,
    i.e. about 43 total. This is the regime where DCP's capacity starts to matter.
- **DP2/DCP2 sits between the two** on every replicated term; capacity per replica is unmeasured.

## 3. Measurements that decide it (cheap ones first)

1. **pdBeperf** (DP1/DCP4) vs the 22k e2g decode at the same concurrency, with
   `tools/decode_stats.dcp.yihou.py`: ms/step, run, gen_total, graph%, usage.
2. **One kineto trace of a decode step per layout** (`llm-torch-profiler-analysis`): kernel time for
   - the indexer (paged MQA logits plus top-k),
   - `fused_qkv_a`,
   - the sparse MLA,
   - the a2a / all-gather,
   - MoE.
   This directly checks the "indexer grows with dcp" claim.
3. **C sweep** (e.g. 40/48/56/64) on DP4/DCP1 vs DP1/DCP4. Watch for the concurrency where DPA4 usage
   approaches 1.0 or queues on decode.
4. **DP2/DCP2 smoke**: math and passkey first (untested combination), then the same perf point.

## 4. Decode-log parser

`tools/decode_stats.dcp.yihou.py` generalizes the kit's `decode_stats.yihou.py`:
- it accepts any `[ts DP? PP? ATTN_CP? MOE_DP? TP? EP?]` prefix (the DCP decode logs `TP0 EP0`, no DP);
- it reads `.gz` logs;
- it takes the window from `--bench benchmark.log` or `--start/--end`;
- it counts one TP rank per DP;
- it adds `graph%`, `gen_total`, `prealloc>0%` and `retracted>0%`.

Validated against the kit on e2gA1 (`logs/e2g/decode-0.a1.log.gz` + `prefill-0.log.gz`, same window).
It reproduces lines=7653, ms/step 47.6, run 3.81, gen/DP 293.4, imb 1.427, pf_batches 5024 exactly.

```
python3 tools/decode_stats.dcp.yihou.py --label pdBeperf --bench <run>/benchmark.log \
    <server-logs>/decode-*.log --prefill <server-logs>/prefill-*.log
```
