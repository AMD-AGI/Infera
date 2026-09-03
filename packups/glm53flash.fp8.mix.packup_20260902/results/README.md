# results/ — what each file is, who measured it, and which to trust

| file | what it is | measured by |
|---|---|---|
| `verify_round0.txt` | full nine-block verification, decode CUDA graphs **OFF**, 05:22 UTC | **this packup's operator** |
| `verify_round1.txt` | same battery, decode CUDA graphs **ON**, 05:33 UTC | **this packup's operator** |
| `ipc_probe.md` | HIP IPC across disjoint `HIP_VISIBLE_DEVICES`, incl. cross-container | **this packup's operator** |
| `sweep_f8_by_lead.txt` | raw `bench_serving` output, two arms | **the team lead** — see provenance below |
| `fixlen_f8_by_lead.csv` | the same two points, machine-readable | **the team lead** |

## Reading the verify transcripts

They contain the engine's `server_args` dump, which is **~20 KB on a single
line**. `grep` it; never `cat` it into an agent's context.

Section map: A worker registry · B `/v1/models` · C arithmetic + reasoning split
· D coherence · E **AITER mHC line counts** · F both memory pools · G which
shared-expert-fusion arm ran · H fault scan · I VRAM after load.

Block **E** is the load-bearing one: `4` and `4`. Block **G** must show
`Shared experts fusion optimization enabled.` **present** here — the opposite of
the MXFP4 packup, where its absence is the pass condition.

## The sweep — provenance, and why the attribution is kept explicit

**The two-point sweep was measured by the team lead**, on this deployment, at
11:32-11:33 UTC on 2026-09-02, using the lead's own harness invocation. The lead
also ran the HIP IPC probe against the same container, `docker cp`'d the
artifacts out, and tore both containers down at 11:35:42 UTC.

The bring-up operator (this packup's author) **declined to benchmark** this node
and did not take these numbers. That split is preserved rather than smoothed
over, at the lead's instruction: *"attribute them to me, keep your name off
them."* The reason survives knowing who measured them — the operator did not
sample contention at the arm endpoints and so cannot certify them the way they
certify the correctness result.

**The lead's own caveat, recorded verbatim in substance:** the colleague's cards
read **0.3 GB before and 0 GB after**, and were **not sampled during** either
arm.

Harness config, read off the run's own `benchmark_args`:

```
sglang.bench_serving, backend sglang-oai-chat, ENGINE port 31400 (not the router)
dataset random, seed 42, random_input_len 7400, random_output_len 320,
random_range_ratio 1.0, temperature 1.0, top_p 0.95, apply_chat_template False,
num_prompts = 10 x conc, cache_report True
engine: TP4, decode CUDA graphs ON, mem-fraction-static 0.60
```

| conc | FP8 output tok/s | total tok/s | mean TTFT ms | mean TPOT ms | cache hit | successful | MXFP4 ref | **MXFP4 is faster by** |
|---|---|---|---|---|---|---|---|---|
| 1 | 99.70 | 2405.32 | 273.91 | 9.20 | 9.9 % | 10/10 | 111.0 | **1.11×** |
| 8 | 456.68 | 11017.34 | 1296.37 | 13.50 | 12.4 % | 80/80 | 561.0 | **1.23×** |

MXFP4 reference measured by the lead on **n01-33**, same shape, same harness,
same TP4, graphs on.

**Ratio direction is fixed throughout this packup: MXFP4 over FP8** — i.e. *what
the quantization buys*. The inverse (FP8 at 0.90× / 0.81× of MXFP4) is the same
arithmetic; the packup deliberately does not mix the two framings.

This is the first direct FP8-vs-MXFP4 comparison on this architecture at a fixed
topology.

### Four limits to carry with the numbers

1. **Contention was not sampled during either arm.** On a node whose neighbour
   cycles 0 → ~30 GiB every 7-15 minutes (`../notes.md` §5), that is the
   measurement that decides whether the absolutes mean anything. Before/after
   readings were clean; the interior was not observed.
2. **Different nodes.** FP8 here is n05-29 (shared); the MXFP4 reference is
   n01-33. The comparison crosses hosts as well as quantizations.
3. **Single run per point, no repeats** — no variance estimate.
4. **The cache-hit column is confounded by construction.** `--dataset-name
   random` draws from ShareGPT under a fixed seed, and `num_prompts = 10 × conc`
   makes each arm's prompt list a strict prefix of the next, so the hit rate
   rises with concurrency by construction — 9.9 % → 12.4 % here, against a
   predicted 12.5 % at conc 8. Part of any apparent "scaling with concurrency"
   is that rising hit rate, not throughput scaling.

Also: EAGLE verify on ROCm takes an `argmax` branch, so `temperature 1.0 /
top_p 0.95` is effectively ignored during verification. **Label the absolutes
greedy-verified, not temperature 1.0.**

**Both sides of the comparison share the cache confound and the greedy-verify
contamination identically, so the ratio is more defensible than either
absolute.** Use 1.11× / 1.23× as the indication; treat the absolutes as
provisional pending an uncontended node with repeats.
