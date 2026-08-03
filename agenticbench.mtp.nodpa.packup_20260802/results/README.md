# Results

**Two measured arms**, plus lat1 as the comparison (in the sibling kit):

| dir | prefill DPA | **global** chunk/step | n | role |
|---|---|---:|---:|---|
| `chunk65536_MAIN/` | **off** | **65,536** | 115 | **the result** — global per-step budget matched to lat1 |
| `chunk8192_ARM/` | off | 8,192 | 175 | the chunk control — run first, retained because it isolates the chunk effect (nil) |

| file (in each dir) | what |
|---|---|
| `summary.json` | the driver's own computation — TTFT/TPOT points, cache, success rate, `phases[]` |
| `metrics.jsonl.gz` | per-window raw samples: `new_ttfts`, `new_prompt_lengths`, `new_generation_lengths`. **The source for every ladder and every fit.** |
| `metadata.json` | flat CLI knobs. **Does not record `random_seed`** — see `../logs/README.md` |
| `*_ladders.json` | `lat1_analyze.py --json` output |
| `*_kvd_before/after.json` | prefill kvd counters bracketing the run |

## Why two arms exist

`--chunked-prefill-size` is a **global** budget and DPA divides it by `dp_size`
(`server_args.py:4902`). lat1 at dp8 requested 65,536 and its `server_args=` reads
8,192 — but that is **per rank**, i.e. 65,536 globally. With DPA off there is no
division, so matching the machine means passing **65,536**.

`chunk8192_ARM` was run before that was read from source, and it therefore ran at
⅛ lat1's per-step budget. Rather than discard it, it is kept: comparing the two
noDPA arms measures the chunk effect in isolation, and it is **nil** (0.98–1.06×
across seven bins). That is what licenses attributing the rest to DPA.

## kvd behaviour, both arms

| | `chunk8192_ARM` | `chunk65536_MAIN` |
|---|---|---|
| `sets` delta | +6,848 | +9,796 |
| `gets` delta | **+0** | +1,424 |
| `misses` | 0 | 0 |

Prefill kvd only fetches on a radix-tree **miss**; at 89 % planned hit with a
single session the in-GPU tree serves nearly everything. Non-zero `gets` on the
second arm reflects a fuller tree by then (the engine had been up ~2.5 h). kvd is
proven correct here, not exercised.

## Reproducing

```bash
/shared_nfs/yihou_agentbench/venv/bin/python3 ../scripts/lat1_analyze.py <run-dir>
```

For the three-arm tables, the self-contained snippet in
`../analysis/nodpa_vs_lat1.md` § "Reproduce both tables above" reads both dirs
here plus the lat1 sibling kit, and needs no arguments.
