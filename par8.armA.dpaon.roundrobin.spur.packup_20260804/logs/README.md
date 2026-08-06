# Logs

All gzipped.

| file | what |
|---|---|
| `armA2_prefill.log.gz` | the DPA-on prefill leg that completed the run |
| `armA2_decode.log.gz` | the DPA-on + MTP decode leg |
| `bench_driver.log.gz` | the Optimus-AgenticBench driver's stdout |
| `build_prefill_010.log.gz` | the prefill node's image build, incl. the DSA anchor + bytecode verifier output |
| `build_decode_081.log.gz` | the decode node's build — **independent**, hence a different image id; `build rc=0`, same 9 markers |
| `attempt1_restart_at_gmu070_truncated.log.gz` | attempt 1's **restart** at GMU 0.70 (13:11:51–13:16:12), cut short when the 12 h allocation expired mid-boot. **Not the crash** — see below |

## ⚠ The `mem-fraction-static 0.80` crash log was LOST

`notes.md` §1 and `README.md` quote attempt 1's abort:

```
HSA_STATUS_ERROR_OUT_OF_RESOURCES … Available Free mem : 52 MB
Fatal Python error: Aborted        (at token usage: 0.05, #running-req: 0)
```

**That log is not in this kit.** `ab_boot.sh` derives each log path from
`<tag>_<role>`, and the GMU-0.70 restart reused the same tag — so relaunching the
leg **truncated and overwrote the crashed run's log in place**.
`attempt1_restart_at_gmu070_truncated.log.gz` is what survived: the restart's boot
at `mem_fraction_static=0.7`, ending abruptly when the wall clock reclaimed the
node.

Verify for yourself that it is not the crash:

```bash
zcat attempt1_restart_at_gmu070_truncated.log.gz | strings | grep -c HSA_STATUS_ERROR
# -> 0
zcat attempt1_restart_at_gmu070_truncated.log.gz | strings \
  | grep -oE 'mem_fraction_static=[0-9.]+' | head -1
# -> mem_fraction_static=0.7        (the crash ran 0.8)
```

**What this costs.** The crash details in `notes.md` §1 — the abort text, the
`token usage: 0.05` reading, and the 4–5 concurrent prefilling ranks — were read
off the live log at the time and are recorded faithfully, but they are **no longer
independently checkable from this kit**. Treat them as first-hand but
unverifiable.

The *conclusion* they support (round-robin on a DPA prefill leg needs a lower
`mem-fraction-static` than kv-aware does) does not rest on them alone:

- this run's working config is **0.70**, and its `avail mem after pool` of
  85.17 GB is in `armA2_prefill.log.gz`;
- `../analysis/routing_distribution.md` shows the 8-way prefill spread — the
  mechanism that drives the activation peak — measured twice, from the router's
  own pick log and from this run's engine log.

**Avoid this on a rerun**: pass a distinct tag per attempt
(`ab_boot.sh prefill armA_try1`, then `armA_try2`), or copy the log aside before
relaunching.

## Reading the logs

**`strings` first.** These logs contain binary bytes; a bare `grep` reports
"binary file matches" and shows nothing:

```bash
zcat armA2_prefill.log.gz | strings | grep 'Memory pool end'
```

**The driver log needs phrase-grep, not line-grep.** Its progress bar uses `\r`
overwrite, so error prints share a physical line with the bar and `tail` or
line-oriented `grep` appear to show no errors at all:

```bash
zcat bench_driver.log.gz | tr -d '\000' \
  | grep -aoE 'timed out|failed: HTTP [0-9]+' | sort | uniq -c
# -> 16 timed out
```

**Per-rank prefill distribution** — the arm's headline, from the engine side:

```bash
zcat armA2_prefill.log.gz | strings \
  | grep -oE 'DP[0-7] TP[0-7] EP[0-7]\] Prefill batch' | grep -oE 'DP[0-7]' \
  | sort | uniq -c
# -> 475-515 per rank, all 8
```
