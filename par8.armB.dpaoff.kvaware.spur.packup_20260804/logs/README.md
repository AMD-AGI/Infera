# Logs

All gzipped. Total ~750 KB.

| file | raw | what |
|---|---|---|
| `armB_prefill.log.gz` | 24 MB | the DPA-off prefill leg, boot through end of run |
| `armB_decode.log.gz` | 5.8 MB | the DPA-on + MTP decode leg |
| `bench_driver.log.gz` | 392 KB | the Optimus-AgenticBench driver's stdout |
| `build_prefill_250.log.gz` | — | the image build, including the DSA anchor + bytecode verifier output |

## Reading them

**`strings` first.** These logs contain binary bytes; a bare `grep` reports
"binary file matches" and shows nothing:

```bash
zcat armB_prefill.log.gz | strings | grep 'Memory pool end'
```

**The driver log needs phrase-grep, not line-grep.** Its progress bar uses `\r`
overwrite, so error prints land on the *same physical line* as the bar and
`tail`/line-oriented `grep` appear to show no errors:

```bash
zcat bench_driver.log.gz | tr -d '\000' | grep -aoE 'timed out|failed: HTTP [0-9]+' | sort | uniq -c
# -> 25 timed out
```
