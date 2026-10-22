# Log index — files NOT copied into this packup (size > 4 MB or raw server logs)

Raw server logs are 9-38 MB each; aiperf raw exports up to 1.4 GB. Read with `strings <log> | grep -a ...` (they contain
binary progress bytes). All on `/shared_nfs` (read from spur006 or any spur node). Sizes in bytes, captured 2026-10-14.

| bytes | path | what |
|---:|---|---|
| 28,091,013 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d/server-logs/decode-0.a1.log` | decode engine log (A1..A3 of the round) |
| 8,920,213 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d/server-logs/prefill-0.log` | prefill engine log (all benches of the round, incl. YIHOU_EXPERT8/EXPERTN load lines) |
| 32,731,758 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/server-logs/decode-0.a1.log` | decode engine log (A1..A3 of the round) |
| 24,909,986 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/server-logs/prefill-0.log` | prefill engine log (all benches of the round, incl. YIHOU_EXPERT8/EXPERTN load lines) |
| 38,404,692 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/server-logs/decode-0.a1.log` | decode engine log (A1..A3 of the round) |
| 24,866,199 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/server-logs/prefill-0.log` | prefill engine log (all benches of the round, incl. YIHOU_EXPERT8/EXPERTN load lines) |
| 32,732,443 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/server-logs/decode-0.a1.log` | decode engine log (A1..A3 of the round) |
| 21,777,533 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/server-logs/prefill-0.log` | prefill engine log (all benches of the round, incl. YIHOU_EXPERT8/EXPERTN load lines) |
| 32,635,045 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/server-logs/decode-0.a1.log` | decode engine log (A1..A3 of the round) |
| 17,863,921 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/server-logs/prefill-0.log` | prefill engine log (all benches of the round, incl. YIHOU_EXPERT8/EXPERTN load lines) |
| 22,902,994 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/q1/server-logs/decode-0.a1.log` | decode engine log (A1..A3 of the round) |
| 248,992 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/q1/server-logs/prefill-0.log` | prefill engine log (all benches of the round, incl. YIHOU_EXPERT8/EXPERTN load lines) |
| 27,386,099 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d/agentx-c48-A1/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 121,622,478 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 51,186,725 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,419,487,060 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d/agentx-c48-A1/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 33,008,912 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/agentx-c48-A1/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 137,235,350 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 57,798,907 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,436,774,527 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/agentx-c48-A1/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 33,435,791 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/agentx-c48-A2/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 140,151,478 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/agentx-c48-A2/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 59,001,886 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/agentx-c48-A2/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,418,116,232 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8p/agentx-c48-A2/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 29,326,777 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c48-A1/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 127,055,082 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 53,493,261 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,427,442,785 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c48-A1/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 30,506,242 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c48-A2/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 130,625,113 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c48-A2/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 55,014,577 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c48-A2/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,458,181,021 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c48-A2/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 30,606,431 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c52-A3/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 128,614,904 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c52-A3/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 54,187,856 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c52-A3/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,470,228,725 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8d2/agentx-c52-A3/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 28,523,329 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/agentx-c48-A1/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 126,336,564 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 53,176,995 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,459,757,574 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/agentx-c48-A1/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 29,209,875 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/agentx-c48-A2/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 128,761,662 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/agentx-c48-A2/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 54,213,616 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/agentx-c48-A2/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,484,592,760 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x8pc/agentx-c48-A2/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 31,798,076 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/agentx-c48-A1/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 137,216,821 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 57,741,635 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/agentx-c48-A1/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,435,809,262 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/agentx-c48-A1/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 32,872,738 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/agentx-c48-A2/aiperf_artifacts/profile_export.jsonl` | aiperf per-request records (use to chase InvalidInferenceResultError) |
| 140,299,079 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/agentx-c48-A2/aiperf_artifacts/profile_export_aiperf_timeslices.csv` | aiperf per-timeslice export |
| 59,043,302 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/agentx-c48-A2/aiperf_artifacts/profile_export_aiperf_timeslices.json` | aiperf per-timeslice export |
| 1,458,530,270 | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/x128p/agentx-c48-A2/aiperf_artifacts/server_metrics_export.json` | aiperf raw server metrics |
| 11,216,358 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x128p/metrics.A1.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 11,454,057 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x128p/metrics.A2.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 11,266,407 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x8d/metrics.A1.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 1,582,163 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x8d/metrics.A2.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 11,261,515 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x8d2/metrics.A1.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 11,361,737 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x8d2/metrics.A2.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 11,372,438 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x8d2/metrics.A3.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 11,146,153 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x8p/metrics.A1.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 10,437,385 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x8p/metrics.A2.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 11,260,821 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x8pc/metrics.A1.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 11,358,274 | `/shared_nfs/yihou_glm52/v0521cap2_20261013/runs/x8pc/metrics.A2.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |
| 11,793,668 | `/shared_nfs/yihou_glm52/v0521cap_20261012/runs/q1/metrics.A3c48.tsv` | /metrics sampler TSV (gzipped copy IS in results/<round>/metrics/) |

Notes:
- x8d `decode-0.a1.log` / `prefill-0.log` contain the 08:39:57 `Memory access fault ... ECC failure` lines.
- x8d2/x8pc share node 206 but have separate round dirs; x128p logs were last written 2026-10-14 01:55 (stack left up after the chain).
- q1 `prefill-0.log` is only 248,992 B [not investigated why; q1 ran on 137 with node-local rw root].
- Copied small logs: `results/<round>/{<round>.chain.log,<round>.pipeline.log,<round>.handover.log,engine-*.log}`,
  `results/<round>/agentx-*.driver.log.gz`, per-run `benchmark.log.gz`/`runner.log.gz`; node prep logs in this dir.
