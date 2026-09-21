# results/ — the evidence

## What's here

| path | what |
|---|---|
| `t2-summary.csv` | the headline table, all 6 points (t2f 40/56/72/96 + t2e 40/56). Regenerate with `scripts/make_summary.yihou.py <workspace>`. |
| `t2f/agentx_conc{40,56,72,96}.json` | per-point aggregate metrics, t2f arm (custom all-reduce OFF) — the numbers themselves |
| `t2e/agentx_conc{40,56}.json` | the A/B arm (custom all-reduce ON), valid timing data |
| `server-info/{prefill,decode}-0.json` | the resolved `server_args` each engine actually ran |
| `t2f-launch-command-lines.txt` | the emitted `docker run` + engine command lines (proves image, flags, sim-acc env) |
| `{router,worker}-health.json`, `workers.json` | bring-up health snapshots |
| `t2f/aiperf/profile_export.c{N}.jsonl.gz` | per-request records, t2f each point (gzipped) — source of truth for the OSL / `InvalidInferenceResultError` analysis |
| `t2f/aiperf/timeslices.c{N}.json.gz` | per-interval time series, t2f each point (gzipped) |
| `t2f-c128-hang/` | CONC=128 2nd-attempt hang evidence: `HANG-SUMMARY.txt`, gzipped `docker logs` tails (both legs), `rocm-smi`, `df -h /home`, health, client log, fault-scan (0) |

## Reading the numbers

Each `agentx_conc<N>.json` carries `request_metrics` (throughput / latency /
tokens / cache), `server_metrics` (cache-hit + kv_usage), and
`request_accounting` (profiled / warmup-dropped / error-dropped). The
server-measured cache-hit collapse across the sweep (95.2% → 22.3% overall) is
the KV-capacity-saturation signal — see `../notes.md` #3.

## What was deliberately NOT included, and where the originals live

Kept lean because `yihou/` is committed (T1's whole kit was 7.1 MB). This kit is
≈35 MB. Two artifact families are EXCLUDED; originals stay in the (gitignored)
workspace `bench/glm5p2_pd/results/yihou-agentx-hicache/t2f-agentx-c<N>/aiperf_artifacts/`:

- **`server_metrics_export.json`** (1.3 / 1.7 / 2.5 / 5.5 GB per point, ~11 GB) —
  EXCLUDED. Its content is covered by each point's aggregate-JSON `server_metrics`
  block (already summarised in `t2-summary.csv`).
- **`profile_export_aiperf_timeslices.csv`** (up to 116 MB/point) — EXCLUDED as
  redundant with the included `timeslices.c<N>.json.gz` (same data, JSON form).
- **`t2f-launch/server-logs/{decode,prefill}-0.log`** — the FROZEN follower files
  (writer died 23:42 in the NFS event, incomplete/misleading). Not shipped; the
  live `docker logs` tails at the point of the CONC=128 hang are in
  `t2f-c128-hang/` instead. See `../notes.md` #6.

INCLUDED (gzipped, under `t2f/aiperf/`): `profile_export.jsonl` ×4 (the
per-request source of truth for the OSL / error analysis) and the timeslices
`.json` ×4.

The two preserved CONC=128 failure partials — `t2f-agentx-c128-nfsfull/` (NFS
death) and `t2f-agentx-c128-hang-partial/` (hang) — remain in the workspace,
undeleted, as failure evidence.
