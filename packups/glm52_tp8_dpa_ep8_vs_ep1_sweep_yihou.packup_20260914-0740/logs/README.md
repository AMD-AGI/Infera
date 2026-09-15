# logs/ — intentionally empty

The sweep drivers were launched as `setsid nohup bash <driver>.sh ... > logs/driver_*.log 2>&1 &`,
but `run_decode.sh` tees each point's child output into that point's own `runtime.log` (mirrored to
`console.log`), so the driver's own stdout stayed at **0 bytes** for all three driver logs
(`driver_c128_yihou.log`, `driver_sweep_yihou.log`, `driver_noep_sweep_yihou.log`).

Nothing was lost and nothing was trimmed. The real logs are per point:
- `evidence/points/<point>/runtime.log.gz` — the full run log of each of the 20 measured points
- `evidence/aborted/<name>/runtime.log.gz` + `failure_tail.txt` — the five failed attempts
