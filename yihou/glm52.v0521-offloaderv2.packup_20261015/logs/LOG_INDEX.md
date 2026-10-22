# Server logs included (user-approved policy of 2026-10-14: gzip if < 4 MB, else excerpt)

All logs here fit as full gzip. Sources: `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/<round>/server-logs/*.log`.

| dir | content | state at packing |
|---|---|---|
| `server/ofs0c` | S0 correctness, prefill + decode | complete |
| `server/ofs0` | S0 perf A1+A2 | complete (stack left running until job end) |
| `server/ofg3c`, `server/ofg2c` | correctness, G3K1 / G2K1 | complete |
| `server/ofg4c` | correctness, G4K1 | snapshot (round running) |
| `server/ofg4c.hang1` | same round, copy taken while investigating the slow PD warm-up | snapshot |
| `server/ofg3` | G3K1 perf bring-up | aborted (266 released) |
| `server/ofg2` | G2K1 perf bring-up | snapshot (in flight) |

Key lines: `grep -a offloader` (one line per offloaded module per rank), `Load weight end`, `max_total_num_tokens`, `Memory pool end`, `Engine startup timings`, `disaggregation warmup`.
