# Patches (engine overlays bind-mounted into the containers; none are in the image)

Mount table: `scripts/harness/mounts.yihou.v2-xferfix2.tsv` (selected by `config.yihou.xferfix2.sh`) plus
`mounts.yihou.offload-attrs.tsv` (prefill only, from `config.yihou.offload.sh`). Full files in `mounts-v0521/`; diffs here.

| file (in `mounts-v0521/`) | mounted at (`/sgl-workspace/sglang/python/sglang/srt/…`) | diff |
|---|---|---|
| `conn.yihou-xferfix2.py` | `disaggregation/mooncake/conn.py` | `0001` (regbase → xferfix, R032/R036 10-10) + `0002` (xferfix → xferfix2, g4tune) |
| `prefill.yihou-nonfinal-event.py` | `disaggregation/prefill.py` | `0003` vs upstream 11b1fb21de |
| `kv_events_publisher.yihou-noblock.py` | `managers/scheduler_components/kv_events_publisher.py` | carried from earlier rounds (see previous kits) |
| `forward_mla_rocm.yihou-noabsorbcfg.py` | `models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` | carried from earlier rounds |
| `offloader.yihou-attrs.py` | `utils/offloader.py` (prefill) | `../../glm52.v0521-offloaderv2-burst-c56c64.packup_20261019/patches/0001-offloader-carry-is_shuffled.diff` |
| `conn.yihou-xferfix.py` | — (base of 0002, for diffing) | |
| `conn.yihou-xfertime.py` | — **not used** in any g4tune round (per-chunk timing variant on regbase, built by `scripts/session/mk_xfertime.yihou.py`) | |

## 0001 / 0002 — conn.py: YIHOU_XFER_LOG and YIHOU_XFER_SHARD

- **What:** `YIHOU_XFER_LOG=1` logs per-shard, per-minute aggregates of the prefill KV sender (`YIHOU-XFER shard=… span=… wait_mean=… calls=… blocks=… GiB=… wall=…`): enqueue→dequeue wait, Mooncake `batch_transfer_sync` calls/blocks/bytes/wall. `YIHOU_XFER_SHARD` selects which of the `SGLANG_DISAGGREGATION_QUEUE_SIZE` (4) transfer worker queues a chunk goes to: `dest` (default, = upstream: sum of destination session ports % 4), `room` (`bootstrap_room % 4`), `mix` (0002: `((room * 0x9E3779B97F4A7C15) mod 2^64 >> 32) % 4`).
- **Why:** Mooncake has no per-transfer timing log (MC_TE_METRIC needs a WITH_METRICS build). With `dest` each prefill DP used one shard (one worker thread, FIFO); in run e2f (10-10) `room % 4` equalled the DP rank, so `room` did not spread either — `mix` hashes the room so a DP's requests spread over 4 workers while each request's chunks stay in one FIFO.
- **How:** `config.yihou.xferfix2.sh` passes `YIHOU_XFER_LOG / YIHOU_XFER_SHARD / YIHOU_NONFINAL_EVENT` in `PREFILL_EXTRA_ENV`. Defaults `dest/orig/LOG=1` = upstream behaviour plus logging (R1).
- **Effect:** R2 (`mix`) vs R1 (`dest`), same node 250: prefill transfer_kv avg 2.16 vs 9.98 s, A2 22,854 vs 21,262. Not adopted by the user (see `../notes.md` §2).

## 0003 — prefill.py: YIHOU_NONFINAL_EVENT

- **What:** chooses the wait_event handed to the transfer worker for a non-final chunk under overlap: `orig` (image behaviour), `launch`, `off`; logs `YIHOU-NONFINAL` once a minute.
- **Why / context:** R036 (10-10) experiment. In g4tune every round ran `orig`, i.e. no behaviour change; the patch is present only because the xferfix2 mount table includes it.
