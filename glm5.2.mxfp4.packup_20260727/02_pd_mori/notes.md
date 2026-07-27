# Notes — 02 PD mori

## The two bugs that blocked this (both were router-flag mistakes, NOT the model/transport)

The GLM-5.2 NSA KV transfer over MoRI RDMA actually worked on the first try — the prefill warmup
completed `status 200`. What blocked a working request was the **router**:

1. **`--kv-events off` is an ENGINE flag, not a router flag.** Passing it to `infera.server`
   → `error: unrecognized arguments: --kv-events off` → router `Exited(2)`.
   **Fix:** drop it from the router launch (the router discovers workers via etcd).

2. **The router must use `--request-transport http`, NOT `nats`.** With `--request-transport nats`,
   the router's NATS request path needs JetStream, but the `nats:2.10` broker is started without
   `-js` → `NoRespondersError` → the router cannot dispatch. The prefill would prefill the request
   (`#inflight-req:1`) but the decode never got tapped, so the client hung.
   **Fix:** router = `--request-transport http --router-policy kv-aware`. The **engines** still use
   `--request-transport nats` (that's fine — they register/serve over per-instance NATS subjects);
   only the router talks HTTP to clients. (This matches the reference program's `launch_pd_router.sh`.)

**Symptom → cause map:** client hangs on first request, prefill log shows `#inflight-req:1
#bootstrap-req:0`, decode idle → router isn't dispatching → check the two flags above.

## GID index — why we use `MORI_IB_GID_INDEX=1` (and when you'd need -1)

On chi2878 + chi2879, **all 8 ionic NICs carry the routable RoCE-v2 GID at index 1** (idx2 empty).
So infera's default `MORI_IB_GID_INDEX=1` is correct here. (The reference program needed
`MORI_IB_GID_INDEX=-1` — MoRI per-NIC auto-select — only because a *different* node, chi2866, has
the routable GID at alternating indices 2/1 across its NICs, where forcing `1` yields ENODATA on
half the NICs → QP RTR crash → TP collapse code=-9.) **If you reproduce on other nodes, check the
per-NIC GID layout first** (`cat /sys/class/infiniband/ionic_*/ports/1/gids/{1,2}`); if it varies
per NIC, set `MORI_IB_GID_INDEX=-1` in `engine.sh`.

## Other required pieces (each cost real debugging in prior work — kept from the reference)

- **libionic mount** — the image ships an older libionic than the host ionic driver; libibverbs then
  enumerates 0 RDMA devices and MoRI silently falls back / sees no fabric. `engine.sh` mounts the
  host `libionic-rdmav34.so` (and `libionic.so.1`) over the image's. Verify 8 active NICs enumerate.
- **NATS broker per node** — the infera engine blocks connecting to `nats://<self>:4222` before it
  registers, even with `--kv-events off`. `up.sh` starts a NATS broker on *each* engine node.
- **NATS JetStream traceback in the engine log is BENIGN.** You'll see `nats.js.errors.
  ServiceUnavailableError` / `NoRespondersError` from `kv_view_store` during engine startup — the
  engine catches it and falls through to etcd registration. Ignore it. (It stems from the broker
  lacking JetStream; harmless because we don't use the JetStream KV-view here.)
- **decode PD-warmup prints `!!!!` garbage** — that's the internal dummy 4-token warmup, not a real
  request. Real requests are coherent. Not a bug.

## RDMA reset ritual (between runs)

Before every re-run: `docker rm -f` all legs/router/etcd/nats, wait for VRAM to return to idle
(~0.3 GB/card, i.e. ~2.4 GB summed over 8), then relaunch. Skipping the drain causes the next run
to OOM. `up.sh` re-creates fresh containers each time.

## Logs

`logs/prefill.log` + `logs/decode.log` are the full engine logs from the passing run
(2026-07-27 06:04–06:10). Grep for `Disaggregation warmup request completed with status 200`
(transfer works), `registered with etcd` (registration), `Prefill batch` / `Decode batch`.
The verbose `server_args=ServerArgs(...)` dump near the top is sglang's normal config echo.
