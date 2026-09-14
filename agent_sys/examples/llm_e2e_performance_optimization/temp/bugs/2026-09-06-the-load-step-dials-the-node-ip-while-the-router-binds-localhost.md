# m2's load step dials the node IP while the router binds localhost

**A successful bring-up, nine of nine hard checks, a written handshake — and the
load step then dialled a host nobody was listening on.** Killed a real chain
(`r8m1a`, run `20260906T122512-fe181c`, 174 cards 4–7) after 45 minutes.

Recorded 2026-09-06 by m1. **This is a second instance of an already-recorded
defect**, in a sibling location: `CLAUDE.md` carries m4's `HS_ENDPOINT` finding
from 2026-09-05, where a kit recorded `http://<node_ip>:<port>` while the router
bound `127.0.0.1`, after a 227-second bring-up that passed `verify: PASS`.

## What the run says, in its own order

```
verification: PASS (9/9 hard checks, 0 warning(s))
[13:09:55] wrote the handshake at /mnt/m2m_nobackup/yihou/e2e_flow_r8m1a/pmoff/deployment.json
[profiling_mode_off] deployment up at http://127.0.0.1:8181 in yihou_r8m1a_sgl_…
[profiling_mode_off] router healthy
[profiling_mode_off] starting AIPerf
[profiling_mode_off] AIPerf failed (rc=1). Last 30 lines:
  ===== preflight =====
    ABORT: no answer from router at http://10.245.152.55:8181
```

**The deployment is at `127.0.0.1:8181`. The preflight dialled
`10.245.152.55:8181`.** Both numbers appear in the same log, eleven lines apart.

## The line

`assets/serve/round.sh:27`

```sh
R="http://${E2E_NODE_IP:?}:${E2E_PORT_ROUTER:?}"
```

**Composed from `E2E_NODE_IP`, not read from the handshake** — and the handshake
had already been written to a path the same script knows, two steps earlier.
`assets/bench/aiperf_replay.sh:55` and `aiperf_synthetic.sh:62` then fail on it,
which is where the `ABORT` comes from; the preflight is correct and is reporting
a URL it was handed.

## Why this is the recorded class, not a new one

`CLAUDE.md`, *"一个只守住复合值里某一个字段的修复"*: the earlier fix stopped
**computing the port** and read it from the handshake instead — *"这个修法对,
现在也仍然对"* — but `HS_PORT` was parsed out of `HS_ENDPOINT` while **the host
in the same value was used verbatim.**

**Here the same split survives one layer out:** the deployment writes a complete,
correct endpoint into `deployment.json`, and the load path rebuilds one from
`E2E_NODE_IP` rather than reading it. **A fix that named one field left its
sibling, and the sibling is the one that decides which host is dialled.**

## What it is not

**Not a deployment failure.** Nine hard checks passed, the worker reported ready
in 0 s, `/get_model_info` answered, the router reported one active worker, and
`/metrics` returned 6069 bytes. **Everything the stage exists to do, it did.**

**Not the empty-materials fault.** This run staged one zone and it was populated;
the failure is upstream of any validator.

**Not a launch-line error.** `--var node_ip=10.245.152.55` was correct and was
resolved from the node. **The value is right; the decision to use it instead of
the handshake is the defect.**

## Cost

45 minutes of a real chain with m1 and m2 both real, on the only free half-node
available. The run died with `output_absent` → `escalated` → *"nothing to push:
the executor is a program body: there is no agent to instruct"* → cut at 900 s,
which is the already-recorded escalation-with-no-receiver path and not a second
defect.

## Not fixed here

`assets/serve/` is m2's. Recorded with the line number, both URLs, and the run id.
**The shape of the fix is the one that already worked once: read the endpoint the
deployment wrote, do not rebuild it.**
