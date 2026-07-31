# Raw engine logs — the two co-located TP4 decode workers

| File | Node | Worker | GPUs | Port | kv-pub | kv-snap | ready |
|---|---|---|---|---|---|---|---|
| `pd_decodeA.log` | chi2867 (10.2.122.44) | decodeA, TP4, gmu 0.70 | 0–3 | 30000 | 5557 | 8801 | 12:08:29 |
| `pd_decodeB.log` | chi2867 (10.2.122.44) | decodeB, TP4, gmu 0.70 | 4–7 | 32000 | 5657 | 8802 | 12:15:03 |

Copied verbatim off the shared FS (`/mnt/vast/c_huggingface/glm52_kvexp/`).
Run 2026-07-30.

**These two files contain the entire measurement.** The result of this
experiment — where each policy sent the work — is counted directly out of them.
That is unusual for this packup and it is why these logs matter more than most.

## The measurement, straight from the logs

```bash
grep -ac 'Decode batch' pd_decodeA.log   # 23   = 17 (arm 1) + 6 (arm 2)
grep -ac 'Decode batch' pd_decodeB.log   #  8   =  0 (arm 1) + 8 (arm 2)
```

The published figures are the **per-arm deltas**, and the two arms are cleanly
separated in time (the router was restarted onto round-robin at ~12:19):

```bash
# ARM 1 — kv-aware, role weights prefill 20.0 / decode 2.0
grep -a 'Decode batch' pd_decodeA.log | grep -cE '2026-07-30 12:1[78]'        # 17
grep -a 'Decode batch' pd_decodeB.log | grep -cE '2026-07-30 12:1[78]'        #  0

# ARM 2 — round-robin
grep -a 'Decode batch' pd_decodeA.log | grep -cE '2026-07-30 12:(19|20|21)'   #  6
grep -a 'Decode batch' pd_decodeB.log | grep -cE '2026-07-30 12:(19|20|21)'   #  8
```

**decodeB ran nothing at all under kv-aware.** Not less — zero. And it was up
and idle at the time: it reached `ready to roll` at 12:15:03, ~2.5 minutes
before arm 1's first request.

Line-level timestamps, if you want to see the shape rather than the count:

```bash
grep -a 'Decode batch' pd_decodeA.log | grep -oE '2026-07-30 [0-9:]+'
grep -a 'Decode batch' pd_decodeB.log | grep -oE '2026-07-30 [0-9:]+'
```

## The two workers really were distinct — all three ports

This is the configuration that makes the experiment possible at all, and the
thing most likely to be got wrong when reproducing:

```bash
grep -aoE 'port=3[0-9]{4}' pd_decodeA.log | sort -u    # port=30000
grep -aoE 'port=3[0-9]{4}' pd_decodeB.log | sort -u    # port=32000

grep -aoE 'events_bind=tcp://[0-9.]+:[0-9]+|snapshot=http://[0-9.]+:[0-9]+' pd_decodeA.log | sort -u
#   events_bind=tcp://0.0.0.0:5557        snapshot=http://10.2.122.44:8801
grep -aoE 'events_bind=tcp://[0-9.]+:[0-9]+|snapshot=http://[0-9.]+:[0-9]+' pd_decodeB.log | sort -u
#   events_bind=tcp://0.0.0.0:5657        snapshot=http://10.2.122.44:8802
```

**And patch 0001 demonstrably held** — this is the run where it actually
mattered, because these two workers share a host:

```bash
grep -aoE '"endpoint": "tcp://\*:[0-9]+"' pd_decodeA.log | sort -u   # tcp://*:25186
grep -aoE '"endpoint": "tcp://\*:[0-9]+"' pd_decodeB.log | sort -u   # tcp://*:13792
```

Two different draws from the randomised scan. Pre-fix both callers got `32764`
deterministically and, with `dp_size=4`, all four per-rank publishers would have
collided — the second worker would have died with `ZMQError: Address already in
use` on `32765`.

**Geometry — TP4 at gmu 0.70, not the TP8 value:**

```bash
grep -aoE 'tp_size=[0-9]+|mem_fraction_static=[0-9.]+|dp_size=[0-9]+' pd_decodeA.log | sort -u
#   dp_size=4   mem_fraction_static=0.7   tp_size=4
```

**Both are legal decode legs** (the auto-appended radix flag), and both wired
kvd on all four of their DP ranks:

```bash
grep -ac 'disaggregation-decode-enable-radix-cache' pd_decodeA.log   # 1
grep -ac 'infera-kvd adapter connected'             pd_decodeA.log   # 4  (TP4 -> 4 ranks)
grep -ac 'infera-kvd adapter connected'             pd_decodeB.log   # 4
```

Four, not eight — these are TP4 workers. Eight would be wrong here.

## The failure that is NOT in these files

decodeB's **first** attempt died on the `--kv-snapshot-port` 8801 collision:

```
INFO:__main__:using etcd registration: endpoint=10.2.122.10:2379 ...
OSError: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8801)
sys.exit(STARTUP_FAILURE)
```

`pd_decodeB.log` here is the **successful relaunch on 8802**; the failed
attempt's log was overwritten by it. Confirmed:

```bash
grep -ac 'Errno 98\|address already in use' pd_decodeA.log pd_decodeB.log   # 0 and 0
```

The traceback survives only as an in-session capture, quoted in
`results/step4_role_weights_routing.txt`. That is a genuine gap — the most
instructive failure in this experiment is the one whose log is gone.

Why it matters so much: the leg prints `ready to roll` **and then** dies during
etcd registration. The engine looks healthy and the worker simply never appears
in `/v1/workers`, so you go hunting for a router or etcd problem.
`scripts/run.sh` checks the worker count for exactly this reason.

## What these logs do NOT contain

- **The prefill leg's log.** Not copied here. It confirms 64 completions across
  the two arms (32 each) but lives with a different experiment's files.
- **The router log** (`/tmp/router.log`, container-local, container removed).
  This is the material gap: the workers do not record which policy was active,
  so the **labelling** of arm 1 as kv-aware and arm 2 as round-robin rests on
  the in-session record — specifically the line
  `INFO:__main__:router-policy=kv-aware overlap_weight=1 prefill=20.0 decode=2.0`
  quoted in `results/step4_role_weights_routing.txt`. The *split* (17/0 then
  6/8) is re-derivable from these files; the *names* are not.
- **Per-request routing decisions.** The router does not log placement, which is
  why the measurement is done from the workers' side.
- **kvd counters** — daemon-side, read over its unix socket, not in any log.
- **decodeB's failed first start**, as above.

Also note: both files end with `Attempting to reconnect to 10.2.122.10:8998` at
12:59, i.e. the prefill leg's bootstrap going away at teardown. That is the end
of the session, not part of the measurement.

## Regenerating

```bash
bash ../scripts/run.sh
```

~25 min: prefill TP8 cold start, two TP4 decode cold starts, worker-count check,
then two arms. Writes `results/routing_kvaware.observed.txt` and
`results/routing_roundrobin.observed.txt`, each with the decode-batch counters
before and after, plus a verdict table.

The script sets all three per-worker ports, guards `--mem-fraction-static`
against the TP4 OOM, and **fails loudly if the router does not see 3 workers** —
because with fewer than two decode workers there is no routing decision to
measure and the experiment cannot produce a result.

With `KEEP=1`, afterwards:

```bash
cp /mnt/vast/c_huggingface/glm52_rt10/pd_decode[AB].log ./
docker exec glm52_rt10 cat /tmp/router.log > router.log   # the file this packup lacks
```
