# Notes — what happened during the final run

## 1. Why this run exists

Eight kits preceded this one and every result in them was obtained by patching a
**running container** by hand. That leaves two things unproven:

1. whether the patch set survives being applied at **image build time** — the
   `.pyc` staleness trap means "the source has the fix" and "the runtime has the
   fix" are different claims;
2. whether the **final** patch set works, because two of the four patches were
   reshaped to follow upstream and the two reshapes were validated in *different*
   runs, never together.

This run closes both. The image is built from the committed Dockerfile and
nothing is patched afterwards.

## 2. Choosing each patch's shape

The rule applied was: take upstream's shape wherever our own measurements
support it, keep ours where they do not, and say which is which.

| patch | decision | basis |
|---|---|---|
| 1 | **upstream #32762** | exp1 (2026-07-30) ran it with the full set: 4/4, 32/32 ×2, 64/64. Its assert-before-restore is strictly better than our v1's silent guard, which would have handed back a short tensor on a shape drift. |
| 2a | **ours** | No upstream counterpart found. Per-diff greps of #31683, #32175, #32209 match neither site. |
| 2b | **ours** | Upstream's shape was ported and **failed**: 0/32 across seven runs on three node pairs. See §3. |
| 3 | **ours** | #32175 carries the same fix upstream; ours is one line and already verified. |
| 4 | **upstream #32209** | exp3b (2026-07-30) measured it at 97.1 % draft-graph usage with zero added collectives, versus our v1's 98.4 % with one gloo all-reduce per `draft()` call. |

The result is that three of the five fixes now match an upstream PR's shape, so
this tree converges with upstream instead of diverging from it.

## 3. The one place we deliberately did NOT follow upstream

#32209's *other* half reconciles decode row counts by **trimming q/top-k**;
ours **expands the page table**. Porting the trim form fails reproducibly at
conc=32 — 0/32 across seven runs, three node pairs, and a rebuilt image — with:

```
ValueError: output tensor size must be equal to world_size times input tensor size
  dp_gather_replicate -> _dp_gather -> _dp_gather_via_all_gather
```

Seventeen candidate causes were instrumented and eliminated. **Root cause is not
identified.** Full account and reproducer:
`../glm52.mxfp4.spur.mooncake.packup_20260731_exp3a_32209_patch2b_unresolved/`.

Until that is understood, patch 2b keeps our form. This is a known divergence
from upstream, recorded rather than papered over.

## 4. Traps hit during this run

### 4.1 Backgrounded `docker save` inside `spur exec` dies — again

Known trap, hit anyway. `docker save` of the 28 GB image, launched with `nohup …
&` inside `spur exec`, died at ~670 MB when the exec namespace tore down. Adding
`setsid` and `disown` did not save it either.

**What worked:** build the image independently on each node from the same build
context tarball. With base layers cached that is ~2 minutes per node, far faster
than moving 28 GB through NFS.

### 4.2 Stale server holding the port

The prefill leg died immediately:

```
ValueError: port_base at 30234 is not available in 30 seconds.
```

The previous debug round's server was still running in the old `dbg2` container.
Killed by explicit PID (never a broad `pkill -f`, which can match your own
shell); `rocm-smi` then showed VRAM back to 0 % on all GPUs, confirming a
container recreate was unnecessary.

### 4.3 Router circuit breaker — diagnosed by latency, not by guessing

After restarting the decode leg for the graph-usage measurement, the probe
returned **0/4, all 503, all in ~0.4 s**.

A 0.4 s 503 is the router's circuit breaker still open from before the restart;
a real backend failure takes 12–23 s. Restarting the router on a **fresh port**
(8160 → 8170) returned 4/4 immediately. The latency is the diagnostic — without
reading it, this looks exactly like the deadlock returning.

### 4.4 Binary bytes in logs

`grep -c "Traceback" decode.log` returns **0** on these logs because they contain
binary bytes and grep treats them as binary. That reads as "clean" when it means
"not checked". Every check in this kit uses `strings … | grep` or `grep -a`.

## 5. On the 92 % graph-usage number

exp3b measured 97.1 % on the same patch 4; this run measured 92.0 %. Both are
"the graph is being used"; neither is a target that was aimed at.

The difference is accounted for by the refusal reason, which is logged:
`future_seed_missing` was 3.5–4.5 % per rank here, i.e. the group correctly going
eager together on iterations where a rank's DSA top-k seed had not yet arrived.
That rate depends on arrival timing and load shape, which differ between a
conc=32 run and a conc=128 run. **This is an explanation consistent with the
data, not a measured cause** — no experiment isolated concurrency as the variable.

What matters for correctness is not the percentage but that it is **identical on
all 8 ranks** (184/200 on every rank). A uniform decision is precisely what the
patch exists to produce.

## 6. What is still open

- **Patch 2a has never had a differential control.** Not in this kit, not in any
  of the eight before it. Its evidence is one observation ("after this fix no
  rank appears in `dsa_backend` in a py-spy dump"). `CLAUDE.md` makes 对拍
  mandatory for this bug class; patch 2a does not meet that bar and never has.
  The cheap experiment is: full set, revert only 2a, verify absent in bytecode.
- **#32209's patch 2b failure is unexplained** (§3).
- **Prefill-leg MTP is off.** Every measurement in this series was taken that
  way. It means the two legs register different RDMA buffer counts, and it means
  the guard's fourth term was permanently true rather than genuinely
  rank-divergent — so the rank-split case patch 4 is designed for has never been
  exercised.
- **No performance comparison** against a DPA-only baseline.
