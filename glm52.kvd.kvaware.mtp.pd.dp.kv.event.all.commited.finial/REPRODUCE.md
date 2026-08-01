# Reproduction kit

From a clean pair of nodes to the result: build the branch's image on both nodes
and run G0 / G1 / G2 + the two stress gates against it, with **no in-container
patching**.

**Estimated time ~2 h 15 m**, of which ~30 min is the two image builds (parallel)
and ~50 min is engine cold start (4× at 3–9 min each). Cold start is **not a
hang** — see `notes.md` §2.

## 0. Prerequisites

**Machines.** Two nodes, 8× MI355X (gfx950), same ionic RoCE fabric:

| role | host used | data-plane IP |
|---|---|---|
| prefill + etcd + router + kvd | `chi2879` | 10.2.122.10 |
| decode + kvd | `chi2867` | 10.2.122.44 |

Substitute your own hosts and IPs — they are `MY_IP` / `ETCD_IP` arguments,
never hardcoded inside a leg script.

**Access.** Jump host `root@149.28.124.225`, then `ssh chi2879`. Key-based;
arrange your own. Commands below are written as if run **on the node** unless
they say otherwise. `scripts/J.sh` wraps the two-hop SSH with a retry — the jump
host is loaded and resets connections intermittently, and without the retry a
transient reset reads as a failed step on the node.

**External dependencies** (absolute paths, not in any repo):

- Model: `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` — shared VAST NFS, mounted at
  `/mnt/vast` on both nodes. Its `generation_config.json` is load-bearing; see
  §5 and `notes.md` §1.
- Host libionic: `/usr/lib/x86_64-linux-gnu/libionic.so.1` — must match the
  host's `ionic_rdma` kmod. Bind-mounted; the image entrypoint swaps it in.

**Base image.** `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`, digest
`sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d`.
Must be present on both nodes:

    docker pull lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x

The tag is **pinned in the Dockerfile and must stay pinned** — the DSA diffs
apply at `--fuzz=0` against sglang `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`, so
a base bump fails the build at the patch step rather than mis-applying silently.

**Disk.** ~30 GB free per node for the built image (`docker system df` to check).
Both nodes here sat at 74% of 838 GB with room to spare.

**Secrets.** None for the run itself — the base image is public, and neither etcd
nor the router is authenticated on this cluster. Only cluster SSH. See
`environment.md`.

## 1. Get the branch

    git clone git@github.com:AMD-AGI/Infera.git && cd Infera
    git checkout yihou.dev.glm52.merged.experiment

If the branch is gone, rebuild it from this kit — `branch/patches/` holds all 25
commits as a `git format-patch` series:

    git checkout -b yihou.dev.glm52.merged.experiment 8692fb4
    git am --keep-cr --whitespace=nowarn <KIT>/branch/patches/*.patch

Verified to apply cleanly, all 25. The whitespace flags matter for an *exact*
match: without them `git am` strips trailing whitespace, which alters two lines
of stray SSH-banner text inside a committed env snapshot and yields a different
tree hash. Content is otherwise identical either way, and nothing that affects
the run is involved — but if you are diffing tree hashes, use the flags.

Confirm you are at the right place:

    git log --oneline -1        # -> 330da16 docs: record the built-image validation…
    git rev-list --count main..HEAD    # -> 25

Run the unit suite before spending an hour on a build:

    python3 -m pytest tests/unit --ignore=tests/unit/gaie -q   # -> 1162 passed, 1 skipped

## 2. Stage the source and build on BOTH nodes

Build **on each node** rather than building once and shipping a 28 GB tarball —
the claim being tested is that the Dockerfile reproduces the run, and a tarball
would only prove it survived the trip. From your workstation, in the checkout:

    REF=yihou.dev.glm52.merged.experiment bash <KIT>/scripts/stage_source.sh

`git archive`, not a tar of the worktree: an uncommitted edit that changed the
result would otherwise ride along invisibly.

Then on each node, in parallel (~15 min each):

    bash /tmp/build_merged.sh        # scripts/build_merged.sh, staged there

Each ends with `=== done: sha256:… ===`.

> **The two image ids WILL DIFFER**, and that is expected — each node built
> independently, so the Rust router objects and layer timestamps differ. Ours
> were `1f7cf6964cee` (chi2879) and `0d478433f1b3` (chi2867). **Do not check for
> equal digests.** Check content equivalence, which is the next step.

## 3. Verify the image before running anything against it

A build log saying a patch script printed success is **not** the same as the
interpreter executing patched code — a stale `__pycache__` entry has already
invalidated a full experiment on this stack (`notes.md` §3). On each node:

    IMAGE=infera/engine-sglang:merged bash <KIT>/scripts/verify_built_image.sh

18 assertions against **freshly compiled bytecode**, then a behavioural smoke
test. Every line must read `OK`, ending with:

    === ALL FIXES VERIFIED IN THE BUILT IMAGE ===

The script also prints which infera copies it found. The built image carries
**one** (`/opt/infera/infera`), unlike the patched base image which carried two —
so the shadowing trap in `notes.md` §3 cannot arise here.

## 4. Reset each node and start the G0 legs

**Do not skip the reset.** Carrying GPU state between rounds is the top cause of
wasted hours on this stack (`notes.md` §4). The script tears down the container
and engine processes, *waits for the GPUs to go idle*, starts a fresh container
from the built image, and verifies 8 `PORT_ACTIVE` inside it.

    # chi2879 (prefill; ETCD=1 also starts etcd here)
    ROLE=prefill MY_IP=10.2.122.10 ETCD=1 bash /tmp/reset_merged.sh
    # chi2867 (decode)
    ROLE=decode  MY_IP=10.2.122.44 bash /tmp/reset_merged.sh

Both must end with `===== node <host> ready for <role> (merged image, unpatched) =====`
and must have printed `PORT_ACTIVE: 8` and `kvd socket OK`.

Note step 5 of that script prints `NO PATCH STEP (that is the point)`. That is
the whole difference from the predecessor kit.

Stage the leg launcher and the probes into the containers (both nodes):

    bash /tmp/stage_probes.sh

It copies from **this kit's** `scripts/`, then asserts inside the container that
`needle.py` and `stress_capture.py` are the fixed versions. The predecessor kit's
probes send `temperature: 0`, which manufactures the corruption signature under
MTP and silently invalidates G2 and the stress gate — `notes.md` §1. The assert
is there because that failure is invisible until you read the tails.

Start G0 (MTP off — a command line byte-identical to the kvaware/kvd baseline, so
any change is attributable to the merge alone):

    # chi2879
    ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MTP=0 TAG=g0 bash /tmp/start_leg.sh
    # chi2867
    ROLE=decode  MY_IP=10.2.122.44 ETCD_IP=10.2.122.10 MTP=0 TAG=g0 bash /tmp/start_leg.sh

Wait for both (poll, do not kill):

    IP=10.2.122.10 PORT=30000 bash <KIT>/scripts/wait_ready.sh   # on chi2879
    IP=10.2.122.44 PORT=30000 bash <KIT>/scripts/wait_ready.sh   # on chi2867

This polls the HTTP endpoint rather than grepping the log for `ready to roll` —
the logs are appended to across runs, so a grep matches a *previous* run's line
within seconds (`notes.md` §5).

## 5. G0 — the baseline replay

Router on the prefill node:

    CTR=merged_run bash <KIT>/scripts/start_router.sh    # -> "router healthy"

The router is `python -m infera.server`. It is **not** `infera.router` — that is a
package with no `__main__` and fails with a message that reads like a missing
dependency. The script has it right; use the script.

    docker exec merged_run python3 /tmp/probe.py http://10.2.122.10:8100 glm5.2-mxfp4
    docker exec merged_run python3 /tmp/prefix_reuse.py
    docker exec merged_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock

Expect **4/4**, **16/16 + 16/16**, and `sets_total: 102, gets_total: 0`.
`gets=0` is *correct* here — the in-GPU radix cache is serving.

### The kvd attribution test

This is the one that proves kvd is serving rather than merely wired. Restarting
the prefill engine empties the GPU cache while the kvd daemon and its L3 keep
running, so any reuse afterwards can only have come from L3:

    CTR=merged_run bash <KIT>/scripts/restart_replay.sh     # ~190 s
    docker exec merged_run python3 /tmp/prefix_reuse.py
    docker exec merged_run python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock

**Expect `gets_total: 102, hits_total: 102, sets_total: 102 (unchanged), misses: 0`.**
`sets` staying put is the load-bearing part: reads, not re-writes.

Record the router's KV view as the G1 baseline:

    CTR=merged_run bash <KIT>/scripts/cache_view.sh

Expect prefill **51**, decode 79–90 (traffic-dependent).

## 6. G1 — MTP on the decode leg

    # chi2879 — prefill stays MTP off
    ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MTP=0 TAG=g1 bash /tmp/start_leg.sh
    # chi2867 — decode turns MTP on
    ROLE=decode  MY_IP=10.2.122.44 ETCD_IP=10.2.122.10 MTP=1 TAG=g1 bash /tmp/start_leg.sh

`MTP=1` also implies `--disable-custom-all-reduce` (the aiter custom all-reduce
kernel deadlocks on gfx950 during EAGLE verify).

**This is the gate where the decode leg died twice before patches 6 and 7.** Both
failures hit during *argument parsing* — within seconds, long before a cold start
would finish. Seeing neither is the positive signal:

    grep -m1 "incompatible with speculative decoding" $LOG   # patch 6 missing
    grep -m1 "mutually exclusive"                     $LOG   # patch 7 missing

Wait for ready as in §4, then restart the router so it rediscovers the workers:

    CTR=merged_run bash <KIT>/scripts/start_router.sh

Then **in this order** — the cache-view check is meaningless before traffic:

    docker exec merged_run python3 /tmp/probe.py http://10.2.122.10:8100 glm5.2-mxfp4
    docker exec merged_run python3 /tmp/prefix_reuse.py
    CTR=merged_run bash <KIT>/scripts/cache_view.sh
    grep -ao "accept len: [0-9.]*" $KIT/logs/g1_decode.log | tail -3

Expect 4/4, 32/32, cache-view **prefill 51 / decode 0**, `accept len` **2.1–2.6**.

- **prefill 51** is the bigram discriminator: unfixed it reads 0, and matching
  G0's 51 exactly shows the flattened keys hash identically to plain ints.
- **decode 0** is expected — patch 6 leaves the decode radix cache off under MTP.
- The router view lives in the router *process*; a freshly restarted router reads
  0 for a trivial reason. Drive traffic first (`notes.md` §6).
- `accept len`, not `accept_len` — grepping the wrong one reports MTP absent when
  it is running fine. And `accept len: 4.00` is **bad news**, not good: it means
  the output is a loop the draft model predicts perfectly (`notes.md` §1).

## 7. G2 — a prompt spanning more than one prefill chunk

    docker exec merged_run python3 /tmp/needle.py \
      http://10.2.122.10:8100 glm5.2-mxfp4 24000 0,0.25,0.5,0.75,1.0 \
      /tmp/g2_needle.json 2048

Expect **5/5**, each `finish=stop` with `</think>` exactly once and 70–260
completion tokens.

Then confirm the prompt was *really* chunked — without this the result is
worthless, because chunked prefill quietly not engaging looks identical to a fix:

    grep -ao "new-token: [0-9]*" $KIT/logs/g1_prefill.log | awk -F': ' '$2>1500' | tail -9

Expect triples like `8192, 8192, 1728` — three chunks at the 8192-per-rank
boundary (`chunked-prefill-size 65536` ÷ `dp-size 8`).

**Keep `max_tokens` at 2048 and leave the probe's sampling alone.** At 256 the
reasoning is cut off mid-thought and the run-on tail mimics corruption exactly;
at `temperature 0` with MTP on, the model loops to the cap and produces the same
false signature. Both have already produced wrong verdicts here — `notes.md` §1
and the predecessor kit's §6a.

## 8. Stress

    docker exec merged_run python3 /tmp/stress_capture.py \
      http://10.2.122.10:8100 glm5.2-mxfp4  16  64 1024 1024 /tmp/stress_c16.json
    docker exec merged_run python3 /tmp/stress_capture.py \
      http://10.2.122.10:8100 glm5.2-mxfp4 128 256 1024 2048 /tmp/stress_c128.json

Expect **64/64 CLEAN** and, at conc=128, **≤1 BAD in 256**.

Note the conc=128 OSL is **2048**, not 1024. At 1024 the run showed 5/256 BAD,
*all of them at the cap*; raising the cap alone dropped it to 1/256. Same class of
artefact as the needle probe at 256.

`TAIL_REPEAT` is not a failure: the needle was retrieved and the response stopped
on its own, only the post-`</think>` tail loops. `BAD` = `DIGIT_LOOP +
CORRUPT_REASONING` is the criterion.

**If you get a BAD, replay it at conc=1 before believing it.** Prompt content is a
pure function of `idx+salt`, so the replay is byte-identical to what failed:

    docker exec -e IDX=114,118,123 -e REP=2 merged_run python3 /tmp/stress_capture.py \
      http://10.2.122.10:8100 glm5.2-mxfp4 1 6 1024 2048 /tmp/replay_c1.json

Ours returned **12/12 CLEAN** on the six prompts that failed under load, which is
what identified the failures as cap-related rather than KV-related.

## 9. What "reproduced" means here

All of these, or you have not reproduced it:

| check | expected |
|---|---|
| image verification | 18/18 OK on both nodes |
| G0 correctness / reuse | 4/4, 32/32 |
| **G0 kvd restart-replay** | **gets 102, hits 102, sets unchanged at 102, misses 0** |
| G1 correctness | 4/4 |
| **G1 router view, prefill** | **51** (not merely non-zero) |
| G1 `accept len` | 2.1–2.6 (**not** 4.00) |
| G2 | 5/5, all `finish=stop`, chunking confirmed |
| conc=16 / conc=128 | 64/64 CLEAN / ≤1 BAD |
| `Traceback` on either leg | 0 |

The two bolded rows are the ones that would go red if a fix were absent or wrong.
Everything else can pass for uninteresting reasons.

## If it doesn't reproduce

`notes.md`, in the order the traps are likely to bite. The three that cost the
most time on this run:

1. **`temperature: 0` + MTP looks exactly like KV corruption** (§1).
2. **`start_leg.sh` must kill the `sglang.launch_server` child**, or the next leg
   dies with `port_base at N is not available` (§5).
3. **Do not grep an appended log for readiness** — it matches the previous run
   (§5).
