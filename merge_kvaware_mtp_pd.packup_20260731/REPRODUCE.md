# Reproduction kit

Goal: reproduce G0 / G1 / G2 + the two stress gates from a clean pair of nodes.

**Estimated time: ~2 h**, of which ~50 min is engine cold start (weight load +
CUDA-graph capture, 4× — G0 both legs, the G0 restart, G1 both legs). The cold
start is **not a hang**; see `notes.md` §1.

This reproduces **what was measured**: the base image patched in-container. To
validate the *deliverable* instead, see §7.

## 0. Prerequisites

**Machines.** Two nodes, 8× MI355X (gfx950) each, on the same ionic RoCE fabric:

| role | host used | data-plane IP |
|---|---|---|
| prefill + etcd + router + kvd | `chi2879` | 10.2.122.10 |
| decode + kvd | `chi2867` | 10.2.122.44 |

Substitute your own hosts and IPs throughout — they appear as `MY_IP` /
`ETCD_IP` arguments, never hardcoded inside the leg script.

**Access.** Jump host `root@149.28.124.225`, then `ssh chi2879`. Key-based;
arrange your own. All commands below are written as if run **on the node**.

**External dependencies** (absolute paths, not in any repo):

- Model: `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` — shared VAST NFS, mounted at
  `/mnt/vast` on both nodes.
- Host libionic: `/usr/lib/x86_64-linux-gnu/libionic.so.1` — must match the
  host's `ionic_rdma` kmod. Bind-mounted; the image entrypoint swaps it in.

**Image.** `infera/engine-sglang:kvaware-kvd`, digest
`sha256:c9292019d0c5a36d97fb87e7b2944b3b6382bab2f33410b89ed03103192a0d80`.
Must be present on **both** nodes and be the **same digest** — verify:

    docker image inspect infera/engine-sglang:kvaware-kvd --format '{{.Id}}'

Built from `deploy/docker/Dockerfile.sglang.kvaware-kvd` on branch
`yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` @ `da65cc7`. See `environment.md`.

**Secrets.** None needed for the run itself — the image is local, and neither
etcd nor the router is authenticated on this cluster. Only cluster SSH.

## 1. Stage this kit onto the shared mount

From your workstation, with `<KIT>` = this packup directory:

    tar czf /tmp/merge_kit.tgz -C <KIT> scripts patches
    scp /tmp/merge_kit.tgz root@149.28.124.225:/tmp/
    ssh root@149.28.124.225 'scp /tmp/merge_kit.tgz chi2879:/tmp/'
    ssh root@149.28.124.225 'ssh chi2879 "mkdir -p /mnt/vast/c_huggingface/merge_20260731 \
        && tar xzf /tmp/merge_kit.tgz -C /mnt/vast/c_huggingface/merge_20260731"'

`/mnt/vast` is shared, so `chi2867` sees it too. Adjust the path if your cluster
mounts the shared FS elsewhere — then update `KIT=` at the top of
`scripts/node_reset_and_patch.sh` and `scripts/start_leg.sh` to match.

> Throughout this file, paths under `/mnt/vast/c_huggingface/merge_20260731/logs/`
> are logs **your run will create on the node**. The `logs/` directory *inside
> this packup* holds the gzipped logs from the original run, for comparison.

## 2. Reset each node and apply the patch set

**Do not skip the reset.** Carrying GPU state between rounds is the top cause of
wasted hours on this stack (`notes.md` §2). The script tears down the old
container and engine processes, *waits for the GPUs to go idle*, starts a fresh
container, verifies 8 `PORT_ACTIVE` inside it, and only then patches.

    # on chi2879 (prefill; ETCD=1 also starts etcd here)
    ROLE=prefill MY_IP=10.2.122.10 ETCD=1 \
      bash /mnt/vast/c_huggingface/merge_20260731/scripts/node_reset_and_patch.sh

    # on chi2867 (decode)
    ROLE=decode MY_IP=10.2.122.44 \
      bash /mnt/vast/c_huggingface/merge_20260731/scripts/node_reset_and_patch.sh

Both must end with `===== node <host> ready for <role> =====`. The patch step
prints a verification block; **every line must show the expected count**:

    PREREQ nextn eh_proj      -> src=1        patch2a max_seqlen_k -> src=1
    dsa_indexer :: _p1v2_trim                       pyc=1
    dsa_backend :: _glm52_match_page_table_rows     pyc=1
    dp_attn / eagle_worker_v2 / eagle_draft_cuda_graph_runner /
      forward_batch_info / schedule_batch / decode  pyc=1..2
    common/utils.py :: wait_event                   pyc=1
    mooncake/conn.py :: _early_send_wait_event      pyc=1   + synchronize() src=1
    prefill.py :: _early_send_wait_event            pyc=1
    /opt/infera/infera/...client.py :: _flat_tokens pyc=2
    /opt/infera/infera/...events.py                 src=1
    /opt/infera/infera/...args.py spec gate         src=1
    /opt/infera/infera/...kvd_wiring.py decode skip src=2
    _flat_tokens smoke                              OK
    === all merged patches verified ===

These are **bytecode** checks, not source greps, for a reason — see `notes.md` §3.

## 3. G0 — the baseline replay (MTP off)

`MTP=0` produces a command line byte-identical to the kvaware/kvd baseline, so
any change here is attributable to the patch set alone.

    # chi2879
    ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MTP=0 TAG=g0 \
      bash /mnt/vast/c_huggingface/merge_20260731/scripts/start_leg.sh
    # chi2867
    ROLE=decode  MY_IP=10.2.122.44 ETCD_IP=10.2.122.10 MTP=0 TAG=g0 \
      bash /mnt/vast/c_huggingface/merge_20260731/scripts/start_leg.sh

Wait for both (~5–9 min each; poll, do not kill):

    L=/mnt/vast/c_huggingface/merge_20260731/logs
    grep -ac "ready to roll" $L/g0_prefill.log $L/g0_decode.log   # want 1 each
    grep -ac Traceback      $L/g0_prefill.log $L/g0_decode.log    # want 0 each

Start the router **on the prefill node**:

    bash /mnt/vast/c_huggingface/merge_20260731/scripts/start_router.sh   # -> "router healthy"

Correctness, then prefix reuse (which fills kvd's L3):

    docker exec merge_g0 python3 /tmp/probe.py http://10.2.122.10:8100 glm5.2-mxfp4
    docker exec merge_g0 python3 /tmp/prefix_reuse.py
    docker exec merge_g0 python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock

Expect **4/4**, **16/16 + 16/16**, and `sets_total: 102, gets_total: 0`.
`gets=0` is correct at this point — the in-GPU radix cache is serving.

### The kvd attribution test

Restart the prefill engine only. That empties the GPU cache while the kvd daemon
and its L3 keep running, so any reuse afterwards can only come from L3:

    bash /mnt/vast/c_huggingface/merge_20260731/scripts/restart_replay.sh   # ~190 s
    docker exec merge_g0 python3 /tmp/prefix_reuse.py
    docker exec merge_g0 python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock

**Expect `gets_total: 102, hits_total: 102, sets_total: 102 (unchanged), misses: 0`.**
`sets` staying put is the load-bearing part: reads, not re-writes.

Record the router's KV view as the G1 baseline (see `scripts/cache_view.sh`):

    bash /mnt/vast/c_huggingface/merge_20260731/scripts/cache_view.sh

Expect prefill **51**, decode **90**.

## 4. G1 — MTP on the decode leg

    # chi2879 — prefill stays MTP off
    ROLE=prefill MY_IP=10.2.122.10 ETCD_IP=10.2.122.10 MTP=0 TAG=g1 \
      bash /mnt/vast/c_huggingface/merge_20260731/scripts/start_leg.sh
    # chi2867 — decode turns MTP on
    ROLE=decode  MY_IP=10.2.122.44 ETCD_IP=10.2.122.10 MTP=1 TAG=g1 \
      bash /mnt/vast/c_huggingface/merge_20260731/scripts/start_leg.sh

`MTP=1` also implies `--disable-custom-all-reduce` (the aiter custom all-reduce
kernel deadlocks on gfx950 during EAGLE verify). Wait for ready as in §3, then
restart the router so it rediscovers the workers:

    bash /mnt/vast/c_huggingface/merge_20260731/scripts/start_router.sh

Then, **in this order** — the cache-view check is meaningless before traffic:

    docker exec merge_g0 python3 /tmp/probe.py http://10.2.122.10:8100 glm5.2-mxfp4
    docker exec merge_g0 python3 /tmp/prefix_reuse.py
    bash /mnt/vast/c_huggingface/merge_20260731/scripts/cache_view.sh
    grep -ao "accept len: [0-9.]*" /mnt/vast/c_huggingface/merge_20260731/logs/g1_decode.log | tail -3

Expect 4/4, 32/32, cache-view **prefill 51 / decode 0**, `accept len` **> 1**
(measured 2.48–2.58).

- **prefill 51** is the bigram-fix discriminator: unfixed it reads 0. It matching
  G0's 51 exactly also shows the flattened keys hash identically to plain ints.
- **decode 0** is expected — patch 7 leaves kvd off there. See `notes.md` §5.
- The router view lives in the router *process*; a freshly restarted router reads
  0 for a trivial reason. Drive traffic first (`notes.md` §4).
- `accept len`, not `accept_len` — grepping the wrong one reports MTP absent
  when it is running fine.

## 5. G2 — a prompt spanning more than one prefill chunk

    docker exec merge_g0 python3 /tmp/needle.py \
      http://10.2.122.10:8100 glm5.2-mxfp4 24000 0,0.25,0.5,0.75,1.0 \
      /tmp/g2_needle.json 2048

Expect **5/5**, each with `finish=stop` and `</think>` appearing **exactly once**.

Then confirm the prompt was *really* chunked — without this the result is
worthless, because chunked prefill quietly not engaging looks identical to a fix:

    grep -ao "new-token: [0-9]*" \
      /mnt/vast/c_huggingface/merge_20260731/logs/g1_prefill.log | tail -8

Expect triples like `8192, 8192, 1728` — three chunks at the 8192-per-rank
boundary.

**Keep `max_tokens` at 2048.** At 256 the reasoning is cut off mid-thought and
the run-on tail mimics the corruption signature exactly — that produced a false
2/5 on the first attempt (`notes.md` §6).

## 6. Stress

    docker exec merge_g0 python3 /tmp/stress_capture.py \
      http://10.2.122.10:8100 glm5.2-mxfp4  16  64 1024 1024 /tmp/stress_c16.json
    docker exec merge_g0 python3 /tmp/stress_capture.py \
      http://10.2.122.10:8100 glm5.2-mxfp4 128 256 1024 1024 /tmp/stress_c128.json

Expect **64/64 CLEAN** and **250 CLEAN / 6 TAIL_REPEAT / 0 BAD**.

`TAIL_REPEAT` is not a failure: the needle was retrieved and the response stopped
on its own, only the post-`</think>` tail loops. `BAD` = `DIGIT_LOOP +
CORRUPT_REASONING` is the criterion. The classifier in `scripts/` is the
**corrected** one — the original had two defects that between them hid the very
failure mode this gate exists to catch (`notes.md` §6).

## 7. Validating the deliverable image (DONE — this is the preferred path)

§2–§6 above patch a base image in place, which is how the fixes were first
measured. The shipped artifact is the Dockerfile, and it has now been built and
re-validated end to end. Prefer this path.

Build **on each node** rather than building once and shipping a 28 GB tarball —
the claim being tested is that the Dockerfile reproduces the run:

    # from a checkout of the merged branch, on each node
    git archive --format=tar HEAD | gzip > /tmp/merged_src.tgz   # on the workstation
    # ... stage to the node, then:
    tar xzf /tmp/merged_src.tgz -C /root/merged_src && cd /root/merged_src
    docker build -f deploy/docker/Dockerfile.sglang -t infera/engine-sglang:merged .

Takes ~15 min per node with the base already pulled (mooncake rebuild and the
Rust router dominate). The build fails loudly if a patch does not apply.

**Verify the image before running anything.** A build log saying a patch script
printed success is not the same as the interpreter executing patched code — see
`notes.md` §3 for the stale-`.pyc` failure that has already invalidated a full
experiment. `scripts/verify_built_image.sh` greps freshly-compiled bytecode for
identifiers the patches introduce, and finishes with a behavioural smoke test:

    IMAGE=infera/engine-sglang:merged bash scripts/verify_built_image.sh
    # -> 18 checks, then "ALL FIXES VERIFIED IN THE BUILT IMAGE"

Then run §3–§6 unchanged **except**: use `scripts/node_reset_and_patch.sh` with
`IMAGE=infera/engine-sglang:merged` and **skip its patch step** — the image
already carries everything, and anything applied after `docker run` defeats the
point of the exercise.

**Two differences from the patched image, both expected:**

- The two nodes' image ids **differ** (`1f7cf6964cee` vs `0d478433f1b3`). Each
  built independently, so Rust objects and layer timestamps differ. Do not check
  for equal digests; check content equivalence with the script above.
- The built image carries **one** infera copy, not two, so the trap in
  `notes.md` §3 cannot arise there.

Results: `results/raw/*.builtimage.json`. Every gate matched the patched-image
run — see the table in `README.md`.

## If it doesn't reproduce

`notes.md` covers, in order of how likely they are to bite: cold start mistaken
for a hang, the reset ritual, the two-infera-copies trap, the router-view timing
trap, the decode-leg kvd finding, the **three** probe defects, and what is
deferred.

Two failure modes that cost the most time on the built-image re-run, both in the
tooling rather than the system:

- **`temperature: 0` + MTP looks exactly like KV corruption** (`notes.md` §6c).
  Always send the model's own `generation_config.json` values — GLM-5.2:
  temperature 1.0, top_p 0.95, top_k 40. `accept len: 4.00` is a *symptom of the
  loop*, not evidence MTP is healthy.
- **`start_leg.sh` leaves the `sglang.launch_server` child alive.** It pkills
  `infera.engine.sglang` only; the child keeps the DP kv-event port block bound
  and the next leg dies with `port_base at N is not available`. Kill the tree and
  wait for the ports before relaunching.
