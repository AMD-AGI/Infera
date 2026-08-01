# Reproduction kit

Goal: reproduce every number in `README.md` — the three patches, the control
that discriminates, and the rebuilt image — from a clean checkout.

**Estimated time: ~40 min**, of which ~15 min per node is the image build
(§5, parallel across nodes). Everything before that is minutes.

No GPU work and no model are needed for §1–§4 and §6. The only step that touches
the model path is §4b, which is the run that **proved nothing** and is included
only so the wrong turn is reproducible too.

## 0. Prerequisites

**Machines.** `chi2879` and `chi2867`, via jump host `root@149.28.124.225`. Only
§5 needs both; everything else runs on `chi2879` alone.

**Access.** Key-based SSH; arrange your own. See `environment.md` §Secrets.

**Container.** `merged_run` must be running on `chi2879` from
`infera/engine-sglang:merged`. It was up 14 h when this ran. If it is gone:

    docker run -d --name merged_run --entrypoint sleep \
      -v /mnt/vast:/mnt/vast infera/engine-sglang:merged infinity

(Rounds 1–4 need no GPUs, so the full device/RDMA container setup is not
required. §4b does need the live PD pair; skip it — see §4b.)

> **Do not build over the `:merged` tag.** It and its containers are the
> ground-truth reference. `notes.md` §8.

**Checkout.** The branch, at or after `b6819a6`:

    git -C <repo> checkout yihou.dev.glm52.merged.experiment
    git -C <repo> log --oneline -6      # d3c0d6f, fd3540d, eef9bfc + 2 docs

To reproduce from *before* group E instead, start at `b304d31` and apply the
patches. **The flags are load-bearing** — without them git strips trailing
whitespace and you get a different tree:

    git checkout -b replay b304d31
    git am --keep-cr --whitespace=nowarn patches/*.patch
    git rev-parse replay^{tree}        # f77c8afe5c156f3bd98c57a262dcf0ece6c8f094
    git rev-parse eef9bfc^{tree}       # must be identical

Verified: the two tree hashes match exactly.

## 1. Confirm what is actually missing (2 min)

Do not take the gap list on faith; PR #56 may have moved.

    gh pr view 56 --json state,headRefName,url
    git fetch origin llying/glm5p2_fp8_fixes
    git log --oneline FETCH_HEAD -7

Expect 7 commits, `OPEN`, SHAs `6121189 b2150c3 d63e48b 01b0534 0bb23c7
1ebdc7e 0360af5`.

**The Python half of the bigram fix is already on the branch.** This must be
empty — if it is not, `6e6fdb7` and `01b0534` have diverged and the split below
needs rechecking:

    git diff 6e6fdb7 01b0534 -- infera/router/kv_event/client.py \
        infera/router/kv_event/events.py tests/unit/router/test_kv_event_e2e.py

## 2. Stage the payload onto the node

From the checkout:

    tar czf /tmp/liying_rest.tgz \
      infera/common/net.py infera/engine/sglang/worker.py \
      manual/reference/environment.md rust/router/src/kv_event.rs \
      tests/unit/common/test_net_port_block.py tests/engine/sglang/test_ready_timeout.py
    tar czf /tmp/rust_src.tgz rust                      # image deletes rust/ after build
    tar czf /tmp/zmqtest.tgz rust/router/tests/kv_event_zmq.rs
    tar czf /tmp/extra_tests.tgz tests/unit/router/test_kv_event_bigram.py \
                                 tests/unit/router/test_kv_event_e2e.py

    J=root@149.28.124.225
    scp /tmp/{liying_rest,rust_src,zmqtest,extra_tests}.tgz $J:/tmp/
    scp scripts/*.sh $J:/tmp/
    ssh $J 'scp /tmp/{liying_rest,rust_src,zmqtest,extra_tests}.tgz /tmp/*.sh chi2879:/tmp/'

Every script the steps below invoke as `/tmp/<x>.sh` is in `scripts/`; pushing
them all once avoids a silently-missing one surfacing later as a confusing
"No such file".

## 3. Patch the live container and test there (~4 min)

    ssh $J 'ssh chi2879 "nohup bash /tmp/patch_and_test.sh > /tmp/patch_test.log 2>&1 &"'
    # poll; do not hold a long foreground ssh (notes.md §5)
    ssh $J 'ssh chi2879 "tail -c 4000 /tmp/patch_test.log"'

Expect, in order:

    as_u32_any occurrences in kv_event.rs: 2
    net::_reserved_nodeport_range                  pyc=1 OK
    worker::INFERA_SGLANG_READY_TIMEOUT            pyc=1 OK
    nodeport guard + randomisation both hold       OK
    ...
    test kv_event::tests::decodes_sglang_bigram_batch_under_mtp ... ok
    test result: ok. 10 passed; 0 failed
    cargo_test_rc=0
    === PY CHECKS PASSED ===

**Bytecode, not source** — a stale `.pyc` has silently reverted a patch on this
stack before.

`patch_and_test.sh`'s own pytest step reports a **usage error** (`rc=4`) — the
image's `tests/` tree predates the bigram suites, so one path does not exist
there. That is expected; `rt_run.sh` is the pytest step, and it stages them:

    ssh $J 'ssh chi2879 "bash /tmp/rt_run.sh > /tmp/rt_run.log 2>&1; tail -40 /tmp/rt_run.log"'

Expect **22 passed**, and — the point of `-v` — all 5 `test_ready_timeout.py`
cases *listed* as PASSED. Listed, not merely counted: if they are absent from
the list, the image's missing `pytest-asyncio` is silently no-opping them
(`notes.md` §3).

## 4. The control that actually discriminates (~2 min)

This is the load-bearing step. It runs the new real-socket bigram test against
**both** versions of `kv_event.rs`.

    ssh $J 'ssh chi2879 "nohup bash /tmp/rust_control.sh > /tmp/rust_control.log 2>&1 &"'
    sleep 110
    ssh $J 'ssh chi2879 "tail -c 3000 /tmp/rust_control.log"'

Expect **exactly this shape** — a pass in the control half means the test is not
testing anything:

    ############ CONTROL: revert as_u32_any, expect FAILURE ############
      reverted: as_u32_any removed, as_u32_vec back to ints-only
    test subscriber_decodes_bigram_tokens_under_mtp ... FAILED
      left: 0
     right: 2
    control_rc=101

    ############ TREATMENT: restore the fix, expect PASS ############
    test subscriber_decodes_bigram_tokens_under_mtp ... ok
    treatment_rc=0

Whole-crate sweep, for regressions:

    ssh $J 'ssh chi2879 "bash /tmp/full_run.sh > /tmp/full_run.log 2>&1; tail -30 /tmp/full_run.log"'

Expect cargo **56 + 12 + 2 = 70 passed, 0 failed**. The 4 `test_kv_event_e2e.py`
pytest failures there are **pre-existing** (`notes.md` §3); confirm by running
the same suites on a workstation that has `pytest-asyncio`:

    python3 -m pytest tests/unit/common/test_net_port_block.py \
      tests/unit/router/test_kv_event_bigram.py \
      tests/unit/router/test_kv_event_e2e.py -q      # 21 passed

### 4b. The live A/B — optional, and it proves nothing

Included only so the wrong turn in `notes.md` §1 is reproducible. It needs the
live PD pair and the model path. **Skip it unless you want to see the trap.**

    ssh $J 'ssh chi2879 "nohup bash /tmp/rust_ab.sh > /tmp/rust_ab.log 2>&1 &"'
    ssh $J 'ssh chi2879 "bash /tmp/abread.sh"'   # strips ANSI; notes.md §6

Expect the **unpatched** leg to report `cache_hits=51` — which is what exposes
the comparison as vacuous.

## 5. Rebuild the image from the branch, on both nodes (~15 min each, parallel)

    REF=yihou.dev.glm52.merged.experiment bash scripts/stage_source.sh
    scp scripts/build_e.sh scripts/verify_e.sh scripts/_inner_verify_e.sh $J:/tmp/
    ssh $J 'for n in chi2879 chi2867; do
              scp /tmp/build_e.sh /tmp/verify_e.sh /tmp/_inner_verify_e.sh $n:/tmp/; done'
    for n in chi2879 chi2867; do
      ssh $J "ssh $n 'nohup bash /tmp/build_e.sh > /tmp/build_e.log 2>&1 &'"
    done

`stage_source.sh` uses `git archive`, not a tar of the worktree, so an
uncommitted edit cannot ride along invisibly. `build_e.sh` tags
**`infera/engine-sglang:merged-e`** — see `notes.md` §8 for why not `:merged`.

Poll until each prints `=== done: sha256:… ===`. The two ids **will differ**;
that is expected (`environment.md`).

No Dockerfile change is needed: it already copies `infera/` and `rust/`.

## 6. Verify group E in the built image (~1 min per node)

    # both scripts were staged alongside build_e.sh in §5
    for n in chi2879 chi2867; do
      ssh $J "ssh $n 'bash /tmp/verify_e.sh > /tmp/verify_e.log 2>&1; cat /tmp/verify_e.log'"
    done

Expect on **both**:

    fn as_u32_any                                  src=1 OK
    subscriber_decodes_bigram_tokens_under_mtp     src=1 OK
    decodes_sglang_bigram_batch_under_mtp          src=2 OK
    net::_reserved_nodeport_range                  pyc=1 OK
    worker::INFERA_SGLANG_READY_TIMEOUT            pyc=1 OK
    binary runs                                    OK
    rust/ removed by the build, as designed
    nodeport guard + randomisation                 OK
    _wait_ready(timeout=None) -> env-resolved      OK
    === GROUP E VERIFIED IN THE BUILT IMAGE ===

It runs a **throwaway** container from the image, not `merged_run` — whose
filesystem carries §3's in-place patches and would answer for the image.

The Rust half is source-checked, not bytecode-checked, for a reason stated at
the check and in `notes.md` §7.

## 7. Confirm both temporary groups are still droppable

Run the rebases; do not assert them.

    git checkout -b t1 eef9bfc && git rebase --onto c0450a4 eef9bfc t1
    git log --oneline -1            # -> c0450a4
    git checkout -b t2 eef9bfc && git rebase --onto c0450a4 eef9bfc t2 \
      && git rebase --onto 7f2dac8 6e6fdb7 t2
    git log --oneline -4            # D replayed onto 7f2dac8, no conflict
    git checkout yihou.dev.glm52.merged.experiment && git branch -D t1 t2

## If it doesn't reproduce

`notes.md`, in the order things bite. The three that cost time here:

- **§1** a live A/B that cannot fail is not a test — check the *control* leg
  actually failed before believing the treatment leg.
- **§2** `cargo` needs `LIBCLANG_PATH`, and the workstation's cargo 1.75 cannot
  read the v4 lockfile at all.
- **§3** the image has no `pytest-asyncio`, so `@pytest.mark.asyncio` tests
  report **passing without executing**.

And two harness traps that make a step look successful while doing nothing:
**§4** `docker exec` without `-i` discards a heredoc and exits 0, and **§6**
`tracing`'s ANSI escapes defeat a naive `grep cache_hits=`.
