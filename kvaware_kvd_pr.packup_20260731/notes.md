# Notes — gotchas, wrong turns, and why each step is shaped this way

Ordered roughly as you would hit them.

---

## 1. The PR branch was 35 commits stale, and it would have broken the image

**What.** `yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` sat on `2df2fed`: 35 commits
behind `origin/main`, zero commits of its own. It was reset to `origin/main`
(`8692fb4`) before anything else.

**Why it matters.** The image that produced the earlier successful runs
(`infera/engine-sglang:pd-unified`) comes from the **unified-Mooncake** rebuild in
`deploy/docker/Dockerfile.sglang`, added by commit `a546137`. That commit is on
`main` but **not** on `2df2fed` — at that base, `Dockerfile.sglang` has no
Mooncake rebuild and `deploy/docker/scripts/build_mooncake_sglang.sh` does not
exist. Building there yields an image whose Mooncake still installs the HIP
intra-node P2P transport unconditionally and prefers it over RDMA, which
**breaks cross-node PD** (`hipIpcOpenMemHandle` cannot open a peer node's handle).

**How to spot it.** Before building, check the base actually has the rebuild:

```bash
grep -c BUILD_MOONCAKE deploy/docker/Dockerfile.sglang        # want >0
ls deploy/docker/scripts/build_mooncake_sglang.sh             # must exist
```

**Context.** The branch had no unique commits, so the reset lost nothing. Always
check `git rev-list --count HEAD..origin/main` on a long-lived feature branch
before you build from it.

---

## 2. A test that passes before *and* after the fix is worthless

**What.** Both fixes ship with tests. Both were checked against the *pre-fix*
code by stashing only the source:

```
3 failed, 48 passed      # source reverted, tests kept
51 passed                # fix restored
```

**Why.** For `0002` in particular the first version of the test passed on the
broken code. The fake `_run` helper returned success for `lsblk` regardless of
its argument, so nothing exercised the actual failure. It had to be taught the
real contract — **real `lsblk` exits non-zero on a bracketed target** — before it
could fail.

**How to apply.** After writing a regression test, revert the fix and watch it
fail. If it doesn't, the test is not testing the bug.

---

## 3. `docker save | gzip` under `nohup &` was silently truncated

**What.** The first image transfer wrote 16.7 GB and stopped. `gzip -t` said
**OK**. `docker load` on the other node said **`unexpected EOF`**.

**Why.** The pipeline was launched as `nohup ... &` through a chain of nested
ssh sessions; when the outer session went away the pipeline died mid-stream. The
gzip *layer* it had written was self-consistent — which is all `gzip -t` checks —
but the TAR inside was cut off.

**How to avoid.**

- Use `docker save -o <file>` rather than a pipe. It writes a `.tmp-*` file and
  renames on success, so a partial result is visibly not the final name.
- Detach with `setsid nohup ... </dev/null &`, not bare `nohup &`.
- Write an explicit status file (`echo rc=$? > save.status`) and check *that*.
- **Verify the tar, not the gzip**: `gunzip -c f.tgz | tar -t >/dev/null`. A
  passing `gzip -t` on a truncated archive is exactly the false confidence that
  cost a 25-minute round trip here.

`docker save` of this image is ~79 GB uncompressed; gzip -1 got it to ~17 GB but
was the slow part. Uncompressed onto shared storage was faster end to end.

---

## 4. The classifier reports `CORRUPT_REASONING` for correct Chinese reasoning

**What.** T2 (conc=32) came back `127 CLEAN / 1 CORRUPT_REASONING`. The flagged
output was completely coherent, found the right needle (`22478`), answered
correctly, and finished with `finish=stop` at 152 tokens. It had simply *reasoned
in Chinese* partway through.

**Why.** `stress_capture.py:salad()` treats `>5` CJK characters as token salad:

```python
or len(re.findall(r"[一-鿿]", t)) > 5
```

That rule was written for a run where CJK only ever appeared as corruption.
GLM-5.2 is bilingual and code-switches legitimately.

**How to apply.** Grade these runs on **whether the needle is present**, not on
the classifier verdict:

```python
sum(1 for r in rows if r["expect"] in r["output"])
```

Re-graded that way: conc=32 is **128/128**, all `finish=stop`.

**Context.** This is the second time in this investigation that a harness reported
near-failure on a model that was visibly producing correct text. When the numbers
say "broken" and the outputs look fine, **suspect the harness first**.

---

## 5. `gets_total = 0` — kvd stored 32 GB and never read a byte

**What.** After the stress tests kvd showed `sets_total=18170`, `gets_total=0`.
Running a shared-prefix reuse workload gave 32/32 correct and *still* `gets=0`.

**Why.** SGLang's in-GPU radix cache serves a repeated prefix without ever
consulting the L3 backend. Everything below the GPU is dead weight as long as the
prefix is still resident.

**How to actually prove kvd serves.** Restart the engine — that empties the GPU
cache, while the kvd daemon and its L3 keep running — then replay the same
prefixes:

| | gets | hits | sets |
|---|---:|---:|---:|
| before restart | 0 | 0 | 18272 |
| after restart | **102** | **102** | **18272** (unchanged) |

Reads climbing with **no new writes** on an empty GPU cache is the only clean
attribution. `scripts/kvaware_restart_replay.sh` does this.

**Context.** In the earlier investigation a reuse phase ran **2.7× faster** with
kvd's counters flat at zero. A latency win is not evidence that kvd did anything.
This is written into `manual/serving/kvaware_kvd_operations.md` for exactly that
reason.

---

## 6. conc=128's `finish=length` tail is the harness, not the stack

**What.** T3 produced 10/512 responses that burned the full 1024-token cap with
`</think>` repeating.

**Why.** These prompts are sent **without a chat template**, and EOS is
suppressed for throughput. A completion that never stops is the expected
consequence. The evidence is unusually clean:

- all 502 good outputs: `finish=stop`, median **149** tokens;
- all 10 bad outputs: `finish=length`, **exactly** 1024 tokens.

The failure is entirely in the stop/EOS decision. The reasoning structure is
intact and there is no KV-corruption signature.

**How to apply.** The conc=128 gate for this stack is **throughput without
errors** — `ERROR=0`, no hangs. Do not count `finish=length` against the KV path.
(Operator ruling, 2026-07-31.)

---

## 7. Ports that collide by default when two workers share a host

Not hit in this run — one worker per node — but the reason the run is *shaped*
that way.

- `--kv-events-bind` defaults to `5557` on every worker.
- `--kv-snapshot-port` defaults to `8801` on every worker.

The snapshot-port collision is the nasty one: the second worker logs
**`ready to roll`** and *then* dies during etcd registration with
`[Errno 98] address already in use`. The leg looks healthy and simply never
appears in the router.

`patches/0001` fixes a third, related case — `free_tcp_port_block` handing the
same base to two workers — but that one is in infera's code, whereas these two are
operator-supplied flags.

**How to spot.** If a leg is "up" but unregistered, grep for `Errno 98`
**after** the ready line.

---

## 8. The image entrypoint replaces the manual libionic dance

**What.** Earlier scripts `docker cp`'d the host's `libionic.so` into the
container by hand. This run bind-mounts it instead:

```
-v $(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1):/host-libionic/libionic.so:ro
```

and the image's `infera-inject-host-ionic` entrypoint does the rest.

**Why it matters.** The container's libionic comes from a different release train
than the host's `ionic_rdma` module. On mismatch libibverbs prints an ABI warning
and `ibv_get_device_list()` returns **zero** devices — Mooncake then silently
falls back to TCP and you get a correct-but-slow run that looks fine.

**How to check.** Immediately after `docker run`:

```bash
docker exec <ctr> bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE'   # want 8
```

Both nodes reported 8 in this run.

---

## 9. Shared-cluster hygiene

`chi2867` was at 96% disk (38 GB free) and could not hold the 78.6 GB image. Both
nodes also run other teams' containers (`primus_train`, `mlperf_gptoss2`, the
`primussafe/*` exporters).

Rules followed here, worth keeping:

- Never `docker system prune` on a shared node.
- Prove a container/image is yours (`docker inspect` → Binds, Env, Created)
  before removing it.
- Space was cleared **by the cluster owner**, not unilaterally.
- Nothing was mounted that wasn't already mounted.

---

## 10. What this run still does **not** establish

Stated so that nothing here reads as more than it is:

- **kvd latency benefit** — not observed. Post-restart replay ran at 0.70 s
  median vs 0.69 s cold. kvd demonstrably *served* 102 blocks; it did not
  demonstrably make anything faster at this scale.
- **The kv-aware overlap weights** (20.0 / 2.0) were *used* but not *validated*.
  One prefill worker and one decode worker means the scorer never had a choice to
  make. The earlier 17/0-vs-round-robin result came from a two-worker setup and
  is not re-demonstrated here.
- **Concurrency vs. locality.** The reuse workload is sequential, so the
  `active_blocks` load term never pushed back against the cache term.
- **True NVMe O_DIRECT L3.** The L3 tier ran on `/tmp/kvd-long`; the fix in
  `patches/0002` is verified by unit test, not by an O_DIRECT probe on real NVMe.
