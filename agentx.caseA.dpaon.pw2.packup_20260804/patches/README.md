# Patches

Two files changed in the repo, plus one node-level `sysctl` change that is not a
patch but is load-bearing for reproduction. The customer's benchmark code was
**not** modified (`spec/replay_caseA.sh` md5 `7cde1afc627c7e4868eac0fd13741baa`,
verified in the run log).

---

## 1. `net_port_block_low_ephemeral.patch` — `infera/common/net.py`

**What.** `free_tcp_port_block(count)` computed its candidate window as
`[1024, low - count]` where `low` is the first value of
`/proc/sys/net/ipv4/ip_local_port_range`, then called
`random.randint(1024, low - count)`. When `low <= 1024 + count` that interval is
empty and `randint` raises before any bind is attempted:

```
File "/opt/infera/infera/common/net.py", line 107, in free_tcp_port_block
  randomised = [random.randint(1024, highest) for _ in range(_PORT_BLOCK_TRIES)]
ValueError: empty range for randrange() (1024, 1017, -7)
```

The patch adds a fallback: when there is no room below the ephemeral range,
scan the **top** of it instead, and log a warning saying so. It also reads
`high` from the same file rather than assuming.

**Why it was needed.** Only `dp_size > 1` reaches this code —
`worker.py:77` sends `count <= 1` to `free_tcp_port()` instead. So a leg with
`--dp-size 8 --enable-dp-attention` **could not start at all** on chi2835, while
the identical image with DPA off ran fine. Turning prefill DPA on is the whole
point of this run, so this was a hard blocker.

**How applied.** Source edited in the repo, then copied into the *running*
container and the stale bytecode deleted:

```bash
docker cp net.py bench_run:/opt/infera/infera/common/net.py
docker exec bench_run rm -f /opt/infera/infera/common/__pycache__/net*.pyc
```

Then **verified against the compiled bytecode**, not the source — par8's Trap 2
is exactly this failure mode:

```bash
docker exec bench_run bash -c \
  'strings /opt/infera/infera/common/__pycache__/net.cpython-310.pyc | grep -c "no room below it"'
# -> 1, and the .pyc mtime is after the patch
```

**Context — the symptom it cured.** `p8_prefill.log` from 2026-08-02 shows the
same traceback: an earlier attempt to bring up prefill with DPA on died here and
was never diagnosed. This patch retroactively explains it.

**Status: fixes a real bug, worth upstreaming.** It is not specific to this
experiment.

## 2. `net_port_block_tests.patch` — `tests/unit/common/test_net_port_block.py`

Three changes, TDD order (test written first, confirmed failing with the exact
production error, then the fix):

- new `test_block_when_whole_port_space_is_ephemeral` — monkeypatches
  `/proc/sys/net/ipv4/ip_local_port_range` to `1024 65535` and asserts a valid,
  contiguous, bindable, NodePort-avoiding block comes back. **Failed with
  `ValueError: empty range in randrange(1024, 1023)` before the fix.**
- `test_block_sits_below_the_ephemeral_range` now **skips** when the host has no
  room below its ephemeral range, instead of asserting something unsatisfiable.
- the module docstring gains the third bug alongside the two already documented.

`python3 -m pytest tests/unit/common/test_net_port_block.py -q` → **13 passed.**

## 3. NOT a patch — chi2835 `sysctl net.ipv4.ip_local_port_range`

**The patch above is necessary but not sufficient.** With the fallback active
the block is drawn from *inside* the ephemeral range, where the kernel may hand
the same port to any `bind(("",0))` caller in the window between our probe
releasing it and the engine binding it. Measured, live:

```
base 37059 chosen; DP0..DP6 bound 37059..37065; DP7 died on 37066 with
zmq.error.ZMQError: Address already in use (addr='tcp://*:37066')
```

Port pressure at the time was trivial (47 distinct local ports in use), so this
is not congestion — sglang's own rank init grabs ports in a burst.

chi2835 read `1024 65535`; chi2879, chi2867 and chi2872 all read the kernel
default `32768 60999`. **chi2835 was the outlier.** Reset:

```bash
sysctl -w net.ipv4.ip_local_port_range="32768 60999"
```

With the default restored, `free_tcp_port_block` takes its **original** path —
scanning `[1024, 32760]`, a region the kernel never allocates — and the race is
gone by construction, not merely narrowed. The next leg picked base **11271**
and all 8 ranks bound cleanly.

**Scope and revert:** runtime only; nothing under `/etc/sysctl.conf` or
`/etc/sysctl.d/` sets this, so a node reboot restores `1024 65535`. Revert
explicitly with `sysctl -w net.ipv4.ip_local_port_range="1024 65535"`.
Full record: `../env/sysctl_change_chi2835.txt`.

**This is a shared node.** The change is to a global kernel parameter and does
affect other containers on chi2835 — it was made deliberately, to the kernel
default, and is recorded here rather than left implicit.

---

## Not patched

| target | why not |
|---|---|
| the customer's `replay_caseA.sh` | zero-modification rule; md5 verified in the run log |
| the customer's corpus / synthesizer | frozen tarball used as shipped |
| aiperf | installed from the published branch |
| `par8.../scripts/start_router.sh` | **copied** to `start_router_pol.sh` and parameterised there, so par8's reproduction path is untouched. With `POLICY` unset the copy emits a byte-identical command line — verified by diff before use. |
| the sglang engine | not rebuilt; only the one `net.py` file was replaced in the running container |
