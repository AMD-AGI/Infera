# Notes — the traps, in the order they bite

Three of these cost real time in this run. All three are tooling, not the
system, which is the pattern on this stack: the code was right long before the
harness could show it.

---

## 1. A green test that never had a chance to fail is not evidence

The most important lesson here, and it nearly shipped as a result.

Round 3 ran the shipped `infera-router` and the freshly-built patched one
against the live PD pair, reading policy.rs's per-pick `cache_hits`:

| leg | picks |
|---|---|
| before (**unpatched**) | `0, 0, `**`51`**`, 0` |
| after (patched) | `0, 0, 0, 0` |

**Why it proved nothing:** the live *prefill* leg runs without
`--speculative-algorithm`. MTP is on the **decode** leg, and `is_eagle` only
switches the radix key to bigrams on a leg that has it — so the wire was
carrying plain ints, which both binaries decode correctly. The unpatched binary
reading **51** is the tell: if it can score a full prefix hit, the bug being
tested is not on this path.

(The "after" leg's four zeros are a second-order artifact of the same setup —
each leg starts a fresh router process and the view lives in the process, so the
first leg simply happened to catch a warm repeat. See the predecessor kit's
`notes.md` §6.)

**What to do instead:** if the fix cannot be *removed* and shown to break
something, the run is not a test. Round 4 drives the same decoder over a real
ZMQ socket with the exact bigram shape and runs it **both ways** — unpatched
fails `left: 0, right: 2`, patched passes.

The cheaper alternative — bringing up an MTP *prefill* leg — costs ~9 min cold
start per leg and exercises the identical decoder, so it buys nothing the
socket-level control does not.

---

## 2. `cargo` in the image needs `LIBCLANG_PATH`, and your workstation's cargo is too old

Two separate walls, both hit in the first five minutes.

**Workstation:** `cargo 1.75` cannot even read the lockfile —

    error: failed to parse lock file
    Caused by: lock file version 4 requires `-Znext-lockfile-bump`

So all Rust work happens in the container. There is no local shortcut.

**Container:** an ad-hoc `cargo test` dies in a *build script*, not in our code —

    error: failed to run custom build command for `onig_sys v69.9.3`
      Unable to find libclang ... set the `LIBCLANG_PATH` environment variable

`Dockerfile.sglang` sets it by searching `/opt/rocm*`; nothing sets it for an
interactive shell. Replicate the Dockerfile's own discovery:

```bash
LIBCLANG_PATH="$(dirname "$(find /opt/rocm* /usr/lib /usr/lib64 -name 'libclang.so*' 2>/dev/null | head -1)")"
case "$LIBCLANG_PATH" in ""|".") LIBCLANG_PATH=/opt/rocm/llvm/lib;; esac
export LIBCLANG_PATH   # here: /opt/rocm-7.2.0/lib/llvm/lib
```

---

## 3. The engine image has no `pytest-asyncio`, so async tests silently *pass*

The failure mode is the dangerous direction: not a red test, a **green** one.

Without the plugin, `@pytest.mark.asyncio` is an unknown mark. The coroutine is
collected, never awaited, and reported as passing — a test that asserts nothing.
pytest only hints at it in a warning nobody reads:

    PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?

Consequences, both real here:

- Four `test_kv_event_e2e.py` tests **fail** in the image (they get far enough to
  error rather than no-op). They pass locally, where the plugin exists. This is
  **pre-existing and unrelated** to group E — verified by running the same
  suites locally: 21/21.
- My new `test_ready_timeout.py` was first written with the marker, so in the
  image it would have "passed" without executing. It was rewritten to call
  `asyncio.run` from **sync** test functions, so it actually runs in the image
  it guards. Confirmed by `-v`: all 5 listed as PASSED, none skipped.

**Rule:** a test that guards the engine image must not depend on a plugin the
engine image lacks.

---

## 4. `docker exec` without `-i` throws your heredoc away — and exits 0

    docker exec CTR bash -s <<'EOF'   # ← runs an EMPTY script, rc=0
    ...everything you meant to run...
    EOF

No stdin is attached, so `bash -s` reads EOF immediately. The step reports
success while doing nothing. Combined with a nested `ssh` that also swallows
output, it took two rounds to notice nothing had run.

**Fix, used by every script in `scripts/`:** stage the inner script as a **file**
and run it by path.

    docker cp /tmp/_inner.sh CTR:/tmp/ && docker exec CTR bash /tmp/_inner.sh

Same family as the known nested-ssh quoting trap: when in doubt, ship a file.

---

## 5. Reading logs back through the jump host

The jump host runs at load ~32 with tens of thousands of zombies and resets
connections. Three things that work:

- Run the work with `nohup ... > /tmp/x.log 2>&1 &`, then poll the log in a
  separate short call. A long foreground `ssh` will be cut.
- Read with `tail -c N /tmp/x.log`, not by streaming stdout through two hops.
- Use `scripts/J.sh` (retry wrapper) for anything that must not fail
  spuriously. Note it needs the *caller's* cwd to resolve — invoke it by
  absolute path.

Deep-nested quoting (`ssh 'ssh "docker exec bash -c \"...\""'`) loses output
even when the command succeeds. Stage a script file.

---

## 6. `tracing` colourises log fields, so naive greps miss them

Round 3 first reported "no picks at all" because of this:

    grep -o 'cache_hits=[0-9]*' router.log     # ← finds nothing

The Rust router's `tracing` subscriber writes ANSI escapes *between* the field
name and `=`. Strip them first:

    sed -r 's/\x1B\[[0-9;]*[mK]//g' router.log | grep -o 'cache_hits=[0-9]*'

The picks had been there the whole time. Ten minutes lost chasing a routing
problem that did not exist.

---

## 7. The Rust half cannot be verified from the built artifact

Stated because the rest of this stack's verification discipline is
"check the bytecode, never the source", and here that is not possible.

`as_u32_any` is a small private fn the release profile inlines, and doc comments
are not in the binary; `rust/` is deleted after the build. So `verify_e.sh`:

- greps **the source the build consumed** (`/root/merged_e_src`, byte-for-byte
  what `docker build` read) for `fn as_u32_any` and both new tests, and
- only sanity-checks that the shipped binary exists and runs.

The behavioural evidence is round 4's control, which is stronger than any grep
would have been: revert the fix, the real-socket test fails `0 vs 2`.

The Python half **is** checked in freshly-compiled bytecode, as usual.

---

## 8. Why `:merged-e` and not `:merged`

The build could have overwritten `infera/engine-sglang:merged`. It must not:
that tag, and the `merged_run` containers created from it, are this line of
work's ground truth (CLAUDE.md names them explicitly as the 对拍 reference).
Overwriting it would make the reference unreproducible to save one tag.

Note the live `merged_run` **container filesystems** now carry the group-E
patches from rounds 2–4. The `:merged` **image** does not. Restart the container
from the image for a clean one.

---

## 9. Merging two fixes into one loop, and one test file

`d63e48b` (skip the NodePort window) and our `826619b` (randomise the scan start)
both rewrite `free_tcp_port_block`'s loop, so neither cherry-picks over the
other. The merge keeps `itertools.chain(randomised, exhaustive)` and puts the
reserved-window skip as the **first statement of the loop body**, so the guard
applies to the random probes and the fallback scan alike.

Liying's `tests/unit/common/test_net_ports.py` was folded into our existing
`test_net_port_block.py` rather than added beside it: same function, same
subject, and two files asserting on one loop would only conflict again on the
next change. The merged suite asserts both properties together — a base must now
be both spread out *and* outside the window — which is the property that
actually matters and that neither original file tested.

---

## 10. What is still not covered

- **gfx942**, entirely — three PR #56 commits, deliberately out of scope, and
  unrunnable on this cluster.
- **A live Rust-router kv-aware run under MTP.** Round 4 proves the decoder;
  it does not prove the end-to-end routing decision on a real MTP prefill leg.
  Doing that means a G1 repeat with `--router-backend rust` *and* `PREFILL_MTP=1`.
  The decoder is the only thing the fix touches, so the marginal value is low —
  but the gap is real and should be named rather than implied away.
- **The NodePort fix on an actual Kubernetes cluster.** The guard is tested
  (blocks avoid the window); the *bug* it prevents cannot be reproduced on bare
  metal with no kube-proxy.
