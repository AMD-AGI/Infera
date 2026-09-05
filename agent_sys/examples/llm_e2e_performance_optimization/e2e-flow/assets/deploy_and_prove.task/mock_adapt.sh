#!/usr/bin/env bash
# MOCK-MAP (A) + (I): make the sealed `deploy_kit` conform to what this package
# requires of one. **An adaptation is a step after the copy, not a variant of
# the copy** — `mock.sh` puts the sealed bytes down verbatim, on purpose, and
# this adds only what did not exist when they were sealed.
#
#     mock_adapt.sh <content dir>          # the handoff's content/, holding items/
#
# Two gaps, both real and both the schema doing its job rather than an oversight:
#
#   (A) `codes/environment.yaml` did not exist. No sealed handoff carries the
#       document `environment.schema.json` describes — the *fields* existed as
#       `items/env/deployment.json`, but three the schema requires (`gpu_arch`,
#       `gpu_count`, `image_id`) were never recorded as data.
#   (I) The runtime contract did not exist. The sealed kit spells all six
#       concepts `DK_*`, so the shim below is a **rename**, not new behaviour —
#       each line falls back to the `DK_` name the kit already honours, and the
#       kit's own defaults still apply when neither is set.
#
# **One definition, two callers.** `../check_deploy_kit.validator/gate.sh` builds
# its positive fixture with this same script, so the thing the mock run produces
# and the thing the both-directions gate calls "a conforming kit" cannot drift
# apart. That is the same reason `deploy_kit.layout.yaml` is one file.
#
# `env_render.py --new` writes the record rather than a heredoc here, because it
# is the **real producer's** path (`readme.md` STEPS step 5) and it validates
# before it writes. A mock that wrote the document a different way would be
# testing a different producer.
set -eu

CONTENT="${1:?usage: mock_adapt.sh <content dir>}"
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:-$(cd "$(dirname "$0")/../.." && pwd)}}"

CODES="$CONTENT/items/codes"
[ -d "$CODES" ] || { echo "mock_adapt: no $CODES — was mock.sh run?" >&2; exit 1; }

PACKUP="$(find "$CODES" -maxdepth 1 -type d -name '*.packup_*' | head -1)"
[ -n "$PACKUP" ] || { echo "mock_adapt: no <name>.packup_<YYYYMMDD> under $CODES" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# (A) the environment record.
#
# The three facts a variable cannot carry are **the sealed run's own measured
# values**, read off its `environment.md`: gfx950, eight cards, and the image
# digest it recorded. Inventing plausible ones would make the mock a fiction
# about hardware; these are a real bring-up's numbers, which is the whole
# premise of mocking from sealed artefacts rather than synthesising.
#
# **`runtime.replayed_from` names the sealed kit this stood in for**, and it is
# the field that makes the whole arrangement honest. In mock mode this handoff is
# a *deployment stand-in*: the evidence in `results/` is the sealed run's, and the
# environment record describes **today's node**, because that is the machine every
# later stage will actually run on. Those are both true and they are about
# different machines, so the record says which is which rather than picking one
# and hoping. Consumers that must not be fooled read it; `check_deploy_kit` reads
# it to decide whether `environment.md` can be compared against the live record at
# all (it cannot — see `rendered_from` in the layout).
: "${MOCK_GPU_ARCH:=gfx950}"
: "${MOCK_GPU_COUNT:=8}"
# **The image digest is looked up on the node, not asserted.** It was a literal
# — the sealed run's `infera/engine-sglang:gfx950-local` — and that broke rung 0:
# the record named a digest that **exists on no node and never will**, because
# `gfx950-local` was a local build on a machine that no longer has it. m3's
# `build_workset` then could not start a container from `fixed.image`.
#
# The rule the mock now follows is the one the mission states for the real
# producer: `image_id` is *discovered during bring-up*, and a variable holding it
# would be a claim rather than a measurement. A replay is a bring-up on today's
# node, so it discovers today's digest.
if [ -z "${MOCK_IMAGE_ID:-}" ] && [ -n "${E2E_IMAGE:-}" ]; then
  MOCK_IMAGE_ID="$(bash -c '. "$1"; on "docker image inspect \"$2\" --format {{.Id}}"' _ \
      "$PKG/assets/lib/remote.sh" "$E2E_IMAGE" 2>/dev/null | tr -d "\r\n" | grep -oE 'sha256:[0-9a-f]+' | head -1)"
fi
if [ -z "${MOCK_IMAGE_ID:-}" ]; then
  echo "mock_adapt: could not read a digest for '${E2E_IMAGE:-<unset>}' on the node." >&2
  echo "  The environment record would name an image that may not exist there, and a" >&2
  echo "  later stage starting a container from fixed.image would fail with something" >&2
  echo "  that names neither this file nor the digest. Pass --var image=<an image on" >&2
  echo "  the node>, or set MOCK_IMAGE_ID explicitly if you know what you are doing." >&2
  exit 3
fi
: "${MOCK_ENDPOINT:=http://${E2E_NODE_IP:-127.0.0.1}:${E2E_PORT_ROUTER:-8101}}"

# `runtime.transport` is **not** set here. `E2E_TRANSPORT` ships defaulted to
# `auto`, which `environment.schema.json` rightly refuses — "decide later" is not
# something a reproducer can act on — and this script resolved it for one
# revision. `env_render.build()` now does it for every producer, which is the
# right place: a mock resolving it separately would be a second rule to keep in
# step with `remote.sh`, and the two would diverge silently.

# **Not bare `python3`.** `env_render.py` validates before it writes, so its
# interpreter must be able to import the validator stack. A task body cannot name
# the run's interpreter — `cli/main.py:668` puts `AGENT_SYS_DEMO_PYTHON` in
# `validation_env` only, and its own comment says a task body never reaches it —
# so the policy `PATH` decides, and on this host that is `/usr/bin/python3`.
#
# **`jsonschema` and `yaml` only.** `referencing` was in this list and that was
# wrong twice over: `/usr/bin/python3` does not have it, which made this the
# blocker for every module's MOCK-MAP (A); and `schema.py` no longer needs it,
# because it inlines cross-file `$ref`s instead of building a registry. Probed
# rather than assumed, so a host where the policy interpreter is thinner still
# fails with a sentence instead of a traceback.
PY=""
for candidate in "${AGENT_SYS_DEMO_PYTHON:-}" python3 /usr/bin/python3; do
  [ -n "$candidate" ] || continue
  if "$candidate" -c 'import jsonschema, yaml' >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
  echo "mock_adapt: no interpreter here can import jsonschema and yaml;" >&2
  echo "env_render.py validates before it writes and cannot run without them." >&2
  exit 2
fi

# `${VAR-default}` and **not** `${VAR:-default}`: the first substitutes only when
# the name is *unset*, the second also when it is set-but-empty. `gate.sh` sets
# `MOCK_REPLAYED_FROM=""` deliberately, to build a fixture that stands for a real
# bring-up and must therefore face the strict `rendered_from` comparison. With
# the colon form that empty string fell through to the default, the fixture was
# marked replayed, and the gate stopped testing the rule it exists to test —
# measured: a planted fault went unreported.
REPLAY_SET=()
_replayed="${MOCK_REPLAYED_FROM-${E2E_MOCK_ROOT:-?}/stage1-deploy/deploy_kit}"
if [ -n "$_replayed" ]; then
  REPLAY_SET=(--set "runtime.replayed_from=${_replayed}")
fi

"$PY" "$PKG/assets/lib/env_render.py" --new --content-type code --out "$CONTENT" \
  --set "fixed.gpu_arch=${MOCK_GPU_ARCH}" \
  --set "fixed.gpu_count=${MOCK_GPU_COUNT}" \
  --set "fixed.image_id=${MOCK_IMAGE_ID}" \
  --set "runtime.endpoint=${MOCK_ENDPOINT}" \
  "${REPLAY_SET[@]}"

# --------------------------------------------------------------------------- #
# (I) the runtime contract.
#
# Appended to `scripts/env.sh`, which every other script in the sealed kit
# sources. Idempotent: a second run of this adaptation must not append a second
# copy, because a mock run that is re-driven is the normal case.
ENVSH="$PACKUP/scripts/env.sh"
[ -f "$ENVSH" ] || { echo "mock_adapt: no $ENVSH to adapt" >&2; exit 1; }

if ! grep -q 'E2E_KIT_RUN_TAG' "$ENVSH"; then
  cat >> "$ENVSH" <<'EOF'

# --- deploy_kit.layout.yaml runtime_contract (added by mock_adapt.sh) --------
# A rename, not new behaviour: each falls back to the `DK_` name this kit already
# honours, so an unset caller gets the sealed run's own defaults byte for byte.
: "${E2E_KIT_RUN_TAG:=${DK_RUN_TAG:-$(date +%Y%m%d-%H%M%S)-$$}}"
: "${E2E_KIT_PORT_BASE:=${DK_PORT_BAND_LO:-8100}}"
: "${E2E_KIT_WORK_ROOT:=${DK_WORK_ROOT:-/mnt/m2m_nobackup/yihou/deploy}}"
: "${E2E_KIT_ENGINE_EXTRA_ARGS:=}"
: "${E2E_KIT_ENGINE_EXTRA_ENV:=}"
: "${E2E_KIT_ROUTER_EXTRA_ARGS:=}"
# The seventh, added 2026-09-04 with the contract entry that requires it. Same
# rename-not-new-behaviour rule as the six above: the sealed kit already binds a
# card, it just bound it as `DK_GPU_ID`, so an unset caller still gets device 4 —
# the sealed run's own choice, byte for byte.
#
# **This line is why the contract needs no replay exemption.** Adding the
# parameter refused the sealed kit, and the first fix written was a
# `skip_if_replayed` escape in the validator. That was wrong twice over: every
# replay in this flow passes through *this script*, so the escape would never
# have fired, and it would have weakened the rule for any replay that did. The
# adapter is the seam that exists for exactly this — bringing an older artefact
# up to the current contract — so the parameter is now required of every kit,
# replayed or produced, with no exemption anywhere.
: "${E2E_KIT_GPU_DEVICES:=${DK_GPU_ID:-4}}"
# The eighth, added the same day and by the same rule: rename, never new
# behaviour. The sealed kit already binds a graph ceiling — `DK_CUDA_GRAPH_MAX_BS`
# at its own `env.sh:105`, value **8** — so an unset caller still gets the
# 2026-09-02 run's own choice byte for byte, and the replayed numbers stay
# exactly as reproducible as they were.
#
# **The sealed 8 is deliberately not corrected here.** It is right for the kit it
# came from, which is `tp_size: 1`, and this script adapts a record forward — it
# does not re-tune a deployment that already happened. A replayed kit that
# quietly served a different ceiling from the one its own logs show would break
# the only thing a replay is for.
: "${E2E_KIT_CUDA_GRAPH_MAX_BS:=${DK_CUDA_GRAPH_MAX_BS:-8}}"
EOF
fi

# --------------------------------------------------------------------------- #
# (J) the deployment entrypoints, so the expensive validator can do real work.
#
# **The alternative was to drop `check_deploy_serves` in mock mode, and that is
# worse.** It is `strength: strong`, and strength qualifies a PASS and never a
# failure (`validator/report.py:177`) — so there is no "record the refusal and
# carry on" switch to build, and a validator skipped because the run was a mock
# is the failure this package exists against. Installing a kit it can genuinely
# serve from means the `gpu_hours` validator is **exercised** in every mock run,
# passes only by earning it, and grades a kit the record already says is mocked.
#
# **The stub goes BESIDE the kit's scripts, never over them** — leader's ruling
# 2026-09-05, and it reverses what this block used to do.
#
# It used to `mv` the real entrypoints to `scripts/sealed/` and write the stub
# into `scripts/`. That satisfied the one consumer it was for —
# `check_deploy_serves`, which must run `deploy.sh` to prove the kit serves and
# on a *mocked* stage would otherwise pay a full bring-up — by lying to every
# other reader of the same path. Measured on rung 2h: m2's `mode_on` came up in
# **1 second** against `stub_…_pmon`, probed 404 and exited 1. Nothing was
# broken; the swap did exactly what it said, and "cheap for the validator" and
# "true for the consumer" were the same file.
#
# They are now two files. The kit keeps its real entrypoints, the stub lives at
# `scripts/stub/`, and the validator is pointed at it by the parameters it
# already had — `deploy_entrypoint` / `teardown_entrypoint`
# (`m1_deploy.yaml:245-246`, confirmed present in a real run's `args.json`).
# **No new plumbing, and `scripts/sealed/` stops being needed** because nothing
# is displaced.
STUB="$PKG/assets/check_deploy_serves.validator/stub_kit"
[ -d "$STUB" ] || { echo "mock_adapt: no stub kit at $STUB" >&2; exit 1; }

mkdir -p "$PACKUP/scripts/stub"
cp "$STUB"/deploy.sh "$STUB"/wait_ready.sh "$STUB"/teardown.sh    "$STUB"/stub_env.sh "$STUB"/stub_router.py "$PACKUP/scripts/stub/"
chmod +x "$PACKUP/scripts/stub"/deploy.sh "$PACKUP/scripts/stub"/wait_ready.sh          "$PACKUP/scripts/stub"/teardown.sh "$PACKUP/scripts/stub"/stub_router.py

if ! grep -q 'MOCKED DEPLOYMENT ENTRYPOINTS' "$PACKUP/notes.md"; then
  cat >> "$PACKUP/notes.md" <<'EOF'

## `scripts/stub/` — what it is, and who it is for

**`scripts/` is untouched and real.** `deploy.sh`, `wait_ready.sh` and
`teardown.sh` are the entrypoints the 2026-09-02 bring-up ran, byte for byte, and
a consumer that runs them is running the real thing.

`scripts/stub/` is a **stand-in for one reader only**: `check_deploy_serves` must
execute a `deploy.sh` to prove the kit serves, and on a *mocked* stage 1 doing
that for real would pay a full bring-up and defeat the mocking. The stub brings
up a small HTTP server that answers the probe set and **serves no model**. Every
number produced through it is meaningless.

**It is reached by parameter, not by path** — a mocked run passes
`--var deploy_entrypoint=scripts/stub/deploy.sh` and
`--var teardown_entrypoint=scripts/stub/teardown.sh`. **Nothing else should point
at `scripts/stub/`.** If you are not `check_deploy_serves`, you want `scripts/`.

This replaces an earlier arrangement in which the stub was written *over*
`scripts/` and the originals moved to `scripts/sealed/`. That satisfied the
validator by lying to everyone else: rung 2h's `profiling_mode_on` came up in
1 second against the stub, probed 404 and exited 1. There is no `scripts/sealed/`
now because nothing is displaced.

A validator passing against the stub has shown that the validator works. It has
**not** shown that a model was served.
EOF
fi

echo "mock_adapt: ${PACKUP##*/} now carries codes/environment.yaml, the runtime contract, and a stub kit at scripts/stub/ (scripts/ left real -- point check_deploy_serves at the stub with --var deploy_entrypoint)" >&2
