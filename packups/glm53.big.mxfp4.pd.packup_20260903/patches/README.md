# Patches — what each one is, why it was needed, and what symptom it cured

Apply all five to `AMD-AGI/Infera` at branch `yihou.dev.glm53.expr`. Order is the
commit order below. **Without them the single-node PD shape does not run**, or
runs on the wrong topology and produces numbers that look fine.

Every one preserves the **two-node** path byte-identically. That was verified by
running the real `engine/up.sh` with `SSH_CMD=echo`, so each per-leg command is
printed instead of executed — no node, no container, no GPU.

---

## `4493e33.patch` — unblock the shape (three defects)

**What.** (a) Both GLM-5.3 wrappers exported `IMAGE`, but `engine/up.sh` and
`common.sh` require `INFERA_IMAGE`. (b) `leg.sh`'s `KV_PUB_PORT`/`KV_SNAP_PORT`
were never forwarded by `up.sh`. (c) `leg.sh` hardcoded
`MC_DISABLE_HIP_TRANSPORT=1` with no override.

**Why.** (a) `IMAGE` is `preflight_rdma.sh`'s variable name; the wrappers picked
it up for the engine one. (b) `on()` runs a fresh remote shell, so nothing a
wrapper exports reaches `leg.sh` except through `up.sh`'s explicit per-leg
prefix — in the two-node shape the legs are on different hosts and the shared
default never collides. (c) See `667b02f` — this turned out to be a dead knob.

**How.** Export both image names and add `INFERA_IMAGE` to the placeholder guard;
forward the KV ports per leg defaulting to `leg.sh`'s own values; make
`MC_DISABLE_HIP_TRANSPORT` overridable.

**Symptom cured.** (a) `up` died at its first `require_env` before a container
started. (b) Both legs bound 5557/8801 in one network namespace; the second died
at bind with `port_base at N is not available`.

**Caveat, added later.** The (c) change is **not a fix** — it makes a dead knob
configurable and changes nothing. Kept only as the carrier for the live name.

---

## `f6ee2da.patch` — fill in the transport values from preflight

**What.** Default `RDMAV_FORK_SAFE=1` in the single-node wrapper only; correct the
`RDMA_IB_DEVICES` placeholder from a rail *list* to a single device.

**Why.** `preflight_rdma.sh` recommends `RDMAV_FORK_SAFE=1` in all three modes,
and `leg.sh:66` honours it **only when passed in** — silent otherwise. And
preflight's rule for this shape is explicit: *"all active rails (cross-node);
**Single-node loopback: pin ONE device on both legs instead**"*. The placeholder
was the cross-node recipe.

**How.** `${VAR:-...}` in the single-node wrapper. Deliberately **not** defaulted
on for two-node: that shape was validated without it, and a validated path should
not move on a recommendation rather than a measurement.

**Context.** Also records a correction: an earlier claim that `common.sh`'s
`docker run <etcd-image> etcd --flags` form was broken is **wrong** —
`docker run --rm quay.io/coreos/etcd:v3.5.14 etcd --version` works. That claim
was inherited from a note and asserted without testing.

---

## `1b5ea46.patch` — the single-knob / per-leg design error

**What.** The GLM-5.3 wrappers used single knobs (`MTP`, `DPA`, `GPUS`,
`EXTRA_ENGINE_ARGS`) against a strictly per-leg interface. Each silently did
nothing.

**Why it is the important one.** **None of them errored**, and each degraded to a
plausible default:
- `MTP=0` → decode leg ran **`mtp=1`**.
- `GPUS` → **both legs on GPUs 0-3** (263.8 GB on 0-3, 0.3 GB on 4-7), and the
  prefill leg then died on `Loaded weights leave no GPU memory for the KV
  cache ... Raise --mem-fraction-static above 0.773` — a number that is
  arithmetically right and diagnostically wrong.
- `EXTRA_ENGINE_ARGS` → `--disable-shared-experts-fusion` never reached the
  engine.
- `DPA=1` → unread, but `up.sh`'s defaults happen to be the intended 0/1, so
  **silently correct by luck**.

**How.** `up.sh` forwards `GPUS` per leg with `${PREFILL_GPUS:+GPUS=...}` —
injection **only when set**, never a `:-` default, which would push a literal GPU
list into the two-node path where `leg.sh`'s `seq 0..TP-1` is correct and stays
correct as TP changes. `EXTRA_ENGINE_ARGS` forwarded through `COMMON_ENV`
(quoted, so multiple flags survive) and appended unquoted in `leg.sh` so they
word-split. Wrappers now export the per-leg names, with the single knob kept as a
convenience seed.

**One change beyond the six:** `cluster.2node.sh` had
`PREFILL_GPUS=DECODE_GPUS=0..7`. Inert before; **live the moment `up.sh` started
forwarding it**, and silently wrong at any TP but 8. Set to unset.
*Generalisable:* when you make a dead knob live, every existing setting of it
becomes a config change.

---

## `667b02f.patch` — `MC_DISABLE_HIP_TRANSPORT` is a dead name

**What.** Forward `MC_DISABLE_HIP`, the name the binary actually reads.

**Why.** Exact-match against the shipped
`mooncake/engine.cpython-310-x86_64-linux-gnu.so`:

```
MC_DISABLE_HIP             1     <- the real knob
MC_DISABLE_HIP_TRANSPORT   0
MC_ENABLE_HIP_TRANSPORT    0
MC_USE_HIP_IPC             1     <- a second gate; inverts its name
```

Behavioural evidence came first and is the stronger half: a leg launched with
`MC_DISABLE_HIP_TRANSPORT=1` **confirmed present in `/proc/<pid>/environ`** still
logged `HIP transport installed` 4× per leg, identical to the run with it at 0.

**How.** `${MC_DISABLE_HIP:+...}` in `up.sh`'s `COMMON_ENV` and `leg.sh`'s
`docker exec env`. Default absent.

**Retraction carried in the commit message.** Any claim that "as shipped, KV
would have moved over loopback RDMA" is **wrong**. hip installs regardless; the
shape was never at risk through these knobs. INFERRED: whoever wrote the original
hardcode probably meant to disable hip and used a name that does not exist —
which would explain why no reason for the line could be established.

**Practical rule recorded in both files:** a hip A/B **must** use
`MC_DISABLE_HIP` and **must** verify via `/proc` + the `:489` source read, never
the install log — which reads 4/4 in both states forever.

---

## `b2b1a08.patch` — README note 3 did not cover the startup-profiling OOM

**What.** Documents a third `--mem-fraction-static` failure form and why the
engine's own advice is a trap in it.

**Why.** Note 3 taught two failure directions, both **under load**. The
single-node bring-up hit a third: an abort during **startup profiling**, weights
loaded, no request served, where raising the fraction as instructed yields a
working deployment on the wrong topology.

**How.** Adds the discriminator between all three forms — this one aborts at
startup profiling; the classic prefill form aborts under load at *low* token
usage; the decode form retracts under load at *high* token usage — plus the two
checks that do **not** work (`--showmemuse`'s `VRAM%`, and per-leg `base_gpu_id`).
