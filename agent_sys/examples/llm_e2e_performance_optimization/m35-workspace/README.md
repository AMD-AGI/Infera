# m35's workspace — what is here and what each thing is worth

**Written 2026-09-06, end of the second cluster's first day.** Owner: m35 (module 3 /
4 / 5 / `packup`). **Nothing here was ever run against a real `operator_workset`** —
six chain runs, none reached m3. Read that limit into everything below.

Every tool here **errors on bad input rather than returning empty**, and every one
was **known-answer tested before use**. Where a test could not be constructed
honestly, that is said rather than simulated.

---

## The tools

### `m3_extract.py` — the gate

```sh
python3 m3_extract.py <operator_workset content dir>     # exit 0 usable, 1 STOP, 2 bad path
```

Reads the four facts a degraded module-4 artefact needs from m3's workset, plus the
one risk that cannot be closed earlier.

| it reads | why |
|---|---|
| `integration.substitution` | **STOP on `call_site_fragment`** — `apply.py:808` hard-stops it with `overlay_files`, and `apply_mode`'s enum has one value, so the pair is unsatisfiable |
| `integration.build_step` | **STOP if non-empty** — `apply.py:395` refuses; needs a rebuild |
| `integration.public_symbol` | → `mk_reverse_payload.py --delegate-to` |
| `integration.target_files` | → `--container-path` |
| `edit_target.entry_function` | → `--function`, the `first_call` marker |
| the Definition's `baseline` | **the shape that decides the whole route** — see below |
| the Definition's `inputs` keys | the delegation-signature risk |

**Known-answer tested on five fixtures under `known/`:** module-shaped, harness-shaped,
`call_site_fragment`, a mixed multi-operator workset, and a missing path. The mixed
case found a real defect (a stop on operator 1 suppressed operator 2's output) that
**both single-operator cases passed** — the defect lived only in the shape the real
input has.

### `make_gate_result.sh` — the artefact generator

```sh
bash make_gate_result.sh <run dir> <workset content dir> [orchestrator pid]
```

Writes `M3-GATE-RESULT.md`. **It exists so the answer lands as a file rather than a
message**, because a hold can end mid-sentence. It carries a banner telling the
reader whether it is a **result** or a **fixture rendering** — check the `workset`
line. **Every copy made on 2026-09-06 is a fixture.**

It reads the launch line from `/proc/<pid>/cmdline`, not from any document: *a record
says what was intended, the process says what it got.* **The run id is not on the
orchestrator's command line** (only `--demo-root`, which six runs share), so pass the
pid explicitly; the file records whether the pid was `supplied explicitly` or
`inferred from --demo-root (an association, not an identity)`.

### `mk_reverse_payload.py` + `control_reverse_payload.py` — the degraded m4 payload

```sh
python3 mk_reverse_payload.py --image <img> --container-path <inside> \
    --operator <id> --function <Class.method> --delegate-to <public_symbol> --out <dir>
```

Builds a **reverse optimisation**: the engine's own module, pulled out of the image,
plus two `print()` markers and a `run()` delegation. **`base_sha256` is derived from
the bytes it pulled**, so the hash and the payload cannot disagree — there is no
`--base-sha256` to supply by hand.

**Why this shape and not the workset's baseline:** two validators pull the same file
in opposite directions. `apply.py:828` requires the **stock module's whole public
surface**; `check_speedup_substantiated` execs it as the harness `--impl` and requires
a top-level **`run`**. Only *stock module + appended delegation* satisfies both.

**Nine controls, all passing** (`bash control_reverse_payload.py probe_payload`),
including: a harness-shaped payload is refused with the 12 dropped definitions named;
markdown fails `compile()`; `compile()` is stricter than `ast.parse()` on 4/4 cases;
`get_type_hints` resolves across **every** script in this directory; and `die()`
terminates, established by calling it.

### `port_idiom_check.sh` — will this kit refuse on the port line

```sh
bash port_idiom_check.sh <run dir>      # 0 overridable, 1 will refuse, 3 no kit yet
```

Checks for the **presence of the good form** (`: "${DK_PORT_ROUTER:=…}"`), not the
absence of the bad one — a non-match is not evidence. **Exits 3 when no kit exists**,
so *not yet written* cannot be read as *clean*. Validated against three runs whose
outcomes were already recorded: it reproduces the refusal and both passes.

**Trap it avoids:** `find <run> -name env.sh` returns two **package examples** under
`workspace/examples/`, which are not the kit.

### `reset_gpus.guard.patch` — **UNAPPLIED, deliberately**

A guard for `assets/serve/reset_gpus.sh`, which is a **node-wide `kill -9` with a
working `sudo` fallback**. It spares any KFD holder belonging to a container we did
not create, ownership by the `infera_e2e_run` label — never by name, since `yihou_` is
shared.

**Fails toward sparing.** Every error path returns `unknown:<why>`, which routes into
a branch the script already has: the VRAM floor times out and `mix_up.sh` aborts
**loudly, before any bring-up**. No new failure mode.

**Seven controls** (`bash test_guard.sh`), including both real foreign containers
spared and — deliberately — an **unlabelled container of ours** spared, because
`mix_up.sh`'s own record says an unlabelled generically-named GPU container is
indistinguishable from a stranger's.

> **Why it is not applied:** the case never arose — the foreign containers never took
> a GPU — and the package is **staged per task**, so an untested change landing
> mid-chain reaches that run's later stages. It lands if a foreign container takes a
> GPU before m5, or after the round as hardening. **Do not apply it during a live run.**

**What no control covers, stated rather than simulated:** a real KFD-holding process.
A foreign GPU workload cannot be manufactured honestly, so none was faked. The guard
changes only the ownership question; the kill is untouched.

### `watch_for_verdict.sh` / `watch_chain.snapshot.sh`

Blocks until a run writes its first `verdict.json`. **Run the `.snapshot.sh` copy, not
the source** — bash reads lazily and editing a running script moves its unread offset.
The snapshot carries the source's sha256 in line 1.

**No `2>/dev/null` anywhere in it, deliberately:** its zero is the thing being trusted,
and it aborts if `find` itself fails rather than counting that as "no verdicts yet".

---

## The documents

| file | what it is |
|---|---|
| `PRE-REGISTER.md` | The bars for all 12 validators on m3/m4/m5/`packup` kinds, **fixed before any result arrived**, with dated append-only corrections. **Never backfilled.** |
| `RUNG5-CHECKLIST.md` | P0–P12, each a stop with a command. Includes the module-5 STOP, the GPU-exclusivity rule during `check_workset_runs`, the redact-prefix trap, and the pre-authorised `eval_names` contingency. |
| `LAUNCH-CHAIN-m35.md` | My independent derivation of the chain launch line, written **before** reading m2's. Kept for the disagreements, which are recorded rather than resolved away. |
| `M3-GATE-RESULT.md` | **A fixture rendering as of 2026-09-06.** Check its banner. |

---

## The three things most worth knowing

1. **The gate's decision is unchanged and unanswered.** Is m3's `baseline`
   **harness-shaped** (defines `run`, not the public symbol → `forge_mock=1` refuses at
   `apply.py:828` after a bring-up → use the reverse payload) or **module-shaped**
   (→ m2's single `forge_mock=1` line survives and my two-form split retires)?
   **One command on a real workset settles it.**

2. **A degenerate m3 answer is pre-registered, not a producer defect.** Four
   explanations, in order: `stack_window_s=0` removes launcher blocks so `identify`
   loses resolution **level 1** and falls to Magpie, which has never completed a real
   scan here; `repos == []`; `min_resolve_ratio: 0.0` refuses nothing; the workload's
   prefix-hit profile.

3. **`packup` has never been reached with module 5 real, anywhere.** The other cluster
   reached `packup` with 3/4/5 replayed and `check_no_regression` passing on a report
   whose comparison blocks self-declare `unavailable_because: mock` — **it passed by
   having nothing to judge.** That gap is the round's actual target.


---

# APPEND 2026-09-07T06:47:51Z — status at stand-down. The header above is out of date; read this.

**The line above says "six chain runs, none reached m3". That was true when it
was written and is false now.** Appended rather than rewritten, because a
snapshot that gets edited to stay current stops being a snapshot.

## What is true at stand-down

```
m1 -> check_deploy_serves -> _off -> _on -> merge -> rank -> identify -> build_workset
    reproduced fully real, 21/21 verdicts, zero refusals
m4 sealed        NEVER, on either cluster
m5 at all        NEVER
packup at all    NEVER  -- 22 e2e_packup handoff records, none ever left `created`
```

**So the limit to read into everything here has moved one stage, not vanished.**
The m4/m5 material below was still never exercised against a real
`kernel_optimization`.

## What in this directory is worth anything, and what is superseded

| file | state |
|---|---|
| `launch-records/` | **the launch lines.** A run does not record its own `--var`s, so these are the only place several runs' parameters exist. Keep. |
| `reset_gpus.guard.sh` / `.patch` / `test_guard.sh` | **UNAPPLIED** ownership guard, seven controls run. Still unapplied and still relevant: `reset_gpus.sh` is unchanged. |
| `mk_reverse_payload.py` / `control_reverse_payload.py` | degraded-m4 payload builder, nine controls. **Never used against a real m4** — m4 has never sealed. |
| `m3_extract.py` / `make_gate_result.sh` | the m3 gate. `M3-GATE-RESULT.md` is a **fixture rendering**, not a result; its banner says so. |
| `identify_unique_names.patch` | **SUPERSEDED** — applied as `649af26b`. Kept for the reasoning, not for the diff. |
| `port_idiom_check.sh`, `watch_*.sh` | session utilities. Low durable value; kept so nothing is lost by my judgement alone. |
| `LAUNCH-CHAIN-m35.md` | an early launch derivation, superseded by `RUNG5-CHECKLIST.md` in the parent directory. |

## The documents that outlived the session are NOT here — they are in the parent

`RUNG5-CHECKLIST.md`, `M5-REACHABILITY.md`, `PACKUP-REACHABILITY.md`,
`PRE-REGISTER.md`, `ON-ARM-REFUSAL.md`, `PREFLIGHT-FIX-FOR-NEXT-LAUNCH.md`,
`environment_flag_tier2.proposal.md`, and `e2e-flow/assets/lib/packup_redact_probe.py`.
**Look there first; this directory is the scratch that had nowhere else to go.**

## One sentence to carry into a restart

**"I read its structure, not its verdict, and those are different things."**
