# LAUNCH-CHAIN — m1 → m5 → `packup` in one `agent-sys run`

**Drafted 2026-09-06T07:30:14Z** (`date -u`, read once and referenced).
**A document. Nothing here has been launched.** m35 is deriving the same line
independently against `RUNG5-CHECKLIST.md` P0-P10; two derivations, not one
reviewed one.

---

## 0. The open question, answered: **ONE form, complete. Not two.**

I was asked to say which I was writing and why, and not to assume. **The answer
changed once I read the mechanism**, so here is the evidence rather than the
conclusion.

The premise of "two forms" was that a degraded m4 needs a payload m35 must build
from a real m3 workset that does not exist yet. **That is true of a *standalone*
m4 and false of the chain**, because the two degradation mechanisms are
different things:

| mechanism | what it does | needs a corpus? |
|---|---|---|
| `--var mock_stages=…,m4` | `mock.sh` copies a sealed handoff from `$E2E_MOCK_ROOT/stage4-kernel-opt/kernel_optimization/content` | **yes — and that does not exist on this cluster** |
| **`--var forge_mock=1`** | reaches the **real** m4 path as `KFO_MOCK` (`m4_kernel_opt.yaml:426`); `30_run_forge.sh:64` skips the campaign | **no** |

`forge_mock=1` does not fabricate a handoff. Read at `30_run_forge.sh:64-125`,
it seeds `optimized_kernel.py` **from the staged workset's own recorded
`baseline`, verbatim**, with a self-declaring header — *"this is the WORKSET'S
OWN BASELINE… there is no optimization in it, so any ratio measured against it
is ~1.0 by construction"* — and writes a `forge_result.json` filled with
**nulls**, carrying `_what_this_file_is: "KernelForge did NOT run…filled with
nulls rather than plausible-looking numbers so that it cannot be mistaken for a
campaign summary."` STEP 4/5 (apply, verify) then run for real.

> **The workset it seeds from is this run's own m3 output.** That is exactly the
> dependency m35 cannot satisfy standalone and the chain satisfies by
> construction, because m3 runs before m4 inside the same graph.

So: **one complete line, m4 degraded by `forge_mock=1`, no dependency on any
payload and no corpus.** The chain does not need to stop before m4.

**What this costs, stated plainly:** m4 contributes a **no-op** optimisation, so
`speedup ≈ 1.0` is the *correct* answer rather than a disappointing one, and
nothing in this run tests the campaign path. That is what this round asked for —
reachability, not improvement.

---

## 1. The command

```sh
env -u PYTHONPATH \
CLAUDE_CONFIG_DIR=<m1's shared farm — get the exact value from the leader> \
python3 /data/yihou/e2e_verify_20260906/m2/launch_chain.py \
  --stall-after 900 run \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /data/yihou/agent_sys_runroot \
  --timeout 21600 \
  --var jobid=29313 \ \
  --var node=smci355-ccs-aus-n04-25 \ \
  --var node_ip=10.235.192.131 \ \
  --var transport=local \ \
  --var model_name=Qwen/Qwen3-32B \ \
  --var model_path=/apps/data/models/Qwen3-32B \ \
  --var image=infera/engine-sglang:qwen3-local-20260906 \ \
  --var context_length=40960 \ \
  --var tp=4 --var expect_ranks=4 \ \
  --var dsa_args=none --var parser_args=none \ \
  --var mock_stages=none \ \
  --var work_root=/data/yihou/e2e_flow11 \ \
  --var scratch_root=/data/yihou/e2e_flow11/kfo \ \
  --var validate_work_root=/data/yihou/e2e_flow11/validate \ \
  --var container=yihou_e2e_chain11 \ \
  --var port_router=8101 --var port_worker=8102 --var port_etcd=8103 \ \
  --var measure_gpu=4 \ \
  --var magpie_root=/data/yihou/Magpie \ \
  --var aiperf_trace=/data/yihou/e2e_verify_20260906/m2/materials/conversation_trace.v2.jsonl \ \
  --var gsm8k_data=/data/yihou/e2e_verify_20260906/m2/materials/gsm8k_test.jsonl \ \
  --var trace_end_ms=120000 \ \
  --var bench_rounds=3 \ \
  --var adhoc_cases=3 \ \
  --var eval_thinking=none \ \
  --var forge_mock=1 \
  --var kernel_table_min_launchers=0 \
  --var gpu=4 \
  --var remote_home=/data/yihou \
  --var transport_env=E2E_REMOTE_HOME=/data/yihou \
  --var instruction='Bring this model up in the requested mode on this host, prove it answers through the router, and hand back a kit somebody else can run. Follow the task readme STEPS, including the STEP 8 self-check against assets/schemas/deploy_kit.layout.yaml. Write the kit scripts FIRST. Then perform EXACTLY ONE bring-up, using the delivered scripts, prove it serves, tear it down, and seal. Do not bring the model up more than once - that single cycle is the shipped-bytes cycle, so the evidence still comes from the delivered code, and repeating it is what has made this stage slow. Every identifier the kit binds - each port, container name and work root - must be written as : "${NAME:=default}" so a second copy of this kit can run beside the first, and this is checked. Do not introduce a second name for a value that already has one: the previous kit added DK_ROUTER_PORT beside the existing DK_PORT_ROUTER and assigned it unconditionally, and that was refused. In the kit deploy script preflight: when a card you were told to take reads busy, do not abort on the first reading. A container counts as holding a GPU only if GPU memory is attributable to it. Mapping /dev/kfd into a container is NOT evidence that it holds a GPU - long-lived development containers on this host map /dev/kfd permanently and hold zero VRAM, so a test on HostConfig.Devices names them on every run and your stranger branch fires every time a card reads busy for any reason. Use rocm-smi occupancy, and when you cannot attribute the memory to a specific container, the honest classification is unknown - and unknown WAITS, it does not abort. Abort as a stranger only when you can name a container AND attribute non-zero VRAM to it. Your abort message must state which of the two conditions was met and how you measured the second. The validator check_deploy_serves performs its own bring-up on the same cards and its container is removed while VRAM is still draining. Expect a busy card with NO container to attribute it to for up to 60 seconds after a preceding stage finishes. That is the single most likely thing your preflight will meet, and it is precisely the case the wait exists for. So: if a container carrying the label deploy_kit_owner=yihou holds the cards, or the memory cannot be attributed to any container, wait and re-read for up to 300 seconds and abort only if the cards are still busy when that time is up. Substituting a different card stays out of scope in every case. Keep writing the deploy_kit_owner=yihou label on every container the kit starts, because this check depends on it. Whatever happens, copy the engine and router logs out of the container into the kit before you tear anything down, including when a step fails - the previous run died with its cause unread because the logs went away with the container. Out of scope: bringing up any additional deployment or configuration this run does not use - the work root is local disk and NFS is not used here - and executing the graph validators yourself. If you find a defect outside this scope, record it in the kit notes and seal anyway.'
```

> **`--demo-root` is `/data`, not `/home`. Corrected 2026-09-06 after m35's
> independent derivation.** Both of my launch documents carried
> `/home/yihou/agent_sys_runroot`, **which does not exist on this host** — I
> inherited it verbatim from `RUN-PLAN.md`'s first-cluster block, the one I was
> told at the outset to read *for mechanism, never for values*. It would have
> opened a second, unrelated run root and worked where nobody was looking.
>
> **`show` cannot catch this and neither could a reviewer**: the path reads
> perfectly sensibly, and `m2_show.sh` does not pass `--demo-root` at all
> because `show` does not need one. It was caught by someone deriving the same
> line independently and putting a different value there.
>
> RUN-PLAN's reason for `/home` was that the run root must be visible *from the
> node* over NFS. **Here `transport=local` makes "the node" this host**, so node
> visibility is not a constraint and `/data` — local disk, where every path this
> team has exchanged today lives — is correct.

> ### ⚠ The comments explaining this command are BELOW it, not inside it
>
> **A `#` comment inside a `\`-continued block joins the previous line and eats
> the rest of the command — and `bash -n` accepts it.** Measured just now while
> writing this file: `echo A \` + a comment line + `B \ C` passes `bash -n`
> with rc=0 and prints only `A`, then reports `B: command not found`.
>
> **I put the annotations inside the block first.** Pasted, that version would
> have silently dropped every `--var` after `mock_stages` — work_root,
> scratch_root, both materials, eval_thinking, forge_mock — and it might still
> have *loaded*, because the six no-default variables all appear above the cut.
> The lesson was already in `CLAUDE.md`; I wrote the bug into a launch line
> anyway, in the paragraph arguing for care. **Do not move these back inside.**

**`--var mock_stages=none` is not optional.** `shared.yaml:208` defaults it to
`all`; omit the line and every stage mocks itself, then dies on a corpus that
does not exist here. `none` names no stage, so `mock.sh` exits 3 for each and
all five run for real.

**DELIBERATELY ABSENT — do not add, either of these:**

- **any `--var m<N>_agent=`.** Each leaf must take its declared agent
  (`e2e_deployer` / `runner` / `workset_builder` / `e2e_kernel_optimizer` /
  `e2e_integrator`). In particular **`--var m4_agent=runner` diverts
  `optimize_kernel` to its MOCK `entry.sh`**, which needs the same absent
  corpus — the trap this line avoids by silence.
- **`--var gpu_devices=`.** `line.sh:181` then takes the set the kit records
  having **taken**, which is the argued-for default. Naming cards here asserts a
  fact about m1's bring-up that nobody has read.

*An omission with no explanation is indistinguishable from an oversight, which
is why both are named here rather than left to be noticed.*

**The wrapper must print `stall_after 20s -> 900s`.**  *(Verified through this
launcher: it does.)*

> ### Why a launcher and not `PYTHONPATH=… python3 …`
>
> **The two requirements cannot both be met by an environment variable, and the
> obvious line silently defeats m1's jsonschema fix.**
>
> `agent_sys/cli/main.py:31,48` imports its siblings by **bare** name
> (`from cli import …`, `from env_mgr import …`). Measured here with no
> `PYTHONPATH`:
>
> ```
> agent_sys.cli.main -> .../git.16-19/infera/agent_sys/cli/main.py     OUR tree
> env_mgr            -> .../git.16-19/infera.aiopt.all/.../env_mgr     A THIRD tree
> cli                -> .../git.16-19/infera.aiopt.all/.../cli         A THIRD tree
> ```
>
> So `-m agent_sys.cli.main` alone loads **our** `main.py` and **another tree's**
> `env_mgr`, silently. `run_with_long_stall.py:111` inserts the *cwd* (repo
> root), which fixes `agent_sys.*` and not the bare names. `<repo>/agent_sys`
> must be on `sys.path`.
>
> But it must **not** be in the environment: `env_mgr/harness.py:107` builds a
> validator's env as `live.get(key, value)` over the settings block, and
> `_RESERVED = (CLAUDE_CONFIG_DIR, CLAUDE_CODE_TMPDIR, TMPDIR, PATH)` —
> **`PYTHONPATH` is not reserved, so a live value wins over the settings file.**
> A `PYTHONPATH=…` prefix would reinstate
> `ImportError: cannot import name 'Draft202012Validator'` on **15 of 15 kinds**.
> The two cannot be merged either: the settings path is a cp310 build and this
> process is 3.13 (`ModuleNotFoundError: No module named 'rpds.rpds'`).
>
> `launch_chain.py` puts the repo on `sys.path` **in-process**, leaves the
> environment alone, **refuses to start if `PYTHONPATH` is set**, and prints
> where all four of `agent_sys.cli.main` / `cli.main` / `env_mgr.harness` /
> `agent` resolved before dispatching anything. All three controls run:
> refuses with `PYTHONPATH` set (rc=2); all four inside our tree without it; the
> stall line prints. If it does not, this is the
bare launch and the 900 s stall bound is not in effect.

**`--var mock_stages=none` is not optional and is the single most dangerous
omission on this line.** `shared.yaml:208` defaults it to **`all`**, so a chain
line that simply omits it mocks **every stage** — and then fails, because
`mock_root`'s default does not exist here. `none` names no stage, `mock.sh`
exits 3 for each, and every stage runs for real.

**No `m<N>_agent` appears anywhere**, deliberately. Each leaf then takes its
declared agent: `deploy_and_prove` → `e2e_deployer`, `build_workset` →
`workset_builder`, `optimize_kernel` → `e2e_kernel_optimizer`,
`integrate_and_verify` → `e2e_integrator`; m2's three leaves and m3's
`identify`/`rank` are `runner` by declaration because they are programs.
**Passing `m4_agent=runner` would take `optimize_kernel`'s mock entry.sh, which
needs the corpus — that is the trap this line avoids by silence.**

**`gpu_devices` is omitted** so `line.sh:181` takes the set the kit records
having taken. Reasoning in `LAUNCH-m2.md`.

---

## 2. The variable table, read as a table against **this** `mock_stages`

**102 declared variables** — not 101; `eval_thinking` is new since `VAR-TABLE.md`
was written, added by m35 at `m5_integration.yaml:144` with the `glm-45` default
preserved. **29 set by this line, 73 left at default.** The set/declared
difference was checked programmatically (`set - declared == []`), so no `--var`
on this line is a name the package does not accept.

### 2a. Set, and why — the rows where the default is wrong

| var | default | why it is overridden |
|---|---|---|
| `mock_stages` | **`all`** | see above — the whole chain mocks itself otherwise |
| `context_length` | `262144` | `Qwen3-32B` `max_position_embeddings = 40960` |
| `transport` | `auto` | m1 mints the resolved value into the record; `_agree_or_die` refuses later *as a message about transports* |
| `tp`, `expect_ranks` | `8`, `8` | this host runs **TP 4**. `expect_ranks` does not track `${tp}` by design |
| `work_root` | `/mnt/m2m_nobackup/...` | **that mount does not exist here** |
| `scratch_root` | `/mnt/m2m_nobackup/.../kfo` | same mount, **and it must sit under `work_root`** — `SCRATCH_ROOT` is not in `packup.py`'s redact prefix table, so an outside path makes packup refuse **after the whole chain has run** (m35) |
| `validate_work_root` | `/mnt/m2m_nobackup/.../validate` | same mount; third member of that family |
| `magpie_root` | `/shared_nfs/chaox/Magpie` | absent here, and the failure is **hard**: `run_profiling_mode_on` dies at ranking |
| `aiperf_trace`, `gsm8k_data` | *(empty)* → `${…:?}` | no default and no source on this cluster |
| `trace_end_ms` | **`180000` in `shared.yaml:149`, `60000` in `m5_integration.yaml:126`** | one variable, two defaults. Passed so neither decides |
| `eval_thinking` | `--thinking-mode glm-45` | this host serves **Qwen3**; the wrong parser scores **0.00 while every request succeeds** |
| `forge_mock` | `0` | §0 |
| `adhoc_cases` | `3` | m5 is real, so `3` is correct — passed to record intent, not to change the value |
| `measure_gpu` | *(empty)* → `${…:?}` | no default; `4` is outside m1's `0-3` |

### 2b. Left at default and **verified safe for this configuration**

- `aiperf_image` — **the image is now local** (pulled 06:34Z, `aiperf --version` → 0.12.0). Was a blocker this morning; is not one now.
- `etcd_image` — `quay.io/coreos/etcd:v3.5.14` **is** present locally.
- `mock_root` — its default does not exist, **and with `mock_stages=none` it is unreachable**: `mock.sh` never runs. *Checked rather than assumed* — the other readers of `E2E_MOCK_ROOT` are mock-only paths, plus `check_deploy_kit.validator/gate.sh`, which is a **standalone developer harness invoked by hand and wired into no step**. So the missing corpus costs us **the ability to re-gate validators, not the run.**
- `served_name` empty → a body resolves it to `E2E_MODEL_NAME`.
- `transport_env` empty → **required for `spur`, not for `local`.** Do not copy the "required on every rung" row out of RUN-PLAN into this line; that row was written for a spur cluster.
- `needle_tokens=31000` — fits inside 40960 with room for the answer.
- `min_requests=50` / `integration_min_requests=50` — **counted**: my trace carries 437 records with `timestamp < 60000`, ~9× the floor.
- `kernelforge_repo`, `gpu`, `forge_max_hours`, `forge_model`, `forge_fellow` — **all inert under `forge_mock=1`**, because the campaign branch is never reached. This is why the chain does not need a KernelForge checkout.

### 2c. Left at default and **flagged** — no default, no source, or a cost

| var | issue |
|---|---|
| **`workset_operator`** | **Cannot be supplied in advance, structurally.** Its correct value is the operator name **m3 produces inside this same run**. Omitted, so `10_read_inputs.py:32` takes `--operator=None` and the code picks. **That is the documented silent-first-match hazard** — absence is not merely tolerated here, it is the only option a single-run chain has. If m3 produces more than one operator, which one m4 optimises is not chosen by anybody. |
| `eval_names` | default is **two** evals (`gsm8k mixed_prefix_gsm8k`), not one. At `eval_examples=100` that is a real time cost in m5, twice over, against a 14:00Z hold. **Worth a deliberate decision rather than a default** — I have not changed it. |
| `stock_vs_m2_tolerance` | `0.10`, and the dispersion limit in `PRE-REGISTER-m2.md` applies: m2's side of that comparison is a **single capture with nothing grading its spread**. |
| `mock_root` | do **not** pass it empty. `packup.py:521` builds a redact prefix from `E2E_MOCK_ROOT`, and an empty prefix in a prefix table is a shape worth avoiding. *I have not read `redact.py` to see whether empty prefixes are filtered — flagging, not claiming.* |

### 2d. Mocked-vs-real divergence under **this** `mock_stages`

**`mock_stages=none` means nothing is mocked, so the entire mocked-value column
is inapplicable and every variable takes its real-run value.** That is worth
stating positively rather than leaving as an absence — it is the one
configuration in which the class that cost the first round three launches
**cannot fire**, because there is no mocked stage for a real value to be wrong
against.

The three that would otherwise differ, and their real-run values here:
`expect_ranks=4` (not the mock's `2`), `adhoc_cases=3` (not the mock's `0`),
`bench_rounds=3`. `image` is the real bring-up's, not a sealed kit's.

---

## 3. `show`, with a negative control

```
$ bash /tmp/yihou_chain.sh full
rc=0    done  6 tasks in the graph; nothing was dispatched
```

Negative control — the same line with exactly `--var model_path=` removed:

```
rc=1  steps/m5_integration.yaml:83:5::$[0].env.E2E_MODEL_PATH: no value for
      ${model_path}, and it declares no default.
```

So the green means all six no-default variables are supplied and every yaml
type-checks.

**What it does not establish** — the sentence, unchanged: *it does not check
that a path exists, that an image is present, that a port is free, that
`mock_root` has content, or that a variable reaches the body meant to read it.*
The first cluster spent five launches on a `--var gpu_devices` that loaded clean
and was inert.

---

## 4. Risks I could not close, carried rather than resolved

1. ~~**`forge_mock=1` aborts if m3's real workset baseline has no top-level
   `run(`.**~~ **CLOSED 2026-09-06T07:38:50Z — by a guard, not by likelihood.** Verified
   here, not accepted from the relay:

   ```
   build_workset.task/harness/_common.py:346   no --impl -> _exec_source(definition[which], …)
   build_workset.task/harness/_common.py:293   if "run" not in namespace:
                                                   raise SystemExit(
                                                     "the Definition's {label!r} defines no `run`")
   steps/m3_analysis.yaml:256-260              check_workset_runs  strength: strong
                                               tags: logic_source: external_dynamic, cost: gpu_hours
   ```

   `check_workset_runs` **runs** the workset's own entrypoint, and with no
   candidate `--impl` that execs the baseline. **A baseline without a top-level
   `run` therefore dies at m3's own validator — before m4, cheaply, naming the
   operator** (`{which}:{operator_id}`). It cannot survive to become a late
   death. *(My first grep for the strength/cost read the wrong two validators'
   tag lines because the window crossed a block boundary; the value above is
   from the `check_workset_runs` block itself.)*

1b. **m35's failure mode is NOT covered by that guard and stays OPEN.** The
   distinction matters and the closure above must not absorb it:

   | baseline shape | `check_workset_runs` | `forge_mock=1` | `apply_patch` |
   |---|---|---|---|
   | no `run` | **refuses at m3** | — | — |
   | `run` only, standalone (harness-shaped) | **passes** | seeds it happily | **refuses at m5, after a bring-up** |
   | stock module + `run` delegation | passes | seeds it | passes |

   **Passing `check_workset_runs` establishes that `run` exists and says nothing
   about whether the file also carries the engine module's public surface.**
   That is the expensive death — unexamined between m3 and m5, surfacing at
   `apply.py:828`. m35 has declined to settle it from first-cluster evidence;
   their `m3_extract.py` decides it in one command against the real workset.
   **Unknown, and the cheap guard closing does not make it less so.**

2. **`identify` will resolve nothing.** `E2E_SGLANG_SRC`/`E2E_AITER_SRC` are
   unwired (bug record entry 1), `magpie_root` is now real but `identify`'s own
   `E2E_MAGPIE_ROOT or "/nonexistent"` fallback is a separate path, and
   `min_resolve_ratio=0.0` means **it will not refuse for having resolved
   nothing**. Combined with the 0.81 prefix-hit workload, that is **two
   pre-registered explanations on the table** if m3's worklist comes out thin.
3. **Both port triples.** `line.sh:48` adds `PORT_OFFSET=10` for
   `profiling_mode_on`, so this line implicitly claims **8101-8103 and
   8111-8113**. Free at 07:20:28Z; m1's bring-ups are in the 83xx band. **Re-read
   at launch, not from here.**
4. **`mix_up.sh` hard-codes 5557 / 8801**, so no second m5 stage can coexist on
   this node whatever ports are passed.
5. **Duration is not estimated.** m2 alone is ~12-14 min of fixed cost plus two
   bring-ups, and its two leaves **serialise on a `resources: {gpu: 8}` lease**.
   m5 adds two more arms. I am not converting that into a figure for a 14:00Z
   hold; the bring-up cost is the term nobody has pinned and m1 has now measured
   it three times.
