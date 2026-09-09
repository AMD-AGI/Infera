# The full-chain launch line — m35's independent derivation

Written 2026-09-06 from `RUNG5-CHECKLIST.md` P0–P10 and the code. **m2's
`LAUNCH-CHAIN.md` has not been opened**; that is the point of there being two.

Sources: `shared.yaml` + `steps/*.yaml` (the only two declaration sites),
`assets/lib/mock.sh`, `assets/lib/mock_m5.sh`, `assets/optimize_kernel.task/`,
`assets/apply_patch.task/apply.py`, `RUN-PLAN.md` **for command shape only**.

---

## 0. The finding that decides the shape, and it is not in the decided-values list

**There is no sealed corpus on this cluster, so NO stage can be mocked at all.**

Measured, stderr visible: `/shared_nfs` holds **0 entries**;
`/data/yihou/cheat_for_mock` and `/data/yihou/mock_corpus` do not exist.

```
mock.sh:61      if [ ! -d "${E2E_MOCK_ROOT}/${stage}" ]; then  … exit 1
optimize_kernel.task/entry.sh:27-31
                rc==1 -> "not falling through to the real path, because a mock
                that half-copied is not an unmocked run."  exit 1
```

`E2E_MOCK_STAGES` defaults to **`all`** (`shared.yaml`, `steps/*.yaml`), so an
unset `mock_stages` makes **every** stage try to mock and **die on the missing
corpus**.

> **`--var mock_stages=none` is mandatory, not "excluding m5".** The decided
> value in the brief — *"`mock_stages` excluding m5"* — is necessary but not
> sufficient: any stage left in the list aborts, because the corpus it would
> copy from does not exist here. This is my first disagreement with the brief
> and I am confident in it.

`none` matches nothing in `mock.sh:48`'s `case`, which yields `exit 3` = *"running
for real"* — the intended decline, not an error.

## 0a. Which means m4 has no route this hold, and that is a decision, not a variable

Three ways past m4, all closed today:

| route | why it is closed |
|---|---|
| real KernelForge campaign | hours; `optimize_kernel` is `kind: ai` and there is *no program body* for it (`entry.sh:58-62`) |
| `--var forge_mock=1` | seeds `optimized_kernel.py` from the workset's **`baseline`** (`30_run_forge.sh:64-115`), which a real m3 writes harness-shaped → `apply.py:828` refuses every dropped definition. **Structural, every operator.** |
| `--var mock_stages=m4` | needs `$E2E_MOCK_ROOT/stage4-kernel-opt` — **does not exist on this cluster** |

> **So reaching `packup` this hold requires building a local
> `stage4-kernel-opt` corpus out of the reverse payload, and then running with
> `mock_stages=m4` and `mock_root` pointed at it.** That is a buildable thing and
> it is the missing piece; it is not a `--var` we already have. **Second
> disagreement with the brief**, which lists `mock_stages` as if a value existed
> that gets us through.

**Consequence: the line exists in two forms and they are separated by work, not
by a flag.**

---

## 1. LINE A — modules 1→3 real, stop before m4

This is what is runnable **today**, and it is the line that produces the
`operator_workset` the payload needs. It is not a reduced version of Line B; it
is its precondition.

```sh
cd /home/yihou/dev/git.16-19/infera
export PYTHONPATH=$PWD/agent_sys          # or you run a different worktree

python3 assets/lib/run_with_long_stall.py --stall-after 900 -- \
python3 -m agent_sys.cli.main run \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /data/yihou/agent_sys_runroot \
  --var transport=local \
  --var jobid=29184 \
  --var node=smci355-ccs-aus-n04-25 \
  --var node_ip=10.235.192.131 \
  --var image=infera/engine-sglang:qwen3-local-20260906 \
  --var model_name=Qwen3-32B \
  --var model_path=/apps/data/models/Qwen3-32B \
  --var tp=4 \
  --var expect_ranks=4 \
  --var context_length=40960 \
  --var dsa_args=none \
  --var parser_args=none \
  --var work_root=/data/yihou/e2e_flow \
  --var scratch_root=/data/yihou/e2e_flow/kfo \
  --var magpie_root=/data/yihou/Magpie \
  --var trace_end_ms=60000 \
  --var bench_rounds=3 \
  --var adhoc_cases=3 \
  --var measure_gpu=4 \
  --var eval_thinking=none \
  --var mock_stages=none
```

**`gpu_devices` deliberately absent** — it is inert on m5's path
(`mix_worker.sh:26` `GPUS="${GPUS:-$(seq -s, 0 $((TP-1)))}"`, set nowhere), and
with `tp=4` the arms take cards 0-3 regardless. Passing it would assert control
we do not have.

### 1a. The six variables with no default — derived twice, because both
instruments were wrong in different ways

| instrument | its failure |
|---|---|
| `agent-sys show`, iteratively supplying what it names | **truncates**: *"4 other errors were produced"*. It stopped naming new ones while `model_path` was still unsupplied. |
| `grep -ohE '\$\{[a-z_]+\}'` over the two declaration files | **hits prose**: `mode`, `result_type`, `packup_skill` are comment text, and `tp` is really `${tp:-8}` |

**Neither alone is right. The intersection, after reading each hit:**

> **`image`, `jobid`, `model_name`, `model_path`, `node`, `node_ip`** — six, no
> defaults, the run will not load without them.

Third §5 instance in one afternoon, and the sharpest: **the loader's answer was
about the errors it chose to print, and the grep's was about a file set that
includes prose.** I would have shipped a five-variable list from either one.

### 1b. Defaults that are wrong for this cluster and must be overridden

| var | default | why it must be passed |
|---|---|---|
| `transport` | `auto` | the probe never selects `local` |
| `tp` | **8** | this hold is TP4 |
| `context_length` | **262144** | Qwen3-32B at 40960 |
| `mock_stages` | **`all`** | §0 — every stage would die on a missing corpus |
| `work_root` / `scratch_root` | `/mnt/m2m_nobackup/...` | not this cluster; and see P4 |
| `magpie_root` | `/shared_nfs/chaox/Magpie` | absent here; **m2's blocker, hard `exit 1`** |
| `eval_thinking` | `--thinking-mode glm-45` | Qwen3 scores 0.00 while every request succeeds |
| `bench_rounds` | 1 | leader's call; `check_no_regression`'s floors need 3 |
| `adhoc_cases` | 3 | already right — **do not lower it**, it is the never-exercised arm |

---

## 2. THE GATE between A and B — not a checklist item, a blocking step

**Line B may not be typed until this exits 0:**

```sh
python3 /data/yihou/e2e_verify_20260906/m35/m3_extract.py <the operator_workset content dir>
```

Exit 1 = STOP and report. It is the only thing that catches, on the login node,
a workset that `check_workset_shape` **admits** and `apply.py:808` refuses after
a bring-up: `call_site_fragment` + `public_symbol: null` passes the validator,
and an absent `substitution` returns silently.

Then, and only then:

```sh
python3 /data/yihou/e2e_verify_20260906/m35/mk_reverse_payload.py \
  --image infera/engine-sglang:qwen3-local-20260906 \
  --container-path <from the extraction> \
  --operator <from the extraction> \
  --function <from the extraction> \
  --delegate-to <from the extraction> \
  --out /data/yihou/e2e_verify_20260906/m35/payload.<op>
```

and the corpus build of §0a, which does not exist yet and is the open work item.

---

## 3. LINE B — the full chain to `packup`

Line A's flags, with two changes:

```
  --var mock_stages=m4                       # was: none
  --var mock_root=/data/yihou/mock_corpus    # NEW — the locally built stage4 corpus
```

**`mock_stages` must contain `m4` and nothing else.** Not `none` (m4 has no real
route), and **never `m5`**: `mock_m5.sh:332` `exit 1`s on the absent
`$E2E_PACKUP_MOCK` corpus and `packup.task/entry.sh:13` rethrows it, so
`packup.py` is never reached. A mocked m5 cannot reach packup on this cluster at
all.

---

## 4. Preconditions ON the line, because a line that can be typed without them is a line someone will type

Ordered; each is a stop.

1. **P0 stale tree** — `GENERATED.txt`'s `source_commit` == `git rev-parse HEAD`.
2. **P1 live chain** — `sh assets/lib/lines.sh`. This one can veto; P2 cannot.
3. **P2 cards + containers on the host** — and **STOP if any container we did not
   create holds a GPU.** `reset_gpus.sh` is node-wide, `KILLABLE_RE` covers
   `python3|pt_main_thread|ray|sglang.*`, `PROTECTED_RE` covers only schedulers,
   and `sudo -n true` → **rc 0 measured here**, so the sudo fallback reaches root
   processes inside other containers.
4. **P2b** — while m3's `check_workset_runs` measures, **nothing else of ours
   touches a GPU**; `max_rsd` 0.10 fails on a busy node and that refusal reads as
   a defect in m3's artefact.
5. **P4 redact** — `scratch_root` **must be under** `work_root`, or `redact.py`
   under `check=True` refuses at `packup.py:528` and `e2e_packup` (`is_end`) is
   never written **after the whole chain has run**.
6. **P6** — write this exact command line into the run directory. It cannot be
   recovered from the artefact; the staged package keeps `${var:-default}`
   unrendered.

**Cost against the 14:00Z hold:** `check_workset_runs` (m3, `gpu_hours`) plus
`check_speedup_substantiated` (m4, **60-minute hard ceiling**, two entrypoint
runs at `timeout_seconds: 1800`), plus two m5 bring-ups.

---

## 5. What I could not derive and am not guessing

- **The corpus build of §0a.** I know what it must contain
  (`stage4-kernel-opt/kernel_optimization/content/…` such that
  `mock.sh` copies it and `optimize_kernel.task/mock_adapt.py` can render `apply`
  and `premise` from the staged workset). I have not built or tested it, and
  whether `mock_adapt.py` accepts a corpus that is not the 2026-09-02 seal is
  **unverified**.
- **The delegation signature.** Only m3's Definition closes it, and a mismatch
  arrives as an empty measurement rather than an error.
