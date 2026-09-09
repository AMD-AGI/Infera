# `packup` — what it needs, what judges it, and what could waste a whole chain

**Written 2026-09-07T06:00:09Z by m35 at the leader's request. A read only: nothing touched,
nothing launched, `20260907T045327-13a18e` untouched.**

**Status: `e2e_packup` has 22 handoff records across every run on this host and
NOT ONE has ever left `created`. It has never been produced anywhere.**

---

## 1. What it consumes — eight inputs, and the two that matter

`steps/m5_integration.yaml:649-677`:

```
deploy_kit  profiling_evidence  operator_workset  kernel_optimization
patch_overlay  stock.measurement  patched.measurement  integration_report
```

**Five exist tonight. Three (`patch_overlay`, `stock.measurement`,
`patched.measurement`) plus `integration_report` are m5's, so `packup` cannot
start before m5 seals** — which the graph already enforces
(`froms: [apply_patch, integrate_and_verify]`).

**The one from a mocked stage: `kernel_optimization`, under `forge_mock=1`.**
Acceptable for reaching `packup` — but see §3, because it is *also* the input
with the largest redact exposure, and mocking it does not reduce that.

## 2. What judges it — two validators, neither has ever run

`e2e_packup` declares `validators: [check_environment, check_packup_shape]`
(`m5_integration.yaml:579`). The `packup` closure's own list is `validators: []`.

| validator | history on this host |
|---|---|
| `check_environment` | run many times on other kinds; **never on `e2e_packup`** |
| `check_packup_shape` | **never run at all** |

> **Both are therefore in the class where a zero-file zone is indistinguishable
> from a real defect**, and the empty-zone reassurance was published and then
> reverted upstream, so the defect ships. **`bash assets/lib/refusal_saw_something.sh <run>`
> before attributing anything either of them says. This is the case it exists for.**

## 3. Is anything unpassable by construction? — No for `check_environment`; the real hazard is `redact`

### `check_environment`: NOT the `compare.py` shape. It is tier 1 here.

`assets/packup.task/entry.sh` ends with an unconditional `exec`:

```sh
exec python3 "$PKG/assets/lib/env_render.py" \
  --inherit "$AGENT_SYS_INPUT_DEPLOY_KIT/items/codes/environment.yaml" \
  --content-type code --out "$AGENT_SYS_OUTPUT_E2E_PACKUP"
```

**No flag, no agent judgement, no `if`.** That is the opposite of `compare.py`,
where the same record is written only under `if args.environment:` and an agent
must remember the flag.

**Two caveats, and the second is a real trap:**

- **It is `--inherit`, so a pass proves the copy succeeded, not that two
  independent sources agree.** One source split in two.
- **`packup`'s program-ness is not guaranteed.** `packup` declares
  `agent: '${m5_agent:-runner}'` while `integrate_and_verify` declares
  `agent: '${m5_agent:-e2e_integrator}'` — **the same variable with different
  defaults.** Setting `--var m5_agent=<any agent name>` silently converts
  `packup` from a program body into an `kind: ai` body, and **a `kind: ai` task
  does not run `entry.sh`** (the yaml says so in its own comment). **The
  environment record would then never be written and `check_environment` would
  refuse the last artefact of the chain.** Do not set `m5_agent`.

### The real hazard: `redact.py` under `check=True`, and it is a hard stop

`packup.py:529` runs `redact.py` with `check=True`. **Any absolute path it
cannot name, in a `.py`/`.sh`/`.json`/`.jsonl` file, exits 1 → the exception
propagates → `packup.py` dies → `e2e_packup` (`is_end: true`) is never written.**
The comment at `:505` records this happening before: *"Passing only the zone's
four left 16 unnameable paths … so `e2e_packup` was never written."*

**The prefixes it will hold** (`packup.py:511-527`): `TASK_PACKAGE`, `ZONE`,
`TMPDIR=/tmp`, `HOME`, plus `MODEL_MOUNT` (parent of `E2E_MODEL_PATH`),
`WORK_ROOT`, `MOCK_ROOT` — the last three only when set and absolute.
Plus `redact.ALLOWED_PREFIXES`: `/usr/ /bin/ /sbin/ /lib/ /lib64/ /etc/ /opt/
/proc/ /sys/ /dev/ /var/lib/ /var/log/ /run/ /srv/ /workspace/ /app/`.

**`MAGPIE_ROOT` is NOT among them.** Verified by calling `redact.offenders()`
after `redact.substitute()` with tonight's real prefixes:

```
ok      work_root: /data/yihou/e2e_flow11/kfo
ok      model: /apps/data/models/Qwen3-32B
ok      "conclusion": ... /dev/md0 ...
REFUSE  magpie: /data/yihou/Magpie/tools      -> unnameable: /data/yihou/Magpie/tools
REFUSE  # ... `/shared_nfs` and `/mnt/m2m_nobackup`  -> unnameable: /mnt/m2m_nobackup
```

**`/shared_nfs` alone is safe** — `CANDIDATE` needs two path segments.

### Where the exposure actually is — and a false alarm I caught on myself

**I first read `packup.py:501`'s comment — *"A packup carries eight upstream
handoffs verbatim"* — found `/mnt/m2m_nobackup` twice in `deploy_kit`'s
`scripts/env.sh`, and was about to report a hard stop already present in the
live run.** Then I read what the code copies instead of what the comment says:

```
from deploy_kit  :133  ONE items/codes/**/README.md   (.md -> not a REFUSE suffix)
                 :156  items/logs/*                   (deploy_kit has NO items/logs)
```

**So `env.sh` never travels, and the two `/mnt/m2m_nobackup` comments are
harmless.** The prose said verbatim; the code is selective, and I believed the
prose. **Scan the artefact, not the sentence describing it.**

**The genuine exposure, in order:**

1. **`kernel_optimization/items/codes` — `shutil.copytree`, wholesale
   (`packup.py:180`).** A whole tree of `.py`, and under `forge_mock=1` it is a
   mocked artefact whose paths nobody has checked. **This is the one to scan the
   moment m4 seals.**
2. **`patch_overlay` / `stock.measurement` / `patched.measurement`'s
   `items/command`, copied as `.sh` (`packup.py:172-177`).** m5's, none exist yet.
3. Assorted `results/*.json`.

**A free check, runnable the instant m4 or m5 seals and before `packup` starts:**

```sh
R=<run dir>
grep -rhoE '/(data|apps|mnt|scratch|shared_nfs)/[A-Za-z0-9._/-]+' \
  --include='*.py' --include='*.sh' --include='*.json' --include='*.jsonl' \
  $R/handoffs | sed -E 's|^(/[^/]+/[^/]+/[^/]+).*|\1|' | sort | uniq -c | sort -rn
```

Anything not under `WORK_ROOT`, `MODEL_MOUNT`, `MOCK_ROOT` or
`ALLOWED_PREFIXES` will stop the chain at its last rung. **Tonight's sealed five
are clean: four `/data/yihou/e2e_flow11`, two `/apps/data/models`, and the three
`/mnt/m2m_nobackup` in a file that does not travel.**

## 4. What it costs — minutes, and nothing on the GPU

**`packup.py` and `entry.sh` contain no `docker run`, no `mix_up`, no
`reset_gpus`, no `rocm-smi`, no `HIP_VISIBLE_DEVICES`** — grep returns nothing
for all of them. **It is file assembly plus one `redact.py` pass.**

> **So unlike m5, `packup` does not own the host and cannot hurt a co-tenant.
> If the chain reaches it, the last rung is cheap.**

## 5. What I could not establish

- **How long it takes.** No run has produced `e2e_packup`, so there is no
  measured duration. The reading that would answer it is the first successful
  `packup`'s task timestamps in `store/task`.
- **Whether `check_packup_shape` passes on a real kit.** It has never executed.
  Its requirements are in `assets/check_packup_shape.validator/check.py` and it
  grades against the `experiment-result-packup` skill's `deliverable_layout.md`
  — **I read its structure, not its verdict, and those are different things.**
- **Whether a `forge_mock=1` `kernel_optimization` satisfies it.** Unknown, and
  it is downstream of `apply_patch` accepting the artefact at all, which m2 has
  pre-registered as the next open question.
