# m2's launch guards and instruments — snapshot 2026-09-07T06:49:11Z

Copied out of `/data/yihou/e2e_verify_20260906/m2/` when the user halted all
tasks on 2026-09-07. **These are scratch tools, not package assets.** They are
here because the analysis in `bug.record` and `validator.failures` refers to
them and because several encode measured defects that took a run each to find.

**They carry absolute paths to that scratch directory and will not run unchanged
from the repo.** Read them for what they assert; re-point the paths before use.

## Instruments (each was run on a case whose answer was already known)

| file | what it answers | provenance / caveat |
|---|---|---|
| `deathwatch.sh` | has a run stopped being able to progress, and why | Reads `store/event`. **STRONG** = `output_absent`, `handling_failed`; **WEAK** = `validation_failed`, `escalated`. The weak split is measured: run 4 (`d9c7af`) carries **seven** `validation_failed` and sealed `_on` green, so the first version of this file declared the best run of the night dead. Controls: run 4 → weak only and silent under `--quiet`; run 10 → STRONG with the `seal_refused` line. **Blind to a wedged engine** — that emits none of these. |
| `capture_engine_stack.sh` | Python thread stacks of a wedged sglang engine | `py-spy` is already in the image at `/opt/venv/bin/py-spy`. Needs `--pid=host --privileged --user 0`; `--cap-add SYS_PTRACE` alone returns Permission denied (`yama/ptrace_scope=1`). Verified against a host process with a planted frame name. **Never used against a real wedge** — run 6 was the only one and the instrument did not exist yet. |
| `artefact_report.sh` | zone file counts **before** any verdict, then verdicts with mtimes | Every report this round led with the counts. No zero-file zone was seen in eleven runs; the upstream empty-zone reassurance was published and then reverted, so keep the count first. |
| `probe_two_windows.sh` | is a second profiler window fatal, and is it `with_stack` or the second session | Written for run 6's engine wedge. **Not run** — the wedge turned out to be 1 in 6, and one trial per arm cannot separate a coin flip. |

## Launch path

| file | what it encodes |
|---|---|
| `do_launch.sh` | Guards, each bought by a lost run: `--var` count asserted; `jobid` must be RUNNING (a stale id **agrees with itself** through `_agree_or_die`); `--timeout` asserted and run-only; `--stall-after` refuses **900** by name (it aliases AIPerf's request timeout, and a dead engine's timeout bursts re-feed the detector); **empty-default `--var`s absent from the line are printed, and `gpu` aborts** — it is the only one with a demonstrated abort. |
| `launch_chain.py` | Refuses if `PYTHONPATH` is set; prints where each module resolves. An editable install points at another worktree. |
| `LAUNCH-CHAIN.md` | The 34-var line as last launched (run 11). `do_launch.sh` parses the fenced block. |
| `VAR-TABLE.md` | The variable audit **and its correction**: it missed `gpu`, because `measure_gpu` was present and made it look covered. Ends with the two-command sweep that would have caught it. |
| `graft_kit.sh` | Corpus grafting; 10 controls. |
| `PRE-REGISTER-m2.md` | m2's validators, written before any result existed. |
| `launch-records/` | One per launch. Each states what the run would and would not establish **before** it ran — including that a stall value gets no credit unless it survives a long quiet stage, and the `apply_patch` prediction. |

## The one thing that never ran

`apply_patch` has never executed in this effort. The prediction is in
`bug.record` (`1d71a809`, revised after the brief edit) and in
`launch-records/launch15`. **It costs seconds and starts no container.**
