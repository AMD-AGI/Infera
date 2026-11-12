---
name: infera-agent-sys
description: Use at the start of any task on large-model (LLM) deployment, serving or performance — sglang, vllm, ATOM, Infera, InferenceX, e2e optimisation, kernels and 算子 optimisation (Hyperloom Forge, KernelForge, AKLR, forge-loop, GEAK, Magpie, Triton, aiter, GEMM, attention, MoE), profiling and traces (TraceLens, rocprof), bench and load tests, agentic workloads, PD disaggregation, KV cache, speculative decoding, quantization (FP8/FP4), MI300X/MI355X — or anything that names agent_sys, a task package or a handoff. 大模型部署 / 推理服务 / 性能优化 / 算子优化 / 压测 / profiling / 加速比. Entry point of the infera agent_sys skills - first looks for prior tasks and handoffs in the shared knowledge repo, then routes the work. Respects explicit mode (engages only when agent_sys is named). Skip for a trivial factual question.
metadata:
  explicit_description: Use ONLY when the user explicitly names agent_sys (agent_sys, agent-sys, agentsys, infera agent sys, /infera-agent-sys) — e.g. use agent_sys to deploy, optimise or profile a model, find an agent_sys task or handoff. The user set explicit mode - do not use this skill for LLM deployment, performance, kernel or profiling requests that do not name agent_sys. Entry point of the infera agent_sys skills - looks for prior tasks and handoffs in the shared knowledge repo, then routes the work.
---

# infera-agent-sys — entry point

The team keeps its know-how in a shared knowledge repo as **agent_sys tasks**
(how to do a kind of work: steps, traps, the bar it must pass) and
**handoffs** (what a predecessor achieved: results, environment, scripts,
gotchas). This skill makes sure that knowledge is looked at **before** work
starts, and that new knowledge is offered back when work ends.

```bash
SKILL_ROOT="$(realpath "${CLAUDE_SKILL_DIR}/..")"
ASTK="$SKILL_ROOT/lib/astk.py"     # executable; one word, so `$ASTK cmd` works in bash and zsh
```

## 0. Should this skill engage at all?

```bash
$ASTK mode --prompt-file - <<'PROMPT'          # exit 0 = engage, 1 = step aside
<the user's request, verbatim>
PROMPT
```

Exit 1 (outside the domain, or explicit mode and agent_sys not named), or a
trivial question (a flag's meaning, a one-line fact): stop here and do the
work normally, without mentioning this skill. Any other exit code means astk
itself is broken: tell the user in one line (quote the error;
`sh $SKILL_ROOT/install.sh --check` diagnoses an install), then go on with
step 1 as far as it works.

## 1. Look for prior work

Invoke the `agent-sys-prior-work` skill (Skill tool) with the user's request,
saying that this entry skill invoked it. It syncs the knowledge repo (and the
local agent_sys examples), searches, reads and judges the candidates, and
returns a report; it offers repo setup once if none is configured. Show the
user the report before going further; the routing below is yours.

## 2. Route the work

First, if the report's judgement depends on facts the request leaves open
(model and size, TP/EP, engine and version, GPU, the metric that matters), ask
for exactly those — nothing else.

| situation | do |
|---|---|
| **e2e / agentic LLM serving optimisation** — throughput of a whole serving stack under an agentic or closed-loop multi-turn workload (AgentX, agent traces), PD disaggregation, a new model's PD bring-up | hand over to the `pd-agentx-tuning` skill (below); use the prior-work report as its input |
| the user only asked what exists | report, and stop; ask before starting any work |
| prior work shows little headroom (e.g. a measured ~1.0x) | measure the real opportunity first (profile the actual workload) and report before spending GPU hours |
| **kernel optimization with Forge / Hyperloom / AKLR** — prepare, run, inspect, steer, resume or finish a campaign | use `agent-sys-kernel-forge` (below); preserve a matching task's workset, validators and engine integration |
| a task fits the request | run it with `agent-sys-task-runner` (or follow its readme's STEPS by hand), seeding inputs from fitting handoffs |
| a handoff fits | start from it: its REPRODUCE / `items/command` reproduces the predecessor's result; its notes list the traps |
| nothing fits | do the work directly, using whatever the report said is partially useful |

**`agent-sys-kernel-forge`.** This skill ships beside the entry skill and is
installed by the same installer. Invoke it with the prior-work report and the
matching task/workset or AKLR schema. If it is not listed in this session,
read `$SKILL_ROOT/agent-sys-kernel-forge/SKILL.md` and follow it. It uses the
session-selected Hyperloom implementation and external AKLR KB references. A matching
agent_sys task still runs through `agent-sys-task-runner`; the Forge skill
governs its campaign, while the task retains its integration and validation.
Route broad serving optimization to the PD handbook first; this is the
kernel-campaign route once an operator has been selected.

**`pd-agentx-tuning`.** If it is not among your available skills, install it
from the knowledge repo, then invoke it with the Skill tool (if this session
does not list it yet, Read `<link>/SKILL.md` from the output and follow it):

```bash
$ASTK remote install-skill pd-agentx-tuning   # links ~/.claude/skills/pd-agentx-tuning into the repo clone
```

Tell the user the tier it came from (`from` in the output): a `draft` skill is
one person's unreviewed handbook. Exit 3 = no knowledge repo configured
(offer `$ASTK remote init`); any other failure: say so and continue without it.

Verify a predecessor's claim before building on it: `draft` is one person's
unreviewed result, an unsealed handoff never met its validators, `release` was
reviewed. A different GPU, image, engine version or model is a reason to
re-measure, not to copy a number.

## 3. When the work is done

Substantive work (a deployment that now works, a measured speedup, a
debugged failure) is knowledge. Offer, in one question: capture **how** to do
it as a task (`agent-sys-task-from-history`), or pack **what** was achieved as
a handoff (`agent-sys-handoff-packup`). Both end by asking whether to back the
result up to the knowledge repo; push nothing the user did not agree to.

## Modes

`$ASTK mode` shows the settings; `$ASTK mode explicit` (alias `off`) engages
only when the user names agent_sys, and re-renders this skill and
`agent-sys-prior-work` with a description that says so; `$ASTK mode auto`
(alias `on`) engages on the domain keywords (default); `$ASTK mode
--ask-remote yes` offers the knowledge-repo setup again.
