---
name: infera-agent-sys
description: Use at the start of any task on large-model (LLM) deployment, serving or performance — sglang, vllm, ATOM, Infera, InferenceX, e2e optimisation, kernels and 算子 optimisation (KernelForge, GEAK, Magpie, Triton, aiter, GEMM, attention, MoE), profiling and traces (TraceLens, rocprof), bench and load tests, agentic workloads, PD disaggregation, KV cache, speculative decoding, quantization (FP8/FP4), MI300X/MI355X — or anything that names agent_sys, a task package or a handoff. 大模型部署 / 推理服务 / 性能优化 / 算子优化 / 压测 / profiling / 加速比. Entry point of the infera agent_sys skills - first looks for prior tasks and handoffs in the shared knowledge repo, then routes the work. Respects explicit mode (engages only when agent_sys is named). Skip for a trivial factual question.
metadata:
  explicit_description: Use ONLY when the user explicitly names agent_sys (agent_sys, agent-sys, agentsys, infera agent sys, /infera-agent-sys) — e.g. use agent_sys to deploy, optimise or profile a model, find an agent_sys task or handoff. The user set explicit mode - do not use this skill for LLM deployment, performance, kernel or profiling requests that do not name agent_sys. Entry point of the infera agent_sys skills - looks for prior tasks and handoffs in the shared knowledge repo, then routes the work.
---

# infera-agent-sys — entry point

The infera team keeps its hard-won know-how as **agent_sys tasks** (how to do a
kind of work: steps, traps, the bar it must pass) and **handoffs** (what a
predecessor actually did: results, environment, scripts, gotchas) in a shared
knowledge repo. This skill makes sure that knowledge is looked at **before**
work starts, and that new knowledge is offered back when work ends.

```bash
SKILL_ROOT="$(realpath "${CLAUDE_SKILL_DIR}/..")"
ASTK="$SKILL_ROOT/lib/astk.py"     # executable; one word, so `$ASTK cmd` works in bash and zsh
DOCS="$SKILL_ROOT/docs"
```

## 0. Should this skill engage at all?

```bash
$ASTK mode --prompt-file - <<'PROMPT'          # exit 0 = engage, 1 = step aside
<the user's request, verbatim>
PROMPT
```

Exit 1: the request is outside the domain, or the user chose **explicit mode**
(`astk mode explicit`) and did not name agent_sys. Stop here and do the work
normally, without mentioning this skill. Also step aside for a trivial
question (a flag's meaning, a one-line fact) — the search costs more than it saves.

## 1. Look for prior work

Invoke the `agent-sys-prior-work` skill (Skill tool) with the user's request,
saying that this entry skill invoked it. It syncs the knowledge repo (and
indexes the local agent_sys examples), searches, reads the promising tasks and
handoffs, and returns a report of which may help and how. If the repo is not
configured it offers setup once (and remembers a "no").

Show the user the report before going further; the routing below is yours.

## 2. Route the work

First, if the report's judgement depends on facts the request leaves open
(model and size, TP/EP, engine and version, GPU, the metric that matters), ask
for exactly those — nothing else.

| situation | do |
|---|---|
| the user only asked what exists | report, and stop; ask before starting any work |
| prior work shows little headroom (e.g. a measured ~1.0x) | measure the real opportunity first (profile the actual workload) and report before spending GPU hours |
| a task fits the request | run it with `agent-sys-task-runner` (or follow its readme's STEPS by hand), seeding inputs from fitting handoffs |
| a handoff fits | start from it: its REPRODUCE / `items/command` reproduces the predecessor's result; its notes list the traps |
| nothing fits | do the work directly, using whatever the report said is partially useful |
| the user wants an agent_sys package run without the runtime | `agent-sys-task-runner` |

Whatever the route: verify a predecessor's claim before building on it — a
`draft` handoff is one person's unreviewed result, an unsealed one never met
its validators; `release` was reviewed. An environment that differs (GPU,
image, engine version, model) is a reason to re-measure, not to copy a number.

## 3. When the work is done

Substantive work (a deployment that now works, a measured speedup, a
debugged failure) is knowledge. Offer, in one question:

- capture **how** to do it as a task → `agent-sys-task-from-history`;
- pack **what** was achieved as a handoff → `agent-sys-handoff-packup`.

Both end by asking whether to back the result up to the knowledge repo
(`<user>/draft/`). Do not push anything the user did not agree to.

## Modes and settings

```bash
$ASTK mode                     # show: trigger auto|explicit, ask_remote yes|no
$ASTK mode explicit            # (alias: off) engage only when the user names agent_sys
$ASTK mode auto                # (alias: on) engage on the domain keywords (default)
$ASTK mode --ask-remote yes    # offer the knowledge-repo setup again
```

Explicit mode also re-renders this skill and `agent-sys-prior-work` with a
description that limits them to requests naming agent_sys (when installed by
`install.sh`), so Claude does not load them for generic requests.
