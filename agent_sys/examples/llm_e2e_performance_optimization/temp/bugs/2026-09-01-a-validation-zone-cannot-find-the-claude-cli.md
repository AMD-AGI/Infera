# A validation zone cannot find the `claude` CLI, and the body blames the handoff

**Found:** 2026-09-01, run `qwen-r1` on `crsuse2-m2m-020`.
**Severity:** a **false FAIL on a good handoff** — the worst verdict a validator
can emit, and it is silent about the real cause.
**Status:** worked around in the package. The underlying gap is `agent_sys`'s and
is reported here rather than fixed.

## What happened

`check_deploy_kit` PASSed. `check_deploy_reproduces` FAILed. In the validation
zone there was **no `claude.log` and no `reproduction.json`** — every other path
through `run_reproducer` writes a transcript, so the body had returned at its
first branch, `claude_command() is None`, before the reproducer ever started.

The handoff was fine. Its producer had even cold-tested it: `ps` on the node
caught the agent copying `$AGENT_SYS_OUTPUT_DEPLOY_KIT` to a scratch directory
and running `deploy_all.sh` from it, ending `teardown complete`.

## Why

A validation zone is given `env_mgr`'s policy-derived `PATH` and an **empty
`home/` of its own** (`validator/environment.py` builds `HOME=<zone>/home`), so:

- `shutil.which("claude")` misses — `~/.local/bin` is not on that `PATH`;
- `$HOME/.local/bin/claude` misses — that `home/` was created empty, measured.

## The gap: there is no published route

`AGENT_SYS_CLAUDE_CLI` looks like the answer and is not. It is a name `env_mgr`
**never published**: `agent/backends/claude_sdk.py:303` records that the first
version read `environment["AGENT_SYS_CLAUDE_CLI"]`, that the report is
`Prepared.agent_cli` instead — *declared* by the `Context` rather than
discovered — and that the unit test meant to pin the seam asserted the literal
against itself, so it stayed green while every prepared AI run refused.

`Prepared.agent_cli` reaches an **agent** through `Assignment.agent_cli`. A
validator body is not an agent, gets no `Assignment`, and has no equivalent. So
a validator that shells out to `claude` — which is how both "AI validators" in
this repository are actually written — has no supported way to find the binary
the run itself already located.

## What the package does instead

The path travels the same route as every other site fact: a package variable.

```yaml
# steps/deploy.yaml, check_deploy_reproduces
args:
  claude_cli: '${claude_cli:-}'
```

```bash
--var claude_cli="$(command -v claude)"
```

`PATH` and `~/.local/bin` are kept as fallbacks, so a site where the binary is
on the policy `PATH` declares nothing. The refusal message was also rewritten to
say what it now knows:

> no claude binary: the package declared none (`--var claude_cli=`), it is not
> on this zone's PATH, and $HOME/.local/bin/claude does not exist. **This is the
> environment, not the handoff.**

## What would fix it upstream

Either export the resolved CLI into the validation configuration the way
`AGENT_SYS_TASK_PACKAGE` is exported, or state in `validator`'s spec that a body
needing an external binary must be given its path through `args`. Today the
first is absent and the second is unwritten, so every package that writes an AI
validator will rediscover this.

**Related, and worth stating separately:** a validator body that cannot run *at
all* currently reports the same shape of failure as a handoff that is genuinely
bad. `ValidatorInvalid` exists for a crashed body (`validator/phase.py:451-492`)
and distinguishes "no judgement" from "judged false" — but a body that fails
gracefully, as this one did, lands in the second bucket. The package's
workaround is a clearer message; the distinction itself is `validator`'s.
