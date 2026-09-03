---
name: envchk-probe
description: Use when asked to report the env_checker **skill** capability token. Carries the skill capability's salt and the one command that turns it into this run's token. Trigger on "skill token", "capability 1", "envchk skill", or the env_checker brief's section 1.
---

# The skill capability's token

This file is the whole of the skill capability in `examples/env_checker`. It
reached the session because the agent spec's asset directory carries
`.claude/skills/envchk-probe/`, `env_mgr` copied that tree into
`$CLAUDE_CONFIG_DIR/skills/`, and Claude Code reads skills from there. Nothing
else in the package names it.

The salt below exists **only** in this file. Compute the token and report it:

```sh
python3 - <<'PY'
import hashlib, os
salt, label = "141016c5de297f4ec6ddac2809856ba1", "skill"
digest = hashlib.sha256(f"{salt}:{label}:{os.environ.get('ENVCHK_NONCE','')}".encode()).hexdigest()[:12]
print(f"ENVCHK-{label.upper()}-{digest}")
PY
```

<!-- ENVCHK_SALT: 141016c5de297f4ec6ddac2809856ba1 -->

Report it under `capabilities.skill` with `"level": "L3"` and, in `how`, the
fact that you invoked this skill by name rather than opening the file — the
brief's section 1 says which of the two you are being asked to do, and the two
are not the same evidence.

**`$ENVCHK_NONCE` must be set in your environment.** If it is not, the command
above still prints a well-formed token and it is the wrong one; say so in `how`
rather than reporting a token you could not derive, because the validator
recomputes it and a mismatch with no explanation is the least useful failure
available.
