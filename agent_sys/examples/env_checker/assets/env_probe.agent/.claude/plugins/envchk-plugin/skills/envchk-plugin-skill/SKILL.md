---
name: envchk-plugin-skill
description: Use when asked to report the env_checker **plugin** capability token. Ships inside the envchk-plugin plugin, not in the agent's own skills directory. Trigger on "plugin token", "capability 3", "envchk plugin", or the env_checker brief's section 3.
---

# The plugin capability's token

This skill did **not** arrive the way `envchk-probe` did. It is inside a
plugin, and the plugin arrived because `env_mgr` ran, against the zone's
`CLAUDE_CONFIG_DIR`:

```sh
claude plugin marketplace add "$CLAUDE_CONFIG_DIR/plugins"
claude plugin install envchk-plugin@envchk-mp
```

Two routes, two salts, and reporting one for the other is the mistake this
separation is here to catch: a package that copies a `skills/` directory and
never installs a plugin looks identical from the outside until the two tokens
differ.

The salt below exists only in this file:

```sh
python3 - <<'PY'
import hashlib, os
salt, label = "fdf787965fb0679e22c3bf4c2988282d", "plugin"
digest = hashlib.sha256(f"{salt}:{label}:{os.environ.get('ENVCHK_NONCE','')}".encode()).hexdigest()[:12]
print(f"ENVCHK-{label.upper()}-{digest}")
PY
```

<!-- ENVCHK_SALT: fdf787965fb0679e22c3bf4c2988282d -->

Report it under `capabilities.plugin` with `"installed_by": "copied"`, and put in `how`
the output of `claude plugin list` — the install record is the part of this
capability that a file read cannot produce.
