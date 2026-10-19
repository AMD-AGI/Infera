# env_facts — measure this machine

A program task: `entry.sh` runs `collect.py`, which writes the `env_facts`
handoff into `$AGENT_SYS_OUTPUT_ENV_FACTS`. No model is involved.

Facts measured: `hostname`, `os`, `cpu_count`, `gpu_render_nodes` (the number of
`/dev/dri/renderD*` devices), `in_container` (`/.dockerenv` exists) and
`claude_cli` (a `claude` executable is on `PATH`).
