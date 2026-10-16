# check_env_facts — completeness, strong

Passes when `items/text.json` is a JSON object carrying every fact key with a
value of the expected type: `hostname` and `os` strings, `cpu_count` and
`gpu_render_nodes` integers, `in_container` and `claude_cli` booleans.
