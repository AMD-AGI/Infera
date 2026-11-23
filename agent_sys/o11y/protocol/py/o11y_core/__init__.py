"""agent_sys o11y protocol library (protocol/py): schemas with cross-file refs, the event catalogue, fleet-root
resolution, locked JSONL append and whole-file writes, and the whole operations in `api`.

The door (agent-sys-o11y-door plugin) and the analyzers import only `api` (plus `store.read_jsonl` /
`store.write_json` and `schemas.problems` for their own files).
"""

__version__ = "0.3.0"
