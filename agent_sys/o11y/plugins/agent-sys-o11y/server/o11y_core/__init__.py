"""agent_sys o11y protocol core: validate, append (flock), query, registry readers, config with defaults.

Public API: `api.Context`, `api.ledger_append`, `api.ledger_query`, `api.read_ledger`, `api.event_report`,
`api.session_register`, `api.fleet_status`, `api.load_config`, `api.validate_file`; low level in `store`,
`schemas`, `claude_sessions`.
"""

__version__ = "0.1.0"
