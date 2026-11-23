"""agent_sys o11y protocol layer (protocol/py): schemas with cross-file refs and the event catalogue, validation,
locked JSONL append and atomic owner-only whole-file writes, fleet root and slug, the ownership rules for
files read outside the fleet root (owned). No whole operations: those are the
door's (`o11y_door.api`)."""

__version__ = "0.3.0"
