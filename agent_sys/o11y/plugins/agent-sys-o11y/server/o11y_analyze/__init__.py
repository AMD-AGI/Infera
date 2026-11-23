"""o11y_analyze: one-shot analysis CLI (subsystem A) of agent_sys o11y.

Reads transcripts, the session registry, the machine pool, KernelForge campaign dirs and e2e-flow
files; writes events (logs/events.jsonl), extracted ledger records and facts/*.json under the fleet root ($AGENT_SYS_O11Y_FLEET_DIR).
It never imports the trigger or display subsystems; it talks to them only through protocol files.
"""

__version__ = "0.1.0"
