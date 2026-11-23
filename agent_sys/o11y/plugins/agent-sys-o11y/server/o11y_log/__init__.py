"""o11y_log — the log analyzer (design §8.0): deterministic detectors plus the log model pass.

`python -m o11y_log` serves queue kind `log`. Output: level records in `logs/events.jsonl` and extracted
ledger records, both through o11y_core; private state in `state/log/`. It reads sources through
`o11y_sources` and never imports `o11y_facts`.
"""
