"""Analysis tests (o11y_job, o11y_sources, o11y_extract, o11y_log, o11y_facts, o11y-launch). Not collected by
CI (agent_sys's suites do not include agent_sys/o11y); run with `python -m pytest agent_sys/o11y/tests/analysis`."""

import analysis_paths  # noqa: F401  (sys.path as o11y-run sets it)
