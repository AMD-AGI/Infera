"""Test packages.

This file is load-bearing, not decorative. Each test directory carries an
`__init__.py` for the import-mode reason `task_graph/docs/design.md` §11 gives
— but without one *here*, pytest's basedir walk stops at `tests/`, inserts it
on `sys.path`, and imports `tests/handoff/__init__.py` under the top-level name
`handoff`, shadowing the package under test. With it, the basedir is
`agent_sys/` and every test module is `tests.<package>.<module>`.
"""
