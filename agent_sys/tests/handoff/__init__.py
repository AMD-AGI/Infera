"""Tests for `handoff`.

An `__init__.py` for the import-mode reason `task_graph/docs/design.md` §11
gives: without it, two test modules of the same basename in different
directories collide under pytest's default rootdir-relative import.
"""
