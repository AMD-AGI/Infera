# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One AgentsView project per run: the call, and every way it may fail.

AgentsView names a project after the session's **deepest** path segment, so one
run's attempts arrive as several unrelated projects — measured on a real nested
fixture, four sessions of one run as four projects. Renaming the directories
cannot join them; only a mapping over the run root can.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from env_mgr.o11y import mapping

MACHINE = "test-box"


class _Recorder:
    """A stand-in for `urlopen` that records requests and replays answers."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.seen: list[dict] = []

    def __call__(self, req, timeout=None):  # noqa: ANN001
        self.seen.append(
            {
                "url": req.full_url,
                "method": req.get_method(),
                "headers": {k.lower(): v for k, v in req.header_items()},
                "body": json.loads(req.data.decode()) if req.data else None,
            }
        )
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer)


class _Response:
    def __init__(self, payload: object, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self, *_a: object) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_a: object) -> None:
        return None


def _machine_answer() -> dict:
    return {"machine": MACHINE, "local_machine": MACHINE, "machines": [MACHINE], "mappings": []}


@pytest.fixture()
def urlopen(monkeypatch):
    def install(*answers: object) -> _Recorder:
        rec = _Recorder(*answers)
        monkeypatch.setattr(mapping.urllib.request, "urlopen", rec)
        return rec

    return install


# --- the name ------------------------------------------------------------- #


def test_a_dash_becomes_an_underscore_before_we_send_it() -> None:
    """Measured: AgentsView stores `MAPPED-GIT` as `MAPPED_GIT`.

    Normalising on our side means the string we post is the string that comes
    back, so a later read or a log line cannot disagree with the panel.
    """
    assert mapping.name_for_run(Path("/x/runs/20260903T104807-76274e")) == (
        "run.20260903T104807_76274e"
    )


def test_the_name_is_built_from_the_run_directory_alone() -> None:
    """Nothing upstream of the run root may change the label."""
    a = mapping.name_for_run(Path("/one/runs/20260903T104807-76274e"))
    b = mapping.name_for_run(Path("/another/place/runs/20260903T104807-76274e"))
    assert a == b


# --- the call ------------------------------------------------------------- #


def test_the_mapping_is_posted_for_this_run_only(urlopen) -> None:
    rec = urlopen(_machine_answer(), {"id": 1})
    run = Path("/state/runs/20260904T101112-abcdef")

    status = mapping.ensure_run_project("http://127.0.0.1:9001", run)

    assert status.running is True
    post = rec.seen[-1]
    assert post["method"] == "POST"
    assert post["body"] == {
        "machine": MACHINE,
        "path_prefix": str(run),
        "project": "run.20260904T101112_abcdef",
        "layout": "explicit",
        "enabled": True,
    }


def test_the_machine_is_read_from_the_daemon_and_never_assumed(urlopen) -> None:
    """A wrong `machine` matches nothing, silently.

    The recon ran in a container whose hostname was not the host's, which is
    exactly how this would have shipped broken: the value has to come from the
    daemon that will do the matching.
    """
    rec = urlopen(
        {"local_machine": "the-real-one", "machines": ["the-real-one"], "mappings": []},
        {"id": 1},
    )

    mapping.ensure_run_project("http://127.0.0.1:9001", Path("/state/runs/r-1"))

    assert rec.seen[0]["method"] == "GET"
    assert rec.seen[-1]["body"]["machine"] == "the-real-one"


def test_every_mutating_call_sends_an_origin_header(urlopen) -> None:
    """Measured: without it the answer is a plain-text `403 Forbidden`, not the
    JSON error shape — which reads exactly like a missing endpoint."""
    rec = urlopen(_machine_answer(), {"id": 1})

    mapping.ensure_run_project("http://127.0.0.1:9001", Path("/state/runs/r-1"))

    assert rec.seen[-1]["headers"]["origin"] == "http://127.0.0.1:9001"


def test_an_existing_mapping_is_success_not_a_warning(urlopen, caplog) -> None:
    """`POST` is not idempotent — uniqueness is `(machine, path_prefix)` — so a
    second run of the same id answers `409`. That is the state we wanted."""
    rec = urlopen(
        _machine_answer(),
        urllib.error.HTTPError("u", 409, "conflict", {}, None),  # type: ignore[arg-type]
    )

    with caplog.at_level("WARNING"):
        status = mapping.ensure_run_project("http://127.0.0.1:9001", Path("/state/runs/r-1"))

    assert status.running is True
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
    assert len(rec.seen) == 2


# --- every failure is one warning and a skip ------------------------------- #


@pytest.mark.parametrize(
    "answers",
    [
        (urllib.error.URLError("connection refused"),),
        (OSError("socket blew up"),),
        ("not-a-dict",),
        ({"machines": []},),  # no local_machine
        (_machine_answer(), urllib.error.HTTPError("u", 500, "boom", {}, None)),
        (_machine_answer(), urllib.error.URLError("died mid-post")),
        (_machine_answer(), TimeoutError("too slow")),
    ],
)
def test_a_failing_mapping_call_is_one_warning_and_a_skip(urlopen, caplog, answers) -> None:
    urlopen(*answers)

    with caplog.at_level("WARNING"):
        status = mapping.ensure_run_project("http://127.0.0.1:9001", Path("/state/runs/r-1"))

    assert status.running is False
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_no_panel_means_no_call_and_no_warning(urlopen, caplog) -> None:
    """`url=None` is "the panel did not start", which was already warned about."""
    rec = urlopen()

    with caplog.at_level("WARNING"):
        status = mapping.ensure_run_project(None, Path("/state/runs/r-1"))

    assert status.running is False
    assert rec.seen == []
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_nothing_escapes_regardless_of_what_the_daemon_says(urlopen) -> None:
    """The module's one law, asserted directly."""
    for answers in (
        (RuntimeError("something nobody thought of"),),
        (_machine_answer(), ValueError("garbage body")),
        (None,),
    ):
        urlopen(*answers)
        mapping.ensure_run_project("http://127.0.0.1:9001", Path("/state/runs/r-1"))
