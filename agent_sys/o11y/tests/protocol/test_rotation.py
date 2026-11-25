"""Rotation of append-only operational files (user ruling 2026-11-24): past the cap a file is renamed to
`<name>.1` under the append lock (replacing the previous `.1`), so a file and its `.1` stay within about twice
the cap; nothing written is lost between them. The run ledger and launches never rotate."""

import json
import multiprocessing as mp

import protocol_paths  # noqa: F401  (sys.path)
from o11y_protocol import store


def _lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_rotates_past_the_cap_and_replaces_the_previous_copy(tmp_path):
    p = tmp_path / "events.jsonl"
    old = p.with_name("events.jsonl.1")
    rec = {"pad": "x" * 80}  # ~90 bytes a line
    for i in range(10):
        store.append(p, {**rec, "i": i}, rotate_bytes=400)
    assert old.exists() and p.stat().st_size <= 400 and old.stat().st_size <= 400
    got = [r["i"] for r in _lines(old) + _lines(p)]
    assert got == list(
        range(got[0], 10)
    )  # contiguous: the newest records, nothing lost between the two
    first = _lines(old)[0]["i"]
    for i in range(10, 20):
        store.append(p, {**rec, "i": i}, rotate_bytes=400)
    assert _lines(old)[0]["i"] > first  # the previous .1 was replaced, never accumulated
    assert sorted(x.name for x in tmp_path.iterdir()) == ["events.jsonl", "events.jsonl.1"]


def test_no_cap_never_rotates(tmp_path):
    p = tmp_path / "runs.jsonl"
    for i in range(50):
        store.append(p, {"i": i, "pad": "y" * 100})
    assert len(_lines(p)) == 50 and not p.with_name("runs.jsonl.1").exists()


def _writer(path, n, tag):
    for i in range(n):
        store.append(path, {"w": tag, "i": i, "pad": "z" * 40}, rotate_bytes=6000)


def test_concurrent_writers_lose_nothing_across_one_rotation(tmp_path):
    p = tmp_path / "events.jsonl"
    procs = [
        mp.Process(target=_writer, args=(p, 40, t)) for t in range(4)
    ]  # ~10 KB in total: one rotation
    for x in procs:
        x.start()
    for x in procs:
        x.join()
    rows = _lines(p.with_name("events.jsonl.1")) + _lines(p)
    assert len(rows) == 160 and len({(r["w"], r["i"]) for r in rows}) == 160
    assert p.with_name("events.jsonl.1").exists()
