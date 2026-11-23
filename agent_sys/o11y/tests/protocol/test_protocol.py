"""The protocol layer (o11y_protocol): prefix and slug drift, fleet root and pin, schemas and their shared
definitions, the event catalogue, bundles. Not collected by CI; run `python -m pytest agent_sys/o11y/tests/protocol`."""

import json
import re
import sys

import pytest
from o11y_protocol import fleet, schemas, store
from protocol_paths import AGENT_SYS, FLEET_DIR_VECTORS, PROTOCOL, SLUG_VECTORS


def _env_mgr_prefix():
    sys.path.insert(0, str(AGENT_SYS))
    try:
        from env_mgr.prefix import Prefix
    finally:
        sys.path.remove(str(AGENT_SYS))
    return Prefix


@pytest.mark.parametrize("case", ["home", "override", "tilde", "relative", "no_home", "symlink"])
def test_prefix_matches_env_mgr(case, tmp_path, monkeypatch):
    Prefix = _env_mgr_prefix()
    real = tmp_path / "real_prefix"
    real.mkdir()
    link = tmp_path / "link_prefix"
    link.symlink_to(real)
    monkeypatch.chdir(tmp_path)
    environ = {
        "home": {"HOME": str(tmp_path / "h")},
        "override": {"HOME": "/x", "AGENT_SYS_HOME": str(real)},
        "tilde": {"HOME": str(tmp_path), "AGENT_SYS_HOME": "~/real_prefix"},
        "relative": {"HOME": "/x", "AGENT_SYS_HOME": "real_prefix"},
        "no_home": {},
        "symlink": {"HOME": "/x", "AGENT_SYS_HOME": str(link)},
    }[case]
    if case == "tilde":
        monkeypatch.setenv("HOME", str(tmp_path))  # expanduser reads the process env
    assert fleet.prefix(environ) == Prefix.resolve(environ).root
    if case == "symlink":
        assert (
            fleet.prefix(environ) == real
        )  # canonical: the mods cannot resolve symlinks (design §4)


def test_slug_vectors_shared_with_ts():
    doc = json.loads(SLUG_VECTORS.read_text())
    assert len(doc["cases"]) >= 8
    for c in doc["cases"]:
        assert fleet.slug(c["in"]) == c["out"], c["in"]
        assert len(fleet.slug(c["in"])) <= 213


def test_fleet_dir_resolution(tmp_path):
    env = {"HOME": "/h", "AGENT_SYS_HOME": str(tmp_path / "p"), "CLAUDE_PROJECT_DIR": "/w/proj.x"}
    assert fleet.fleet_dir(env) == tmp_path / "p" / "state" / "o11y" / "fleets" / "-w-proj-x"
    env["AGENT_SYS_O11Y_FLEET_DIR"] = str(tmp_path / "explicit")
    assert fleet.fleet_dir(env) == tmp_path / "explicit"


@pytest.mark.parametrize("case", json.loads(FLEET_DIR_VECTORS.read_text())["cases"])
def test_fleet_dir_vectors_shared_with_ts(case, tmp_path, monkeypatch):
    """protocol/test_vectors/fleet_dir.json, also checked by tests/mods/protocol.test.ts."""
    env = dict(case["env"])
    monkeypatch.setenv("HOME", env["HOME"])  # Path.expanduser reads the process HOME
    if case["pin"]:  # the pin decides; its file lives under a writable prefix
        env["AGENT_SYS_HOME"] = str(tmp_path / "prefix")
        store.write_json(
            fleet.pin_path("s-vec", env),
            {
                "v": 1,
                "session_id": "s-vec",
                "fleet_dir": case["pin"],
                "role": "main",
                "pinned_at": store.now(),
            },
        )
    assert str(fleet.fleet_dir(env, launch=case["launch"], session_id="s-vec")) == case["out"]


def test_fleet_pin_comes_first(tmp_path):
    env = {
        "HOME": "/h",
        "AGENT_SYS_HOME": str(tmp_path / "p"),
        "CLAUDE_PROJECT_DIR": "/w/a",
        "AGENT_SYS_O11Y_FLEET_DIR": str(tmp_path / "explicit"),
    }
    assert fleet.fleet_dir(env, session_id="s1") == tmp_path / "explicit"  # no pin yet
    pin = fleet.pin_path("s1", env)
    store.write_json(
        pin,
        {
            "v": 1,
            "session_id": "s1",
            "fleet_dir": str(tmp_path / "pinned"),
            "role": "child",
            "pinned_at": store.now(),
        },
    )
    assert schemas.problems("session_pin.v1", store.read_json(pin)) == []
    assert fleet.fleet_dir(env, session_id="s1") == tmp_path / "pinned"
    assert fleet.pinned_role("s1", env) == "child"
    no_role = {k: v for k, v in store.read_json(pin).items() if k != "role"}
    assert any("'role'" in p for p in schemas.problems("session_pin.v1", no_role))
    pin.write_text("{torn")
    assert fleet.fleet_dir(env, session_id="s1") == tmp_path / "explicit"  # a torn pin is ignored


def test_schemas_share_common_definitions():
    for path in sorted((PROTOCOL / "schemas").glob("*.schema.json")):
        doc = json.loads(path.read_text())
        assert doc["$id"] == path.name
        if path.name != "common.v1.schema.json":  # one definition of time/id/writer (review C5)
            assert not {"time", "id", "writer"} & set(doc.get("$defs", {})), path.name
    good = {"v": 1, "log": {"seq": 3, "ts": store.now(), "reason": "timer"}}
    assert schemas.problems("requests.v1", good) == []
    assert schemas.problems("requests.v1", {"v": 1, "display": {"seq": 1, "ts": store.now()}})
    assert (
        schemas.problems(
            "otel_health.v1", {"v": 1, "ts": store.now(), "healthy": False, "outages": 2}
        )
        == []
    )
    assert "machine" not in schemas.load("runs.v1")["properties"]["kind"]["enum"]


def test_event_catalogue_is_well_formed():
    cat = schemas.events()
    pattern = schemas.load("event.v1")["properties"]["event_name"]["pattern"]

    for name, entry in cat.items():
        assert re.fullmatch(pattern, name), name
        assert entry["level"] in ("INFO", "WARN", "ERROR") and entry["level"] in entry.get(
            "levels", [entry["level"]]
        )
    for gone in ("o11y.session.started", "o11y.session.ended", "o11y.protocol.invalid_line"):
        assert gone not in cat  # review C4


def test_bundles_are_current_and_self_contained():
    """A schema handed to `claude --json-schema` is bundled; the committed bundle must equal a fresh one."""
    for name in schemas.BUNDLED:
        committed = json.loads((schemas.BUNDLE_DIR / f"{name}.bundle.json").read_text())
        assert committed == schemas.bundle(name), f"rerun: python -m o11y_protocol.schemas ({name})"
        assert "common.v1.schema.json" not in json.dumps(committed)


def test_append_cap_and_modes(tmp_path):
    path = tmp_path / "fleet" / "x" / "a.jsonl"
    store.append(path, {"a": 1})
    assert (path.stat().st_mode & 0o777) == 0o600 and (path.parent.stat().st_mode & 0o777) == 0o700
    with pytest.raises(store.LineTooLong):
        store.append(path, {"a": "x" * (65 * 1024)})
