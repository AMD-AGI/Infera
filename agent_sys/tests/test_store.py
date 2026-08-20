"""Persistence — criterion 29.

`create` and `update` are separate because their preconditions differ and both
are worth enforcing: `add` means *new*, `persist` means *existing*. An upsert
would accept both mistakes silently.

Both implementations are held to the same contract, so the JSON store is not a
second-class citizen tested only once.
"""

import json

import pytest

from agent_sys.store import JsonFileStoreMgr, MemoryStoreMgr, StoreMgr

RECORD = {"id": "abc", "status": "running", "nested": {"n": [1, 2]}}


@pytest.fixture(params=["memory", "json"])
def store(request, tmp_path) -> StoreMgr:
    if request.param == "memory":
        return MemoryStoreMgr()
    return JsonFileStoreMgr(tmp_path / "state")


# --------------------------------------------------------------- the contract


def test_create_then_read(store):
    store.create("task", "abc", RECORD)
    assert store.read("task", "abc") == RECORD


def test_read_a_missing_record_is_none_not_an_error(store):
    assert store.read("task", "nope") is None


def test_create_rejects_an_existing_key(store):
    store.create("task", "abc", RECORD)
    with pytest.raises(KeyError, match="abc"):
        store.create("task", "abc", RECORD)


def test_update_replaces(store):
    store.create("task", "abc", RECORD)
    store.update("task", "abc", {**RECORD, "status": "succeeded"})
    assert store.read("task", "abc")["status"] == "succeeded"


def test_update_rejects_a_missing_key(store):
    with pytest.raises(KeyError, match="abc"):
        store.update("task", "abc", RECORD)


def test_delete(store):
    store.create("task", "abc", RECORD)
    store.delete("task", "abc")
    assert store.read("task", "abc") is None
    assert not store.exists("task", "abc")


def test_delete_rejects_a_missing_key(store):
    with pytest.raises(KeyError, match="abc"):
        store.delete("task", "abc")


def test_exists(store):
    assert not store.exists("task", "abc")
    store.create("task", "abc", RECORD)
    assert store.exists("task", "abc")


def test_read_all(store):
    store.create("task", "a", {"id": "a"})
    store.create("task", "b", {"id": "b"})
    assert sorted(r["id"] for r in store.read_all("task")) == ["a", "b"]


def test_read_all_of_an_unknown_kind_is_empty(store):
    assert store.read_all("nothing-here") == []


def test_kinds_are_separate_namespaces(store):
    store.create("task", "same", {"which": "task"})
    store.create("handoff", "same", {"which": "handoff"})
    assert store.read("task", "same") == {"which": "task"}
    assert store.read("handoff", "same") == {"which": "handoff"}
    assert len(store.read_all("task")) == 1


def test_a_key_with_path_characters_survives(store):
    """Callers pass opaque strings and should not have to know they are paths."""
    key = "a/b:c d?e"
    store.create("task", key, RECORD)
    assert store.read("task", key) == RECORD
    assert store.exists("task", key)
    store.delete("task", key)
    assert not store.exists("task", key)


def test_a_stored_record_is_not_a_live_reference(store):
    """Otherwise 'reload from persistence' returns objects nobody ever wrote,
    and every recovery test passes vacuously."""
    record = {"id": "abc", "nested": {"n": [1]}}
    store.create("task", "abc", record)

    record["id"] = "mutated-after-write"
    record["nested"]["n"].append(2)
    assert store.read("task", "abc") == {"id": "abc", "nested": {"n": [1]}}

    first = store.read("task", "abc")
    first["nested"]["n"].append(3)
    assert store.read("task", "abc") == {"id": "abc", "nested": {"n": [1]}}


# ------------------------------------------------------- JSON store specifics


def test_json_store_writes_one_readable_file_per_record(tmp_path):
    store = JsonFileStoreMgr(tmp_path / "state")
    store.create("task", "abc", RECORD)

    path = tmp_path / "state" / "task" / "abc.json"
    assert json.loads(path.read_text()) == RECORD


def test_json_store_leaves_no_temporary_file_behind(tmp_path):
    """The write is <name>.json.tmp + Path.replace, which is atomic on POSIX."""
    store = JsonFileStoreMgr(tmp_path / "state")
    store.create("task", "abc", RECORD)
    store.update("task", "abc", RECORD)
    assert list((tmp_path / "state" / "task").glob("*.tmp")) == []


def test_json_store_survives_a_fresh_instance_over_the_same_root(tmp_path):
    JsonFileStoreMgr(tmp_path / "state").create("task", "abc", RECORD)
    assert JsonFileStoreMgr(tmp_path / "state").read("task", "abc") == RECORD


def test_json_store_ignores_a_non_json_file_in_a_kind_directory(tmp_path):
    store = JsonFileStoreMgr(tmp_path / "state")
    store.create("task", "abc", RECORD)
    (tmp_path / "state" / "task" / "README").write_text("not a record")
    assert store.read_all("task") == [RECORD]
