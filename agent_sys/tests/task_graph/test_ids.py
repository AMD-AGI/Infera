"""Typed identities — spec criterion 27.

Two ids of different kinds built from the same bytes must not be equal and must
not collide in one dict. ``uuid.UUID.__eq__`` compares ``self.int`` against any
``UUID``, so this only holds because ``_Id`` overrides it.
"""

import uuid

import pytest
from pydantic import BaseModel

from task_graph.ids import AgentId, HandoffId, TaskId

KINDS = [TaskId, AgentId, HandoffId]


def test_new_is_unique_and_typed():
    for kind in KINDS:
        a, b = kind.new(), kind.new()
        assert a != b
        assert isinstance(a, kind)
        assert isinstance(a, uuid.UUID)


def test_same_kind_same_bytes_is_equal():
    raw = uuid.uuid4()
    for kind in KINDS:
        assert kind(raw.hex) == kind(raw.hex)
        assert hash(kind(raw.hex)) == hash(kind(raw.hex))


def test_cross_kind_same_bytes_is_not_equal():
    raw = uuid.uuid4()
    for left in KINDS:
        for right in KINDS:
            if left is right:
                continue
            assert left(raw.hex) != right(raw.hex), f"{left.__name__} == {right.__name__}"


def test_cross_kind_does_not_collide_in_one_dict():
    raw = uuid.uuid4()
    d = {kind(raw.hex): kind.__name__ for kind in KINDS}
    assert len(d) == len(KINDS)
    for kind in KINDS:
        assert d[kind(raw.hex)] == kind.__name__


def test_a_plain_uuid_is_not_equal_to_a_typed_one():
    raw = uuid.uuid4()
    assert TaskId(raw.hex) != raw
    assert raw != TaskId(raw.hex)


def test_str_round_trip():
    for kind in KINDS:
        original = kind.new()
        assert kind(str(original)) == original


def test_repr_names_the_kind():
    tid = TaskId.new()
    assert repr(tid).startswith("TaskId(")
    assert str(tid) in repr(tid)


def test_malformed_value_is_rejected():
    with pytest.raises(ValueError):
        TaskId("not-a-uuid")


# ---- pydantic integration: a bare UUID subclass raises without the schema ----


class _Holder(BaseModel):
    tid: TaskId
    versions: dict[HandoffId, int] = {}


def test_pydantic_accepts_a_typed_id():
    tid = TaskId.new()
    assert _Holder(tid=tid).tid == tid


def test_pydantic_coerces_a_string():
    tid = TaskId.new()
    holder = _Holder(tid=str(tid))
    assert holder.tid == tid
    assert isinstance(holder.tid, TaskId)


def test_pydantic_coerces_a_plain_uuid():
    raw = uuid.uuid4()
    holder = _Holder(tid=raw)
    assert isinstance(holder.tid, TaskId)
    assert holder.tid == TaskId(raw.hex)


def test_json_mode_dumps_a_string_python_mode_keeps_the_object():
    tid = TaskId.new()
    holder = _Holder(tid=tid)
    assert holder.model_dump(mode="json")["tid"] == str(tid)
    assert holder.model_dump()["tid"] == tid


def test_round_trip_preserves_the_type_of_a_dict_key():
    hid = HandoffId.new()
    holder = _Holder(tid=TaskId.new(), versions={hid: 3})
    back = _Holder.model_validate(holder.model_dump(mode="json"))
    assert back.versions == {hid: 3}
    (restored_key,) = back.versions
    assert isinstance(restored_key, HandoffId)


def test_round_trip_survives_json_text():
    import json

    holder = _Holder(tid=TaskId.new(), versions={HandoffId.new(): 0})
    assert _Holder.model_validate(json.loads(holder.model_dump_json())) == holder


def test_the_types_are_statically_incompatible():
    """Criterion 27's other half — the checker's job, asserted here as the
    structural fact it rests on: three unrelated leaf classes."""
    for left in KINDS:
        for right in KINDS:
            if left is not right:
                assert not issubclass(left, right)
