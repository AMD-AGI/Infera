"""Criteria 3 and 5: the four content types, and runtime-generated keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from handoff import CONTENT_TYPES
from handoff import content as content_mod
from handoff.errors import Malformed


def _content(root: Path, items: dict[str, object], *, files: tuple[str, ...] = ()) -> Path:
    (root / content_mod.ITEMS_DIR).mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# H\n\n## Purpose\n\nWhy.\n", encoding="utf-8")
    for name in files:
        (root / content_mod.ITEMS_DIR / name).write_text("x\n", encoding="utf-8")
    if items:
        content_mod.write_items_json(root / content_mod.ITEMS_DIR, items)
    return root


def test_the_four_types_are_a_table() -> None:
    assert set(CONTENT_TYPES) == {"reproducible", "code", "structured_text", "text"}
    with pytest.raises(Malformed, match="unknown content_type"):
        content_mod.content_type("prose")


@pytest.mark.parametrize(
    ("type_name", "files", "data"),
    [
        ("reproducible", ("script", "result"), {"env": {"gpu": "MI300X"}}),
        ("code", ("codes",), {}),
        ("structured_text", ("text.json",), {}),
        ("text", ("content",), {}),
    ],
)
def test_four_types_accept_and_reject(
    tmp_path: Path, type_name: str, files: tuple[str, ...], data: dict
) -> None:
    """Criterion 3: each type accepts its declared items and rejects one it
    does not define."""
    ctype = content_mod.content_type(type_name)

    ok = content_mod.load(_content(tmp_path / "ok", data, files=files))
    content_mod.check_items(ok, ctype)

    bad = content_mod.load(_content(tmp_path / "bad", data, files=(*files, "sneak")))
    with pytest.raises(Malformed, match="not defined by content_type"):
        content_mod.check_items(bad, ctype)


def test_a_missing_required_item_is_rejected(tmp_path: Path) -> None:
    ctype = content_mod.content_type("text")
    empty = content_mod.load(_content(tmp_path / "e", {}))
    with pytest.raises(Malformed, match=r"requires \['content'\]"):
        content_mod.check_items(empty, ctype)


def test_one_of_script_or_command_is_required(tmp_path: Path) -> None:
    """`reproducible` needs `script` *or* `command`, which the frozen
    `ContentType` has no field for — so the alternatives are a second table."""
    ctype = content_mod.content_type("reproducible")
    neither = content_mod.load(_content(tmp_path / "n", {"env": {}}, files=("result",)))
    with pytest.raises(Malformed, match="requires one of"):
        content_mod.check_items(neither, ctype)

    with_command = content_mod.load(
        _content(tmp_path / "c", {"env": {}}, files=("result", "command"))
    )
    content_mod.check_items(with_command, ctype)


def test_runtime_key_follows_additional_properties(tmp_path: Path) -> None:
    """Criterion 5, and the mechanism is the one JSON Schema already has —
    no second mechanism and no `allow_runtime_keys` flag."""
    ctype = content_mod.content_type("text")
    loaded = content_mod.load(_content(tmp_path / "r", {"operator_0": 1}, files=("content",)))

    permissive = {
        "type": "object",
        "properties": {"content": {}},
        "additionalProperties": True,
    }
    content_mod.check_items(loaded, ctype, permissive)

    closed = {
        "type": "object",
        "properties": {"content": {}},
        "additionalProperties": False,
    }
    with pytest.raises(Malformed, match="additionalProperties|Additional properties"):
        content_mod.check_items(loaded, ctype, closed)


def test_a_key_present_as_both_a_file_and_data_is_malformed(tmp_path: Path) -> None:
    root = _content(tmp_path / "d", {"content": 1}, files=("content",))
    with pytest.raises(Malformed, match="has one source"):
        content_mod.load(root)


def test_a_directory_item_is_a_tree_and_a_file_is_a_file(tmp_path: Path) -> None:
    root = _content(tmp_path / "k", {}, files=("script",))
    (root / content_mod.ITEMS_DIR / "codes").mkdir()
    loaded = content_mod.load(root)
    assert loaded.items["codes"].kind == "tree"
    assert loaded.items["script"].kind == "file"


def test_readme_sections_are_ordered_for_a_template_and_checked_as_a_set() -> None:
    """markdownlint #394 is a required-headings matcher that accepted every
    document — it failed *open*. Ordered for templating, membership at the
    check."""
    ctype = content_mod.content_type("reproducible")
    assert isinstance(ctype.readme_sections, tuple)
    assert content_mod.required_sections(ctype, ("Caveats",))[-1] == "Caveats"
    assert content_mod.required_sections(ctype, ("Purpose",)) == ctype.readme_sections


def test_check_items_schema_gives_one_error_not_eight() -> None:
    """`$ref`-ing the 2020-12 metaschema turns `{"type": "nonsense"}` into
    eight identical errors, one per failing `anyOf` branch. `check_schema` is a
    named step for that reason."""
    with pytest.raises(Malformed) as exc:
        content_mod.check_items_schema({"type": "nonsense"}, origin="k.jsonnet")
    assert str(exc.value).count("is not valid") == 1


@pytest.mark.parametrize("not_an_object", ["notaschema", ["a"], 42, None, ("a", "b"), [("a", 1)]])
def test_a_non_object_items_schema_is_malformed(not_an_object) -> None:
    """The branch `closure`'s question found, driven.

    `check_items_schema` coerced with `dict(items_schema)` and caught
    `(TypeError, AttributeError)`. Nothing in the suite had ever taken that
    `except`, and driving it showed it was catching the wrong things:

    | input | before |
    |---|---|
    | `"notaschema"` — **main design §3.5's own example** | `dict()` raises `ValueError`, uncaught, so it escaped past every caller catching `Malformed` |
    | `[("a", 1)]` | `dict()` **succeeds**, so a list of pairs was accepted as a schema object |

    `isinstance` refuses both, and `Malformed` is what the contract promises.
    """
    with pytest.raises(Malformed, match="must be an object"):
        content_mod.check_items_schema(not_an_object, origin="k.jsonnet")


def test_a_valid_items_schema_still_passes() -> None:
    """The positive case, because a refusal test alone passes when the function
    refuses everything."""
    content_mod.check_items_schema({"type": "object", "properties": {"a": {}}}, origin="ok")
