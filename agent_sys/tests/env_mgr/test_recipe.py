# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
import dataclasses
import textwrap
from pathlib import Path

import pytest

from env_mgr.recipe import Item, RecipeError, load_recipe
from env_mgr.runner import detect_conflicts

ROOT = Path(__file__).resolve().parents[2] / "env_mgr"


def _write(tmp_path, body):
    p = tmp_path / "recipe.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_minimal_recipe(tmp_path):
    p = _write(
        tmp_path,
        """
        version: 1
        target:
          kind: repo
          name: sglang
          path: /tmp/sglang
        items:
          - installer: uv
            importance: required
            ref: pyproject.toml
            version: ">=0.5"
            tags: [python, runtime]
    """,
    )
    target, items = load_recipe(p)
    assert target.kind == "repo"
    assert target.name == "sglang"
    assert len(items) == 1
    it = items[0]
    assert it.installer == "uv"
    assert it.importance == "required"
    assert it.version == ">=0.5"
    assert it.tags == ["python", "runtime"]
    assert it.spec["ref"] == "pyproject.toml"


def test_item_name_prefers_name_then_provides_then_installer(tmp_path):
    p = _write(
        tmp_path,
        """
        version: 1
        target: {kind: repo, name: x, path: /tmp/x}
        items:
          - installer: bin
            importance: suggested
            name: pyright-langserver
          - installer: uv
            importance: required
            provides: serena
          - installer: apt
            importance: suggested
            packages: [jq]
    """,
    )
    _, items = load_recipe(p)
    assert items[0].name == "pyright-langserver"
    assert items[1].name == "serena"
    assert items[2].name == "apt"


def test_item_name_falls_back_to_installer_when_provides_is_dict():
    # apt items use `provides` as a package->command dict (e.g. ripgrep -> rg),
    # not a name candidate. Item.name must not return that dict.
    it = Item(
        installer="apt",
        importance="suggested",
        spec={"packages": ["jq"], "provides": {"jq": "jq"}},
    )
    assert it.name == "apt"
    assert isinstance(it.name, str)
    assert {it.name: "usable as dict key"}  # must be hashable


def test_item_name_uses_provides_when_it_is_a_string():
    it = Item(
        installer="uv",
        importance="required",
        spec={"provides": "serena"},
    )
    assert it.name == "serena"


def test_detect_conflicts_does_not_crash_on_dict_provides():
    items = [
        Item(
            installer="apt",
            importance="suggested",
            spec={"packages": ["ripgrep"], "provides": {"ripgrep": "rg"}},
        ),
        Item(
            installer="apt",
            importance="suggested",
            spec={"packages": ["fd-find"], "provides": {"fd-find": "fdfind"}},
        ),
    ]
    assert detect_conflicts(items) == []


def test_missing_installer_raises(tmp_path):
    p = _write(
        tmp_path,
        """
        version: 1
        target: {kind: repo, name: x, path: /tmp/x}
        items:
          - importance: required
    """,
    )
    with pytest.raises(RecipeError, match="installer"):
        load_recipe(p)


def test_bad_importance_raises(tmp_path):
    p = _write(
        tmp_path,
        """
        version: 1
        target: {kind: repo, name: x, path: /tmp/x}
        items:
          - installer: uv
            importance: nice-to-have
    """,
    )
    with pytest.raises(RecipeError, match="importance"):
        load_recipe(p)


# ------------------------------------------------- the layer model is gone
#
# `layer` was a required, validated field on every item, and it is removed:
# where an installed thing lands is derived, not declared (`docs/spec.md` §9.1).
# These five tests are what replaces `test_bad_layer_raises` and
# `tests/env_mgr/test_layer.py`, both deleted with it. The first two are the
# removal; the third is the shipped artefacts; the last two are the migration
# guard, which exists because removing the key from an *exclusion* set would
# otherwise have turned a stale `layer:` into a silent spec key rather than an
# error.


def test_item_has_no_layer_field():
    fields = {f.name for f in dataclasses.fields(Item)}
    assert "layer" not in fields
    with pytest.raises(TypeError):
        Item(installer="uv", importance="required", layer="system")  # type: ignore[call-arg]


def test_item_without_a_layer_key_loads(tmp_path):
    # Decision B: the key is optional-and-gone, not optional-and-tolerated.
    # A recipe that never mentions it is the normal case now.
    p = _write(
        tmp_path,
        """
        version: 1
        target: {kind: repo, name: x, path: /tmp/x}
        items:
          - installer: uv
            importance: required
            ref: pyproject.toml
    """,
    )
    _, items = load_recipe(p)
    assert len(items) == 1
    assert items[0].spec == {"ref": "pyproject.toml"}


@pytest.mark.parametrize("name", ["serena.yaml", "sglang.repo.yaml"])
def test_the_shipped_recipes_carry_no_layer(name):
    # The two recipes this repository ships were the only two with `layer:`
    # keys. Read the file, not the diff: the key must be gone from the text,
    # and — because `_CLI_KEYS` is an *exclusion* list — it must also not have
    # reappeared inside `Item.spec`, which is where an unremoved key would go.
    path = ROOT / "recipes" / name
    assert "layer:" not in path.read_text()
    _, items = load_recipe(path)
    assert items
    assert not any("layer" in it.spec for it in items)


def test_a_stale_layer_key_is_rejected_and_says_where_the_concept_went(tmp_path):
    # `layer` was listed in `_CLI_KEYS`, which is the set of keys held *back*
    # from `Item.spec`. Deleting the name from an exclusion set does not make a
    # stale `layer:` an error — it makes it an ordinary spec key, carried
    # silently into the installer. Silent pass-through is the wrong answer: an
    # author still writing `layer:` is working from superseded documentation,
    # and an unexpected key in `item.spec` surfaces somewhere else as something
    # else. So it is rejected, and the message has to carry the author to the
    # replacement, not merely refuse.
    p = _write(
        tmp_path,
        """
        version: 1
        target: {kind: repo, name: x, path: /tmp/x}
        items:
          - installer: uv
            importance: required
            layer: galaxy
    """,
    )
    with pytest.raises(RecipeError) as exc:
        load_recipe(p)
    msg = str(exc.value)
    assert "items[0]" in msg
    assert "removed on 2026-09-04" in msg
    assert "§9.1" in msg


def test_the_layer_guard_rejects_every_value_including_the_ones_that_were_legal(tmp_path):
    # The guard keys on the *key*, not on the value, and this is the half that
    # would be easy to get wrong: a check written as "reject an unknown layer"
    # would let `layer: system` through, and `system` is the value every
    # shipped recipe used, so it is the one most likely to be left behind.
    for value in ("system", "workspace", "project", "repo", "worktree"):
        p = _write(
            tmp_path,
            f"""
            version: 1
            target: {{kind: repo, name: x, path: /tmp/x}}
            items:
              - installer: uv
                importance: required
                layer: {value}
        """,
        )
        with pytest.raises(RecipeError, match="removed on 2026-09-04"):
            load_recipe(p)


def test_oneline_run_multiline_raises(tmp_path):
    p = _write(
        tmp_path,
        """
        version: 1
        target: {kind: repo, name: x, path: /tmp/x}
        items:
          - installer: oneline
            importance: suggested
            run: |
              echo one
              echo two
    """,
    )
    with pytest.raises(RecipeError, match="one line"):
        load_recipe(p)
