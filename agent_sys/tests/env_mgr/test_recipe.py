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
# These four tests are what replaces `test_bad_layer_raises` and
# `tests/env_mgr/test_layer.py`, both deleted with it. The first two are the
# removal; the third is the shipped artefacts; the fourth is the one consequence
# of the removal that is not obvious from the diff.


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


def test_a_stale_layer_key_becomes_an_ordinary_spec_key(tmp_path):
    # Characterisation, and the reason it is written down: `layer` was listed in
    # `_CLI_KEYS`, which is the set of keys held *back* from `Item.spec`. With
    # the field gone the name is no longer special, so an author-written
    # `layer:` left behind in a recipe this repo does not own is neither
    # rejected nor dropped — it is carried into the installer's `item.spec` like
    # any other unrecognised key. That is the existing contract for unknown
    # keys, not a new one; it is asserted so the behaviour is a decision on
    # record rather than something the next reader discovers in an installer.
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
    _, items = load_recipe(p)
    assert items[0].spec["layer"] == "galaxy"


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
