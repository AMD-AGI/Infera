# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
import textwrap

import pytest

from env_mgr.recipe import Item, RecipeError, load_recipe
from env_mgr.runner import detect_conflicts


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
            layer: repo
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
    assert it.layer == "repo"
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
            layer: system
            name: pyright-langserver
          - installer: uv
            importance: required
            layer: system
            provides: serena
          - installer: apt
            importance: suggested
            layer: system
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
        layer="system",
        spec={"packages": ["jq"], "provides": {"jq": "jq"}},
    )
    assert it.name == "apt"
    assert isinstance(it.name, str)
    assert {it.name: "usable as dict key"}  # must be hashable


def test_item_name_uses_provides_when_it_is_a_string():
    it = Item(
        installer="uv",
        importance="required",
        layer="system",
        spec={"provides": "serena"},
    )
    assert it.name == "serena"


def test_detect_conflicts_does_not_crash_on_dict_provides():
    items = [
        Item(
            installer="apt",
            importance="suggested",
            layer="system",
            spec={"packages": ["ripgrep"], "provides": {"ripgrep": "rg"}},
        ),
        Item(
            installer="apt",
            importance="suggested",
            layer="system",
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
            layer: repo
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
            layer: repo
    """,
    )
    with pytest.raises(RecipeError, match="importance"):
        load_recipe(p)


def test_bad_layer_raises(tmp_path):
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
    with pytest.raises(RecipeError, match="layer"):
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
            layer: system
            run: |
              echo one
              echo two
    """,
    )
    with pytest.raises(RecipeError, match="one line"):
        load_recipe(p)
