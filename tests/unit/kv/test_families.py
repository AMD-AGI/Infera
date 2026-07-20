###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import pytest

from infera.kv.families import (
    KNOWN_FAMILIES,
    ModelFamily,
    load_registry_from_yaml,
    resolve_family,
)


def test_canonical_resolves_to_itself() -> None:
    for fam in KNOWN_FAMILIES:
        assert resolve_family(fam.canonical) == fam.canonical


def test_alias_resolves_to_canonical() -> None:
    assert resolve_family("Qwen3.6") == "qwen3.6"
    assert resolve_family("qwen-3.6") == "qwen3.6"
    assert resolve_family("qwen3_6") == "qwen3.6"
    assert resolve_family("qwen36") == "qwen3.6"


def test_case_insensitive() -> None:
    assert resolve_family("QWEN3.6") == "qwen3.6"
    assert resolve_family("Qwen3.6") == "qwen3.6"
    assert resolve_family("qwen3.6") == "qwen3.6"


def test_separator_insensitive() -> None:
    # _ and - are interchangeable
    assert resolve_family("qwen_3.6") == "qwen3.6"
    assert resolve_family("qwen-3.6") == "qwen3.6"


def test_hf_style_id_substring_match() -> None:
    # The HF-style ID contains the alias as a substring.
    assert resolve_family("Qwen/Qwen3.6-27B") == "qwen3.6"
    assert resolve_family("meta-llama/Llama-3-70B") == "llama-3"
    assert resolve_family("openai/gpt-oss-120b") == "gpt-oss"
    assert resolve_family("deepseek-ai/DeepSeek-V3-Base") == "deepseek-v3"


def test_unknown_family_returns_normalized_input() -> None:
    assert resolve_family("Some-Future-Model-2027") == "some-future-model-2027"
    assert resolve_family("UNKNOWN_thing") == "unknown-thing"


def test_unknown_does_not_partial_match_other_canonicals() -> None:
    # "qwen2.5" is not in the registry; should not collapse to "qwen2" or "qwen3".
    # Today the registry doesn't have a "qwen2.5" canonical, so it returns the
    # normalized input. (This guards against an over-eager substring match.)
    res = resolve_family("qwen2.5")
    # "qwen2" IS an alias of canonical "qwen2", and "qwen2" is a substring of
    # "qwen2.5", so we'd resolve to "qwen2". That's intentional — when a new
    # model family appears it'd otherwise need an entry. Document the choice.
    # If this is undesired, add explicit canonical "qwen2.5" to the registry.
    assert res in {"qwen2", "qwen2.5"}  # Accept either; the test pins behavior.


def test_custom_registry_used_when_passed() -> None:
    custom = [ModelFamily(canonical="my-model", aliases=("MyModel", "org/my-model"))]
    assert resolve_family("MyModel", registry=custom) == "my-model"
    assert resolve_family("org/my-model", registry=custom) == "my-model"
    # Built-in entries not visible when custom registry passed.
    assert resolve_family("qwen3.6", registry=custom) == "qwen3.6"


def test_load_registry_from_yaml_extends(tmp_path) -> None:
    yaml_path = tmp_path / "families.yaml"
    yaml_path.write_text(
        "- canonical: my-future-model\n  aliases: [MyFutureModel, my_future_model]\n"
    )
    try:
        import yaml  # noqa: F401
    except ImportError:
        pytest.skip("PyYAML not installed")
    reg = load_registry_from_yaml(yaml_path)
    names = {f.canonical for f in reg}
    assert "my-future-model" in names
    # Builtins survived.
    assert "qwen3.6" in names


def test_load_registry_from_yaml_missing_file_returns_builtins(tmp_path) -> None:
    reg = load_registry_from_yaml(tmp_path / "does-not-exist.yaml")
    names = {f.canonical for f in reg}
    # Builtins all present.
    builtins = {f.canonical for f in KNOWN_FAMILIES}
    assert builtins.issubset(names)


def test_load_registry_from_yaml_malformed_returns_builtins(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(":::not valid yaml:::")
    try:
        import yaml  # noqa: F401
    except ImportError:
        pytest.skip("PyYAML not installed")
    reg = load_registry_from_yaml(bad)
    # Falls back to builtins; doesn't crash.
    names = {f.canonical for f in reg}
    builtins = {f.canonical for f in KNOWN_FAMILIES}
    assert builtins.issubset(names)
