###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Speculative acceptance read from measured acceptance-length files.

Every acceptance the projector uses comes from a file: a model's own golden
curve, its sibling method's, or the method's default curve, which is itself
derived from the golden files and must regenerate from them unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from infera.projection.core.projection.inference_projection import golden_al as ga


def test_every_golden_file_parses_with_a_known_method():
    curves = ga.golden_curves(ga.GOLDEN_DIR)
    assert len(curves) >= 11
    for c in curves:
        assert c.method in ("mtp", "eagle3", "dspark"), c.name
        pts, _ = c.points("on")
        assert all(1.0 < al <= k + 1 for k, al in pts.items()), c.name


def test_default_files_regenerate_from_the_golden_files():
    derived = ga.derive_defaults(ga.GOLDEN_DIR)
    assert set(derived) == {"mtp", "eagle3", "dspark", "draft_model", "ngram"}
    for method, entry in derived.items():
        on_disk = (ga.DEFAULTS_DIR / f"{method}.yaml").read_text()
        assert on_disk == ga.render_default(method, entry), (
            f"defaults/{method}.yaml is stale; regenerate with "
            f"python -m infera.projection.core.projection.inference_projection.golden_al "
            f"--write-defaults"
        )


@pytest.mark.parametrize("method", ["mtp", "eagle3", "dspark", "draft_model", "ngram"])
def test_default_curves_are_non_decreasing_over_the_sweep(method):
    c = ga.default_curve(method)
    for mode in ("on", "off"):
        pts, got = c.points(mode)
        assert got == mode
        assert sorted(pts) == list(ga.DEFAULT_DEPTHS)
        vals = [pts[k] for k in ga.DEFAULT_DEPTHS]
        assert vals == sorted(vals)
        assert all(1.0 < v <= k + 1 for k, v in zip(ga.DEFAULT_DEPTHS, vals))


def test_default_mtp_curve_sits_inside_the_models_it_summarises():
    on = ga.default_curve("mtp").points("on")[0]
    at3 = [ga.load_curve(n).points("on")[0][3] for n in ("glm5.2_mtp", "dsv4_mtp", "qwen3.5_mtp")]
    assert min(at3) <= on[3] <= max(at3)


def test_depths_interpolate_and_extrapolate_from_measurement():
    pts = {1: 1.8, 2: 2.4, 4: 3.2}
    assert ga.al_at(pts, 0) == 1.0
    assert ga.al_at(pts, 2) == 2.4
    assert ga.al_at(pts, 3) == pytest.approx(2.8)
    beyond = [ga.al_at(pts, k) for k in (4, 5, 6, 10)]
    assert beyond == sorted(beyond)
    assert beyond[0] == 3.2 and beyond[-1] <= 11.0
    # A short sweep is carried past its last depth by the fitted chain, not
    # held flat and not run away.
    flash = ga.load_curve("dsv41flash_dspark.yaml").points("on")[0]
    assert max(flash) == 5
    assert flash[5] < ga.al_at(flash, 7) < flash[5] + 0.6


def _model(**spec):
    return SimpleNamespace(mtp_num_layers=1, speculative=spec or None)


def test_resolution_order():
    glm = _model(mtp={"acceptance": "glm5.2_mtp.yaml"})
    acc = ga.resolve("mtp", model_config=glm)
    assert acc.source == "model:glm5.2_mtp.yaml[thinking_on]"
    assert acc.at(3) == 2.99
    # EAGLE3 stands in for MTP, and the other way round.
    e3 = ga.resolve("eagle3", model_config=glm)
    assert e3.source.startswith("model mtp as eagle3:glm5.2_mtp.yaml")
    assert e3.at(3) == 2.99
    # No curve of its own or its sibling's: the method's default.
    ds = ga.resolve("dspark", model_config=glm)
    assert ds.source.startswith("default:dspark.yaml")
    # An explicit file wins over the model's.
    f = ga.resolve("mtp", model_config=glm, curve="qwen3.5_mtp")
    assert f.source.startswith("file:qwen3.5_mtp.yaml") and f.at(3) == 3.39


def test_engine_variants_and_thinking_modes():
    kimi = SimpleNamespace(
        speculative=SimpleNamespace(
            default="dspark",
            dspark=SimpleNamespace(
                acceptance=SimpleNamespace(
                    default="kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml",
                    atom="kimik3_dspark.yaml",
                )
            ),
        )
    )
    assert ga.default_method(kimi) == "dspark"
    assert ga.resolve("dspark", model_config=kimi, engine="vllm").at(7) == 3.84
    assert ga.resolve("dspark", model_config=kimi, engine="atom").at(7) == 3.78
    dsv4 = _model(mtp={"acceptance": "dsv4_mtp.yaml"})
    assert ga.resolve("mtp", model_config=dsv4, thinking="off").at(3) == 2.97
    assert ga.resolve("mtp", model_config=dsv4, thinking=False).at(3) == 2.97
    # A curve measured in one mode only is read in that mode, and says so.
    k3 = ga.resolve("dspark", model_config=kimi, thinking="off")
    assert k3.source.endswith("[thinking_on]")


def test_override_directory(tmp_path, monkeypatch):
    (tmp_path / "toy_mtp.yaml").write_text(
        yaml.safe_dump({"toy": {"thinking_on": {1: 1.5, 2: 1.9, 3: 2.1}}})
    )
    monkeypatch.setenv("INFERASIM_GOLDEN_AL_DIR", str(tmp_path))
    acc = ga.resolve("mtp", model_config=_model(mtp={"acceptance": "toy_mtp.yaml"}))
    assert acc.at(3) == 2.1


def test_shipped_presets_name_files_that_exist():
    from pathlib import Path

    from infera.projection.core.config.yaml_loader import parse_yaml

    models = Path(ga.__file__).resolve().parents[3] / "configs" / "models" / "megatron"
    seen = 0
    for path in sorted(models.glob("*.yaml")):
        cfg = parse_yaml(str(path))
        spec = ga.model_speculative(SimpleNamespace(speculative=cfg.get("speculative")))
        for method, entry in spec.items():
            if method == "default":
                continue
            refs = entry.get("acceptance")
            for ref in refs.values() if isinstance(refs, dict) else [refs]:
                assert ga.find_file(ref).is_file(), (path.name, ref)
                seen += 1
    assert seen >= 9
