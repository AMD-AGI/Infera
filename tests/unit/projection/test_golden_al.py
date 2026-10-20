###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Speculative acceptance read from measured acceptance-length files.

Every acceptance the projector uses comes from a file: a model's own golden
curve, its sibling method's, or the method's default curve. The golden curves
live outside the repo (InferenceX's golden_al_distribution, found through
INFERASIM_GOLDEN_AL_DIR); the tests that read them skip without it. The
defaults are checked in and must regenerate from the golden files unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infera.projection.core.projection.inference_projection import golden_al as ga


@pytest.fixture
def golden():
    env = os.environ.get("INFERASIM_GOLDEN_AL_DIR")
    if not env or not Path(env).is_dir():
        pytest.skip("INFERASIM_GOLDEN_AL_DIR is not set to InferenceX's golden_al_distribution")
    return Path(env)


@pytest.fixture
def curves(tmp_path, monkeypatch):
    """A golden directory of toy curves, standing in for InferenceX's."""

    def write(name, **modes):
        body = {f"thinking_{m}": sweep for m, sweep in modes.items()}
        (tmp_path / name).write_text(yaml.safe_dump({Path(name).stem: body}))

    write("toy_mtp.yaml", on={1: 1.8, 2: 2.5, 3: 3.0}, off={1: 1.9, 2: 2.6, 3: 3.1})
    write("other_mtp.yaml", on={1: 1.9, 2: 2.7, 3: 3.4})
    write("toy_dspark.yaml", on={1: 1.7, 2: 2.3, 3: 2.8, 7: 3.84})
    write("toy_dspark_atom.yaml", on={1: 1.7, 2: 2.3, 3: 2.8, 7: 3.78})
    monkeypatch.setenv("INFERASIM_GOLDEN_AL_DIR", str(tmp_path))
    return tmp_path


def test_every_golden_file_parses_with_a_known_method(golden):
    curves = ga.golden_curves(golden)
    assert len(curves) >= 11
    for c in curves:
        assert c.method in ("mtp", "eagle3", "dspark"), c.name
        pts, _ = c.points("on")
        assert all(1.0 < al <= k + 1 for k, al in pts.items()), c.name


def test_default_files_regenerate_from_the_golden_files(golden):
    derived = ga.derive_defaults(golden)
    assert set(derived) == {"mtp", "eagle3", "dspark", "draft_model", "ngram"}
    for method, entry in derived.items():
        on_disk = (ga.DEFAULTS_DIR / f"{method}.yaml").read_text()
        assert on_disk == ga.render_default(method, entry), (
            f"defaults/{method}.yaml is stale; regenerate with "
            f"python -m infera.projection.core.projection.inference_projection.golden_al "
            f"--write-defaults --golden-dir {golden}"
        )


def test_deriving_defaults_needs_the_golden_files(monkeypatch):
    monkeypatch.delenv("INFERASIM_GOLDEN_AL_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="INFERASIM_GOLDEN_AL_DIR"):
        ga.derive_defaults()


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


def test_default_mtp_curve_sits_inside_the_models_it_summarises(golden):
    on = ga.default_curve("mtp").points("on")[0]
    at3 = [
        ga.load_curve(golden / f"{n}.yaml").points("on")[0][3]
        for n in ("glm5.2_mtp", "dsv4_mtp", "qwen3.5_mtp")
    ]
    assert min(at3) <= on[3] <= max(at3)


def test_depths_interpolate_and_extrapolate_from_measurement():
    pts = {1: 1.8, 2: 2.4, 4: 3.2}
    assert ga.al_at(pts, 0) == 1.0
    assert ga.al_at(pts, 2) == 2.4
    assert ga.al_at(pts, 3) == pytest.approx(2.8)
    beyond = [ga.al_at(pts, k) for k in (4, 5, 6, 10)]
    assert beyond == sorted(beyond)
    assert beyond[0] == 3.2 and beyond[-1] <= 11.0
    # A short sweep (InferenceX's dsv41flash_dspark.yaml, thinking_on) is
    # carried past its last depth by the fitted chain, not held flat and not
    # run away.
    flash = {1: 1.81, 2: 2.43, 3: 2.90, 4: 3.26, 5: 3.51}
    assert flash[5] < ga.al_at(flash, 7) < flash[5] + 0.6


def _model(**spec):
    return SimpleNamespace(mtp_num_layers=1, speculative=spec or None)


def test_resolution_order(curves):
    toy = _model(mtp={"acceptance": "toy_mtp.yaml"})
    acc = ga.resolve("mtp", model_config=toy)
    assert acc.source == "model:toy_mtp.yaml[thinking_on]"
    assert acc.at(3) == 3.0
    # EAGLE3 stands in for MTP, and the other way round.
    e3 = ga.resolve("eagle3", model_config=toy)
    assert e3.source.startswith("model mtp as eagle3:toy_mtp.yaml")
    assert e3.at(3) == 3.0
    # No curve of its own or its sibling's: the method's default.
    ds = ga.resolve("dspark", model_config=toy)
    assert ds.source.startswith("default:dspark.yaml")
    # An explicit file wins over the model's, by name or by path.
    f = ga.resolve("mtp", model_config=toy, curve="other_mtp")
    assert f.source.startswith("file:other_mtp.yaml") and f.at(3) == 3.4
    p = ga.resolve("mtp", model_config=toy, curve=str(curves / "other_mtp.yaml"))
    assert p.at(3) == 3.4


def test_engine_variants_and_thinking_modes(curves):
    toy = SimpleNamespace(
        speculative=SimpleNamespace(
            default="dspark",
            dspark=SimpleNamespace(
                acceptance=SimpleNamespace(default="toy_dspark.yaml", atom="toy_dspark_atom.yaml")
            ),
        )
    )
    assert ga.default_method(toy) == "dspark"
    assert ga.resolve("dspark", model_config=toy, engine="vllm").at(7) == 3.84
    assert ga.resolve("dspark", model_config=toy, engine="atom").at(7) == 3.78
    mtp = _model(mtp={"acceptance": "toy_mtp.yaml"})
    assert ga.resolve("mtp", model_config=mtp, thinking="off").at(3) == 3.1
    assert ga.resolve("mtp", model_config=mtp, thinking=False).at(3) == 3.1
    # A curve measured in one mode only is read in that mode, and says so.
    one = ga.resolve("dspark", model_config=toy, thinking="off")
    assert one.source.endswith("[thinking_on]")


def test_a_model_file_not_found_reads_the_default(monkeypatch):
    monkeypatch.delenv("INFERASIM_GOLDEN_AL_DIR", raising=False)
    acc = ga.resolve("mtp", model_config=_model(mtp={"acceptance": "glm5.2_mtp.yaml"}))
    assert acc.source == "default:mtp.yaml[thinking_on] (glm5.2_mtp.yaml not found)"
    assert acc.points == ga.default_curve("mtp").points("on")[0]


def test_an_explicit_file_not_found_is_an_error(monkeypatch):
    monkeypatch.delenv("INFERASIM_GOLDEN_AL_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="INFERASIM_GOLDEN_AL_DIR is unset"):
        ga.resolve("mtp", curve="glm5.2_mtp.yaml")


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


@pytest.fixture
def github(tmp_path, monkeypatch):
    """A fake GitHub: ``files`` maps a file name to its text; ``None`` is offline."""
    import urllib.error

    state = {"files": {}, "calls": 0}

    def urlopen(url, timeout=None):
        state["calls"] += 1
        if state["files"] is None:
            raise urllib.error.URLError("offline")
        name = url.rsplit("/", 1)[-1]
        if url.startswith(ga.GOLDEN_URLS[0]) and name in state["files"]:
            return _Response(state["files"][name].encode())
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(ga.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("INFERASIM_GOLDEN_AL_FETCH", "1")
    monkeypatch.setenv("INFERASIM_GOLDEN_AL_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("INFERASIM_GOLDEN_AL_DIR", raising=False)
    ga.fetch_golden.cache_clear()
    yield state
    ga.fetch_golden.cache_clear()


_REMOTE = yaml.safe_dump({"remote": {"thinking_on": {1: 1.7, 2: 2.2, 3: 2.6}}})


def test_a_model_file_is_fetched_from_inferencex_and_cached(github, tmp_path):
    github["files"]["remote_mtp.yaml"] = _REMOTE
    model = _model(mtp={"acceptance": "remote_mtp.yaml"})
    acc = ga.resolve("mtp", model_config=model)
    assert acc.source == "model:remote_mtp.yaml[thinking_on]" and acc.at(3) == 2.6
    assert (tmp_path / "cache" / "remote_mtp.yaml").is_file()
    # A later process reads the cached copy without asking GitHub again.
    ga.fetch_golden.cache_clear()
    calls = github["calls"]
    assert ga.resolve("mtp", model_config=model).at(3) == 2.6
    assert github["calls"] == calls


def test_a_file_inferencex_does_not_have_reads_the_default(github, tmp_path):
    acc = ga.resolve("mtp", model_config=_model(mtp={"acceptance": "gone_mtp.yaml"}))
    assert acc.source == "default:mtp.yaml[thinking_on] (gone_mtp.yaml not found)"
    assert (tmp_path / "cache" / "gone_mtp.yaml.missing").is_file()
    ga.fetch_golden.cache_clear()
    calls = github["calls"]
    ga.resolve("mtp", model_config=_model(mtp={"acceptance": "gone_mtp.yaml"}))
    assert github["calls"] == calls


def test_offline_keeps_a_stale_copy_and_otherwise_reads_the_default(github, tmp_path):
    github["files"]["remote_mtp.yaml"] = _REMOTE
    assert ga.fetch_golden("remote_mtp.yaml") is not None
    cached = tmp_path / "cache" / "remote_mtp.yaml"
    old = cached.stat().st_mtime - 2 * 24 * 3600
    os.utime(cached, (old, old))
    github["files"] = None
    ga.fetch_golden.cache_clear()
    assert ga.fetch_golden("remote_mtp.yaml") == cached
    acc = ga.resolve("mtp", model_config=_model(mtp={"acceptance": "other_mtp.yaml"}))
    assert acc.source.startswith("default:mtp.yaml")
    assert not (tmp_path / "cache" / "other_mtp.yaml.missing").exists()


def test_a_malformed_download_is_not_cached(github, tmp_path):
    github["files"]["bad_mtp.yaml"] = "<html>rate limited</html>"
    assert ga.fetch_golden("bad_mtp.yaml") is None
    assert not (tmp_path / "cache" / "bad_mtp.yaml").exists()


def test_a_local_directory_wins_over_inferencex(github, curves):
    github["files"]["toy_mtp.yaml"] = _REMOTE
    acc = ga.resolve("mtp", model_config=_model(mtp={"acceptance": "toy_mtp.yaml"}))
    assert acc.at(3) == 3.0 and github["calls"] == 0


def test_shipped_presets_name_files_that_exist(golden, monkeypatch):
    from infera.projection.core.config.yaml_loader import parse_yaml

    monkeypatch.setenv("INFERASIM_GOLDEN_AL_DIR", str(golden))
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
