"""sources.campaign_roots (config.analysis.v1): a campaign root recorded as seen inside a container is read on the
host through the path map (longest `from`, whole path components), and a model-extracted root is accepted under
a `to` root; every ownership check still applies."""

import os
import shutil
from pathlib import Path

import analysis_paths  # noqa: F401  (sys.path as o11y-run arranges)
from analysis_paths import FIX
from o11y_protocol import schemas
from o11y_sources import forge

MAP = [
    {"from": "/shared_nfs/yihou", "to": "/perf_apps/yihou"},
    {"from": "/shared_nfs", "to": "/srv/nfs"},
]


def _camp(tmp: Path, *parts: str) -> Path:
    """A KernelForge campaign copied under tmp, owner-only (copies get the umask's group bits)."""
    dst = tmp.joinpath(*parts)
    shutil.copytree(FIX / "forge" / "decode", dst)
    for d, _, files in os.walk(tmp):
        for p in [d, *(os.path.join(d, f) for f in files)]:
            os.chmod(p, os.stat(p).st_mode & ~0o022)
    return dst


def test_map_root_whole_components_longest_prefix_and_normalised():
    assert forge.map_root("/shared_nfs/yihou/kf/runs", MAP) == "/perf_apps/yihou/kf/runs"
    assert forge.map_root("/shared_nfs/yihou", MAP) == "/perf_apps/yihou"
    assert (
        forge.map_root("/shared_nfs/yihouX/kf", MAP) == "/srv/nfs/yihouX/kf"
    )  # not under /shared_nfs/yihou
    assert forge.map_root("/shared_nfsX/kf", MAP) == "/shared_nfsX/kf"  # no `from` matches
    assert (
        forge.map_root("/shared_nfs/yihou/../other/kf", MAP) == "/srv/nfs/other/kf"
    )  # `..` resolved first
    assert forge.map_root("/shared_nfs/yihou/kf/../../../etc", MAP) == "/etc"
    assert forge.map_root("rel/kf", MAP) == "rel/kf"  # relative: unchanged, refused later
    assert forge.map_root("/a/b", [{"to": "/x"}]) == "/a/b"  # an entry without `from` maps nothing


def test_extracted_root_accepted_under_a_to_root_after_mapping(tmp_path):
    host = tmp_path / "host"
    camp = _camp(tmp_path, "host", "kf", "runs", "decode")
    cmap = [{"from": "/container/yihou", "to": str(host)}]
    ext = {"campaign_root": "/container/yihou/kf/runs/decode", "writer": {"source": "extracted"}}
    dirs, bad, _ = forge.discover([ext], [], cmap)
    assert bad == [] and dirs == [str(camp / "forge_experiments")]
    # without the map the same record is refused (outside every allowed base), never guessed
    dirs, bad, _ = forge.discover([ext], [], [])
    assert dirs == [] and "outside" in bad[0][1]
    # an extracted root that maps nowhere and lies outside every `to` root is refused
    other = _camp(tmp_path, "elsewhere", "decode")
    dirs, bad, _ = forge.discover(
        [{"campaign_root": str(other), "writer": {"source": "extracted"}}], [], cmap
    )
    assert dirs == [] and "sources.campaign_roots" in bad[0][1]


def test_escapes_refused(tmp_path):
    host = tmp_path / "host"
    _camp(tmp_path, "host", "kf", "decode")
    outside = _camp(tmp_path, "outside", "decode")
    cmap = [{"from": "/container/yihou", "to": str(host)}]
    allowed = forge.allowed_bases([], [], cmap)
    dotdot = forge.map_root("/container/yihou/../../" + str(outside.relative_to("/")), cmap)
    assert forge.campaign_root_problem(dotdot, allowed)  # `..` cannot climb out of the `to` root
    (host / "link").symlink_to(outside)  # a symlink inside the `to` root pointing elsewhere
    why = forge.campaign_root_problem(forge.map_root("/container/yihou/link", cmap), allowed)
    assert why and "outside" in why


def test_allowlisted_root_below_a_world_writable_parent_is_read_with_one_note(tmp_path):
    shared = tmp_path / "perf_apps"
    camp = _camp(tmp_path, "perf_apps", "yihou", "kf", "decode")
    os.chmod(shared, 0o777)  # like /perf_apps: world-writable, not sticky
    cmap = [{"from": "/shared_nfs/yihou", "to": str(shared / "yihou")}]
    ext = {"campaign_root": "/shared_nfs/yihou/kf/decode", "writer": {"source": "extracted"}}
    dirs, bad, notes = forge.discover([ext], [], cmap)
    assert bad == [] and dirs == [str(camp / "forge_experiments")]
    assert [n for _, n in notes if n.startswith("untrusted_path: ")] and "not sticky" in notes[0][1]
    os.chmod(camp / "forge_experiments" / "events.jsonl", 0o666)  # ownership checks still apply
    assert forge.discover([ext], [], cmap)[0] == []


def test_config_schema_and_did_you_mean():
    ok = {
        "v": 1,
        "sources": {
            "campaign_roots": [
                {"from": "/shared_nfs/yihou", "to": "/perf_apps/yihou"},
                {"to": "/x"},
            ]
        },
    }
    assert schemas.problems("config.analysis.v1", ok) == []
    assert schemas.defaults("config.analysis.v1")["sources"]["campaign_roots"] == []
    typo = schemas.problems(
        "config.analysis.v1", {"v": 1, "sources": {"campaign_roots": [{"form": "/a", "to": "/b"}]}}
    )
    assert any("did you mean 'from'" in p for p in typo), typo
    relative = schemas.problems(
        "config.analysis.v1", {"v": 1, "sources": {"campaign_roots": [{"to": "rel"}]}}
    )
    assert relative
