"""User round 2026-11-25: machines in use and what each runs; campaigns on another cluster from their
reported progress; what each session is doing now. dsv41 shape: the main serves vLLM PD on openai n03-33
with no open run, the KF child runs a forge round whose root lives only on crusoe."""

import analysis_paths  # noqa: F401  (sys.path as o11y-run arranges)
import pytest
from o11y_facts import handle as facts_handle
from o11y_job import run_consumer
from o11y_protocol import store

MAIN, KF = "s-main", "s-kf"
REMOTE_ROOT = "/shared_nfs/yihou/dsv41_kf_20261120/forge/runs"  # exists on crusoe only


def _rec(kind, ts, sid, role, **fields):
    w = {"session_id": sid, "role": role, "source": "written", "session_name": sid}
    return {"v": 1, "kind": kind, "id": f"rec-{kind}-{ts}", "ts": ts, "writer": w, **fields}


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    for k, v in {
        "AGENT_SYS_O11Y_FLEET_DIR": root,
        "CLAUDE_CONFIG_DIR": tmp_path / "claude",
        "AGENT_SYS_O11Y_OTEL_DIR": tmp_path / "otel",
        "AGENT_SYS_HOME": tmp_path / "prefix",
    }.items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.delenv("AGENT_SYS_O11Y", raising=False)
    rows = [
        {"session_id": MAIN, "pid": 1, "alive": True, "role": "main", "status": "idle", "active": True,
         "name": "infera-dsv41", "status_since": "2026-11-25T10:20:00Z"},
        {"session_id": KF, "pid": 2, "alive": True, "role": "child", "status": "busy", "name": "infera-dsv41-kf",
         "status_since": "2026-11-25T10:21:00Z"},
    ]  # fmt: skip
    store.write_json(
        root / store.SESSIONS,
        {"v": 1, "generated_at": store.now(), "main_session_id": MAIN, "sessions": rows},
    )
    ledger = [
        _rec("machine", "2026-11-25T09:00:00Z", MAIN, "main", node="n03-31", cluster="openai", state="held",
             task="old bench"),
        _rec("machine", "2026-11-25T10:00:00Z", MAIN, "main", node="n03-33", cluster="openai", state="held",
             task="vLLM PD baseline"),
        _rec("machine", "2026-11-25T10:05:00Z", MAIN, "main", node="n03-31", cluster="openai", state="released"),
        _rec("run_start", "2026-11-25T10:10:00Z", KF, "child", run_id="kf-r3", theme="kf: Forge round 3",
             point="explore", machines=[{"node": "v3-140", "cluster": "crusoe"}]),
        _rec("campaign", "2026-11-25T10:11:00Z", KF, "child", campaign_root=REMOTE_ROOT,
             operator_ids=["mhc_seam_bf16_decode"], started_at="2026-11-25T10:11:00Z",
             machine={"node": "v3-141", "cluster": "crusoe"},
             progress={"as_of": "2026-11-25T10:30:00Z", "status": "running", "iterations": 7, "best_speedup": 1.31}),
    ]  # fmt: skip
    for r in ledger:
        store.append(root / store.LEDGER, r)
    return root


def _facts(root):
    assert run_consumer("facts", facts_handle, root=root, force=True)["passes"] == 1
    return {k: store.read_json(root / store.FACTS[k]) for k in ("general", "detail", "kernelforge")}


def test_machines_in_use_with_what_each_runs(fleet):
    f = _facts(fleet)
    got = {m["node"]: (m.get("task"), m["source"]) for m in f["general"]["machines"]}
    assert got == {
        "n03-33": ("vLLM PD baseline", "machine"),  # held; n03-31 was released
        "v3-140": ("kf: Forge round 3", "ledger"),  # an open run
        "v3-141": ("kf: mhc_seam_bf16_decode", "campaign"),  # a running campaign
    }
    detail = {m["node"]: m for m in f["detail"]["machines"]}
    assert (
        detail["n03-33"]["task"] == "vLLM PD baseline"
        and detail["n03-33"]["session"] == "infera-dsv41"
    )  # the registry name


def test_a_campaign_on_another_cluster_is_listed_from_its_reported_progress(fleet):
    (c,) = _facts(fleet)["kernelforge"]["campaigns"]
    assert c["source"] == "reported" and c["campaign_root"] == REMOTE_ROOT
    assert (c["status"], c["iterations"], c["best_speedup"], c["as_of"]) == (
        "running",
        7,
        1.31,
        "2026-11-25T10:30:00Z",
    )
    assert c["problem"].startswith("reported, not readable here: ")


def test_what_each_session_is_doing_now(fleet):
    now = _facts(fleet)["detail"]["now"]
    assert [(n["session_id"], n["activity"]) for n in now] == [
        (MAIN, "idle"),
        (KF, "run kf: Forge round 3 (explore)"),
    ]
    rows = store.read_json(fleet / store.SESSIONS)
    rows["sessions"][0].update(status="waiting", waiting_for="permission prompt")
    rows["sessions"][1]["status"] = "busy"
    store.write_json(fleet / store.SESSIONS, rows)
    store.append(fleet / store.LEDGER, _rec("run_end", "2026-11-25T10:40:00Z", KF, "child", run_id="kf-r3",
                                           status="failed", conclusion="node lost"))  # fmt: skip
    now = _facts(fleet)["detail"]["now"]
    assert [n["activity"] for n in now] == ["waiting (permission prompt)", "busy (no run recorded)"]


def test_every_open_run_names_its_session(fleet):
    """dash, live dsv41: open runs showed session null; a run's session is its writer's registry name."""
    store.append(
        fleet / store.LEDGER,
        {**_rec("run_start", "2026-11-25T10:12:00Z", MAIN, "main", run_id="umbrella", theme="tools parity",
                point="validate"), "writer": {"session_id": MAIN, "role": "main", "source": "written"}},
    )  # fmt: skip
    runs = {r["run_id"]: r["session"] for r in _facts(fleet)["detail"]["current_runs"]}
    assert runs == {"kf-r3": "infera-dsv41-kf", "umbrella": "infera-dsv41"}


def test_runs_name_their_session_across_clear_and_in_recent_runs(fleet):
    """lead, live dsv41: session-written runs had session null. A run opened by the main before a /clear (its
    old id is in the registry lineage) names the main; recent runs name their session too."""
    rows = store.read_json(fleet / store.SESSIONS)
    rows["sessions"][0]["lineage"] = ["s-main-0"]
    store.write_json(fleet / store.SESSIONS, rows)
    w_old = {"session_id": "s-main-0", "role": "main", "source": "written"}
    store.append(fleet / store.LEDGER, {**_rec("run_start", "2026-11-25T09:30:00Z", MAIN, "main", run_id="pre-clear",
                                              theme="umbrella", point="validate"), "writer": w_old})  # fmt: skip
    store.append(fleet / store.LEDGER, {**_rec("run_start", "2026-11-25T09:31:00Z", MAIN, "main", run_id="done1",
                                              theme="t", point="explore"), "writer": w_old})  # fmt: skip
    store.append(fleet / store.LEDGER, {**_rec("run_end", "2026-11-25T09:40:00Z", MAIN, "main", run_id="done1",
                                              status="failed", conclusion="x"), "writer": w_old})  # fmt: skip
    d = _facts(fleet)["detail"]
    assert {r["run_id"]: r["session"] for r in d["current_runs"]}["pre-clear"] == "infera-dsv41"
    assert {r["run_id"]: r.get("session") for r in d["recent_runs"]}["done1"] == "infera-dsv41"
