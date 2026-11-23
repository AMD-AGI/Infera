"""Deterministic detectors of the log job (design §8.3, §8.4). No model here.

Sessions are judged only by the trigger's registry (review A4); new tokens come from o11y_facts' `spend`
(review A7); collector outages from `otel/health.json` (A6). What was reported is kept in
`state/log/detector.json`; every event carries `o11y.dedup_key`. Events are appended before cursors and state
are committed: a crash in between can repeat an event (same dedup key), never lose one.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from o11y_job import CursorStore, door, read_private, write_private
from o11y_sources import Problems, owned_open
from o11y_sources import e2eflow as e2e_src
from o11y_sources import forge as forge_src
from o11y_sources import otel as otel_src
from o11y_sources import pool as pool_src
from o11y_sources import transcripts as tr_src
from o11y_sources.cursor import read_new

WAIT_REASONS = {"permission prompt", "sandbox request", "worker request"}  # spec §6.1
ENDED_GRACE_S = 600  # an outbox `ended` waits this long for a `released` of the same request
SEVERITY = {"info": "INFO", "warning": "WARN", "error": "ERROR"}


def _sig(path: str) -> str:
    """File signature for "repeat only after the file changes" (spec §6.6)."""
    try:
        st = os.stat(path)
        return f"{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return "missing"


def _ev(path: str, off: int, uuid: str | None = None) -> dict:
    return {"path": path, "offset": off, **({"uuid": uuid} if uuid else {})}


class Detectors:
    def __init__(
        self, root: Path, cfg: dict, state_dir: Path, sessions: list, now: float | None = None
    ):
        self.root, self.cfg, self.fleet = Path(root), cfg, sessions
        self.now = now or time.time()
        self.cursors = CursorStore(state_dir)
        self.state_path = Path(state_dir) / "detector.json"
        self.state = read_private(self.state_path, {})
        self.problems = Problems()
        self.pending: list = []  # cursors committed after the events are written
        self.emitted = 0
        # spec principle 7: history that existed before the first pass is the baseline for every detector;
        # the first pass only establishes state (current-state detectors still report)
        self.baseline = "baseline_at" not in self.state

    def emit(
        self,
        sev: str,
        name: str,
        body: str,
        attrs: dict,
        sess=None,
        dedup: str = "",
        source="detector",
        once: tuple | None = None,
        history: bool = False,
    ):
        """Append one event; with `once=(bucket, key)` only the first time that key is seen; a `history` event
        is only remembered, not reported, on the baseline pass."""
        if once and not self.once(*once):
            return
        if history and self.baseline:
            return
        dedup = dedup or (f"{once[0]}:{once[1]}" if once else "")
        if door.event(
            self.root,
            sev,
            name,
            body,
            {**attrs, "o11y.dedup_key": dedup or f"{name}:{body[:80]}"},
            sess,
            source,
        ):
            self.emitted += 1

    def once(self, bucket: str, key: str, keep: int = 500) -> bool:
        """True the first time `key` is seen in `bucket` (bounded memory)."""
        seen = self.state.setdefault(bucket, [])
        if key in seen:
            return False
        seen.append(key)
        del seen[:-keep]
        return True

    # ------------------------------------------------------------------ registry, config
    def registry(self, problem: str | None, config_problems: list[str], config_new: bool) -> None:
        if problem:
            self.emit(
                "WARN",
                "o11y.registry.unavailable",
                problem[:500],
                {},
                once=("registry", problem[:200], 20),
            )
        if config_problems and config_new:
            self.emit(
                "WARN",
                "o11y.config.invalid",
                "; ".join(config_problems)[:500] + " (last valid config kept)",
                {},
                dedup=f"config:{'|'.join(config_problems)[:200]}",
            )
        for s in self.fleet:
            if s.crashed:
                self.emit(
                    "ERROR",
                    "o11y.session.crashed",
                    f"{s.label} ({s.role}) is gone without a clean exit (pid {s.pid})",
                    {"o11y.pid": s.pid, "o11y.tmux": s.tmux},
                    s,
                    once=("crashed", s.session_id),
                )
            since = s.status_since_epoch
            if not s.alive or since is None:
                continue
            age, key = self.now - since, f"{s.session_id}|{s.status}|{s.status_since}"
            if (
                s.status == "waiting"
                and (s.waiting_for or "") in WAIT_REASONS
                and age > self.cfg["perm_wait_s"]
            ):
                self.emit(
                    "WARN",
                    "o11y.session.permission_wait",
                    f"{s.label} has waited {int(age // 60)} min on a {s.waiting_for} (tmux {s.tmux or '?'})",
                    {
                        "o11y.waiting_for": s.waiting_for,
                        "o11y.waiting_since": s.status_since,
                        "o11y.tmux": s.tmux,
                    },
                    s,
                    dedup=f"wait:{key}",
                    once=("status_alerts", key),
                )
            elif s.status == "idle" and age > self.cfg["idle_warn_s"]:
                self.emit(
                    "WARN",
                    "o11y.session.idle",
                    f"{s.label} idle for {int(age // 60)} min",
                    {"o11y.idle_since": s.status_since},
                    s,
                    dedup=f"idle:{key}",
                    once=("status_alerts", key),
                )

    # ------------------------------------------------------------------ histories
    def histories(self) -> None:
        _, _, scans, commit = tr_src.scan_sessions(
            self.cursors, self.fleet, self.problems, "detect"
        )
        self.pending += commit
        owner = {
            f: s
            for s in self.fleet
            if s.transcript_path
            for f in tr_src.session_files(s.transcript_path)
        }
        k = int(self.cfg["api_error_k"])
        for sc in scans:
            s, path, ing = owner.get(sc.path), sc.path, sc.ingest
            if sc.chunk.reset:
                print(
                    f"o11y log: {path} was replaced or rewritten; re-scanned from the start",
                    file=sys.stderr,
                )
            for c in ing.compactions:
                self.emit(
                    "INFO",
                    "o11y.compaction.done",
                    f"{s.label if s else '?'} compacted ({c['trigger']}): {c['pre_tokens']} -> {c['post_tokens']} tokens",
                    {
                        "o11y.trigger": c["trigger"],
                        "o11y.pre_tokens": c["pre_tokens"],
                        "o11y.post_tokens": c["post_tokens"],
                        "evidence": [_ev(path, c["offset"], c["uuid"])],
                    },
                    s,
                    once=("compaction", c["uuid"] or f"{path}:{c['offset']}"),
                    history=True,
                )
            for e in ing.api_errors:
                if e["fatal"]:
                    self.emit(
                        "ERROR",
                        "o11y.api.error",
                        f"{s.label if s else '?'}: fatal API error ({e['error']}, {e['status']}): {e['text'][:200]}",
                        {
                            "o11y.error_kind": e["error"],
                            "o11y.status": e["status"],
                            "evidence": [_ev(path, e["offset"], e["uuid"])],
                        },
                        s,
                        once=("api_fatal", f"{path}:{e['offset']}"),
                        history=True,
                    )
            streaks = self.state.setdefault("api_streak_alerted", {})
            if ing.api_error_streak_max >= k and not streaks.get(path):
                streaks[path] = True
                last = ing.api_errors[-1]
                self.emit(
                    "ERROR",
                    "o11y.api.error_repeated",
                    f"{s.label if s else '?'}: {ing.api_error_streak_max} consecutive API errors, last {last['status']}: {last['text'][:200]}",
                    {
                        "o11y.count": ing.api_error_streak_max,
                        "o11y.status": last["status"],
                        "evidence": [
                            _ev(path, x["offset"], x["uuid"]) for x in ing.api_errors[-3:]
                        ],
                    },
                    s,
                    dedup=f"api:{path}:{last['offset']}",
                    history=True,
                )
            if sc.chunk.cursor.extra.get("api_err_streak", 0) == 0:
                streaks.pop(path, None)

    # ------------------------------------------------------------------ tokens (from o11y_facts' spend)
    def tokens(self) -> None:
        spend = (read_private(self.root / "facts" / "general.json", {}) or {}).get("spend") or {}
        step = int(self.cfg["token_step"])
        steps = self.state.setdefault(
            "token_steps", {}
        )  # last reported step per scope; first sight = baseline
        by_id = {s.session_id: s for s in self.fleet}
        scopes = [
            (x["session_id"], x["tokens"]["new"], x["tokens"]["cache_read"])
            for x in spend.get("sessions") or []
            if isinstance(x, dict) and isinstance(x.get("tokens"), dict)
        ]
        if "tokens_new" in spend:
            scopes.append(("_fleet", spend["tokens_new"], spend.get("tokens_cache_read", 0)))
        for sid, new, cache_read in scopes:
            n = int(new) // step
            if sid in steps and n > steps[sid]:
                s = by_id.get(sid)
                who = "fleet" if sid == "_fleet" else (s.label if s else sid[:8])
                self.emit(
                    "INFO",
                    "o11y.tokens.crossed",
                    f"{who} passed {n * step / 1e6:g}M new tokens (input + cache creation + output)",
                    {
                        "o11y.scope": "fleet" if sid == "_fleet" else "session",
                        "o11y.tokens_new": new,
                        "o11y.tokens_cache_read": cache_read,
                        "o11y.step": step,
                        **({} if s or sid == "_fleet" else {"session.id": sid}),
                    },
                    s,
                    dedup=f"tokens:{sid}:{n}",
                )
            steps[sid] = max(n, steps.get(sid, n))

    # ------------------------------------------------------------------ machine pool
    def pool(self) -> None:
        pdir = self.cfg["sources"]["pool_dir"]
        if not pool_src.trusted_dir(pdir, self.problems):
            return
        path = str(Path(pdir) / "outbox.jsonl")
        cur = self.cursors.get("pool", path)
        first = not cur.inode
        chunk = read_new(cur, opener=lambda q: owned_open(q, external=True))
        if chunk.problem:
            self.problems.add("pool", path, "(file)", f"not trusted: {chunk.problem}")
        self.pending.append(("pool", chunk.cursor))
        held = self.state.setdefault("pool_ended", {})  # req -> outbox `ended` held back
        if first:
            return  # history before the first pass is the baseline
        for e in pool_src.parse_outbox(path, chunk.lines, self.problems):
            sess = pool_src.attribute(
                e.get("notify") or "", pool_src.request_is_ours(pdir, e["req"]), self.fleet
            )
            if sess is None or e["type"] not in pool_src.TYPE_MAP:
                continue
            if e["type"] == "released":
                held.pop(e["req"], None)
            if e["type"] == "ended":
                held[e["req"]] = {
                    **e,
                    "_session": sess.session_id,
                    "_seen": self.now,
                    "_path": path,
                }
                continue
            self._pool_event(e, sess, path)
        by_id = {s.session_id: s for s in self.fleet}
        for req, e in list(held.items()):
            if self.now - e["_seen"] >= ENDED_GRACE_S:
                held.pop(req)
                if e["_session"] in by_id:
                    self._pool_event(e, by_id[e["_session"]], e["_path"])

    def _pool_event(self, e: dict, sess, path: str) -> None:
        what, sev = pool_src.TYPE_MAP[e["type"]]
        self.emit(
            sev,
            f"o11y.machine.{what}",
            f"pool {e['req']}: {e['type']}: {str(e.get('msg', ''))[:240]}",
            {
                "o11y.request_id": e["req"],
                "o11y.cluster": e.get("cluster"),
                "o11y.at": e["ts"],
                "evidence": [{"path": path, "offset": e["_offset"]}],
            },
            sess,
            dedup=f"pool:{e['req']}:{what}:{e['ts']}",
        )

    # ------------------------------------------------------------------ ledger, KernelForge, e2e-flow
    def ledger(self) -> None:
        recs, invalid = door.api.read_ledger(self.root)
        if invalid > self.state.get("ledger_invalid", 0):  # review C4: a diagnostic, not an event
            print(f"o11y log: ledger has {invalid} invalid line(s); skipped", file=sys.stderr)
        self.state["ledger_invalid"] = invalid
        by_id = {s.session_id: s for s in self.fleet}
        for r in recs:
            w = r.get("writer") or {}
            if r["kind"] not in ("incident", "finding") or w.get("source") != "written":
                continue
            sess = by_id.get(w.get("session_id"))
            attrs = {
                "o11y.record_id": r["id"],
                **({} if sess else {"session.id": w.get("session_id")}),
            }
            if r["kind"] == "incident":
                self.emit(
                    SEVERITY[r["severity"]],
                    "o11y.agent.incident",
                    f"[{r['category']}] {r['summary']}"[:500],
                    {**attrs, "o11y.category": r["category"]},
                    sess,
                    once=("ledger", r["id"], 5000),
                    history=True,
                )
            else:
                self.emit(
                    "INFO",
                    "o11y.agent.finding",
                    r["text"][:500],
                    {**attrs, "o11y.evidence_rank": r["evidence_rank"]},
                    sess,
                    once=("ledger", r["id"], 5000),
                    history=True,
                )
        camps = [r for r in recs if r["kind"] == "campaign"]
        dirs, bad, notes = forge_src.discover(camps, [s.launch_dir for s in self.fleet])
        for root, why in bad:
            self.problems.add("forge", root, "campaign_root", why)
        for root, note in notes:
            if note.startswith(
                "untrusted_path: "
            ):  # read anyway (display data); said once per root
                self.emit(
                    "WARN",
                    "o11y.source.untrusted_path",
                    f"{root}: {note[16:]}",
                    {"o11y.path": root},
                    once=("untrusted_path", root),
                )
            elif self.once("forge_notes", note):
                print(f"o11y log: forge {note}", file=sys.stderr)
        stale = self.state.setdefault("forge_stale", {})
        for d in dirs:
            c = forge_src.read_campaign(
                d, self.problems, now=self.now, stale_s=self.cfg["forge_stale_s"]
            )
            if c and c["stale"] and not stale.get(c["campaign_id"]):
                stale[c["campaign_id"]] = True
                self.emit(
                    "WARN",
                    "o11y.campaign.stale",
                    f"campaign {c['operator']} says running but events.jsonl is unchanged for {c['idle_s'] // 60} min",
                    {
                        "o11y.campaign_id": c["campaign_id"],
                        "evidence": [{"path": d + "/events.jsonl"}],
                    },
                    dedup=f"campaign_stale:{c['campaign_id']}",
                )
            elif c:
                stale.pop(c["campaign_id"], None)
        globs = self.cfg["sources"]["e2e_flow_globs"]
        if globs:  # read here only for the drift checks; the values are o11y_facts' business
            e2e_src.hot_ops(globs, self.problems)
            e2e_src.integration_reports(globs, self.problems)

    # ------------------------------------------------------------------ OTel copy (optional)
    def otel(self) -> None:
        h = read_private(self.root / "otel" / "health.json", None)
        if isinstance(h, dict) and isinstance(h.get("outages"), int):
            last = self.state.get("otel_outages")
            if last is not None and h["outages"] > last:
                since = h.get("down_since") or h.get("last_change") or h.get("ts")
                self.emit(
                    "ERROR",
                    "o11y.otel.collector_down",
                    f"local OTel collector down{' since ' + since if since else ''} "
                    f"({h['outages'] - last} new outage(s)); telemetry is lost while it is down",
                    {
                        "o11y.outages": h["outages"],
                        "o11y.down_since": h.get("down_since"),
                        "evidence": [{"path": str(self.root / "otel" / "health.json")}],
                    },
                    dedup=f"otel_down:{h['outages']}",
                )
            self.state["otel_outages"] = h["outages"]
        odir = self.cfg["sources"]["otel_dir"]
        if not odir or not os.path.isdir(odir):
            return
        files, unsafe = otel_src.private_jsonl_files(odir)
        for path, why in unsafe:
            self.problems.add("otel", path, "(file)", f"not trusted: {why}")
        sids = {s.session_id: s for s in self.fleet}
        for path in files:
            ino = os.stat(path).st_ino
            key = f"otel:{ino}"
            chunk = read_new(self.cursors.get(key, path, follow_rename=True), opener=owned_open)
            if chunk.problem:
                self.problems.add("otel", path, "(file)", f"not trusted: {chunk.problem}")
            self.pending.append((key, chunk.cursor))
            for r in otel_src.records(path, chunk.lines, self.problems):
                s = sids.get(r["attrs"].get("session.id"))
                if s is not None and r["event"] == "api_retries_exhausted":
                    a = r["attrs"]
                    self.emit(
                        "ERROR",
                        "o11y.api.retries_exhausted",
                        f"{s.label}: API failed after {a.get('total_attempts')} attempts: {str(a.get('error'))[:200]}",
                        {
                            "o11y.status": a.get("status_code"),
                            "evidence": [{"path": path, "offset": r["offset"]}],
                        },
                        s,
                        dedup=f"otel:{ino}:{r['offset']}",
                        source="otel",
                        history=True,
                    )

    # ------------------------------------------------------------------ source problems
    def report_problems(self) -> None:
        seen = self.state.setdefault("mismatch_seen", {})
        for p in {p.key: p for p in self.problems.items}.values():
            sig = _sig(p.path)
            if seen.get(p.key) == sig:
                continue  # unchanged file: reported already (spec §6.6)
            seen[p.key] = sig
            self.emit(
                "ERROR",
                "o11y.source.schema_mismatch",
                f"{p.source}: {p.path}: field '{p.field}' {p.detail}".strip(),
                {"o11y.source_kind": p.source, "o11y.path": p.path, "o11y.field": p.field},
                dedup=f"mismatch:{p.key}:{sig}",
            )

    def run(self, registry_problem, config_problems, config_new) -> None:
        self.registry(registry_problem, config_problems, config_new)
        self.histories()
        self.tokens()
        self.pool()
        self.ledger()
        self.otel()
        self.report_problems()

    def commit(self) -> None:
        for key, cur in self.pending:
            self.cursors.put(key, cur)
        self.pending.clear()
        self.state.setdefault("baseline_at", self.now)
        write_private(self.state_path, self.state)
