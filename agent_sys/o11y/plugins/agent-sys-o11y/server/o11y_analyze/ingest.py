"""Deterministic ingestion and detectors (design §8.2, §8.3). No model here.

Runs under the `ingest` lock in every analyzer kind, so the `ingest:*` cursors have one writer. Detectors
are idempotent through `state/analyze/detector.json` (what was already reported) and every event carries
`o11y.dedup_key`. Events are appended before cursors and state are committed: a crash in between can repeat
an event (same dedup key), never lose one.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import core
from .cursor import CursorStore, read_new
from .sources import Problems
from .sources import forge as forge_src
from .sources import otel as otel_src
from .sources import pool as pool_src
from .sources import registry as reg_src
from .sources import transcripts as tr_src

WAIT_REASONS = {"permission prompt", "sandbox request", "worker request"}  # spec §6.1 / design §8.3
ENDED_GRACE_S = 600  # an outbox `ended` waits this long for a `released` of the same request


def _sig(path: str) -> str:
    """File signature for "repeat only after the file changes" (spec §6.6)."""
    try:
        st = os.stat(path)
        return f"{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return "missing"


class Ingestor:
    def __init__(self, root: Path, cfg: dict, now: float | None = None):
        self.root = Path(root)
        self.cfg = cfg
        self.th = cfg["thresholds"]
        self.now = now or time.time()
        self.cursors = CursorStore(root)
        self.state_path = self.root / "state" / "analyze" / "detector.json"
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            self.state = {}
        self.problems = Problems()
        self.pending_cursors: list[tuple[str, object]] = []
        self.events: list[dict] = []
        self.emit_failures: list[str] = []
        self.main_alive: bool | None = None

    # ------------------------------------------------------------------ helpers
    def emit(
        self,
        severity: str,
        name: str,
        body: str,
        attrs: dict | None = None,
        sess=None,
        dedup: str | None = None,
        source: str = "detector",
    ) -> None:
        a = {}
        if sess is not None:
            a.update({"session.id": sess.session_id, "session.role": sess.role})
            if sess.name:
                a["session.name"] = sess.name
        a.update({k: v for k, v in (attrs or {}).items() if v is not None})
        if dedup:
            a["o11y.dedup_key"] = dedup
        try:
            core.append_event(
                self.root,
                severity=severity,
                event_name=name,
                body=body,
                attributes=a,
                source=source,
            )
        except core.ValidationError as e:
            self.emit_failures.append(f"{name}: {e.problems[:2]}")
            return
        self.events.append(
            {"severity": severity, "event_name": name, "body": body, "attributes": a}
        )

    def commit(self) -> None:
        for key, cur in self.pending_cursors:
            self.cursors.put(key, cur)
        self.pending_cursors.clear()
        core.write_json_atomic(self.state_path, self.state)

    def _history_reset(self, path: str, chunk, sess=None) -> None:
        if chunk.reset:
            self.emit(
                "INFO",
                "o11y.protocol.history_reset",
                f"{Path(path).name} was replaced or rewritten; detectors re-scanned it from the start",
                {"evidence": [{"path": path}]},
                sess,
                dedup=f"reset:{path}:{chunk.cursor.inode}:{self.now:.0f}",
            )

    # ------------------------------------------------------------------ registry
    def registry(self) -> list:
        fleet = reg_src.read_fleet(self.root, self.cfg, self.problems)
        prev = self.state.get("sessions")
        first = prev is None
        prev = prev or {}
        cur = {}
        by_pid_prev = {v.get("pid"): sid for sid, v in prev.items() if v.get("pid")}
        crashed = self.state.setdefault("crashed", [])
        for s in fleet:
            cur[s.session_id] = {
                "pid": s.pid,
                "alive": s.alive,
                "status": s.status,
                "name": s.name,
                "role": s.role,
            }
            if first or s.session_id in prev:
                continue
            old = by_pid_prev.get(s.pid) or next((x for x in s.lineage if x in prev), None)
            if old and old != s.session_id:
                self.emit(
                    "INFO",
                    "o11y.session.cleared",
                    f"{s.label}: /clear, new session id",
                    {"o11y.previous_session_id": old},
                    s,
                    dedup=f"cleared:{s.session_id}",
                )
            elif s.alive is not False:
                self.emit(
                    "INFO",
                    "o11y.session.started",
                    f"{s.label} started ({s.role})",
                    {},
                    s,
                    dedup=f"started:{s.session_id}",
                )
        # A dead session (registry file left behind, pid gone or reused). For a main this is seen when a main of
        # the same fleet next runs the analyzer (spec §1.3, criterion 11).
        for s in fleet:
            if s.alive is False and s.session_id not in crashed:
                since = s.status_since_epoch
                if since and self.now - since < self.th["crash_grace_s"]:
                    continue
                crashed.append(s.session_id)
                self.emit(
                    "ERROR",
                    "o11y.session.crashed",
                    f"{s.label} ({s.role}) is gone without a clean exit (pid {s.pid} dead, registry file left behind)",
                    {"o11y.pid": s.pid, "o11y.tmux": s.tmux},
                    s,
                    dedup=f"crashed:{s.session_id}",
                )
        del crashed[:-200]
        if not first:
            live_pids = {v.get("pid") for v in cur.values()}
            for sid, v in prev.items():
                if (
                    sid in cur
                    or sid in crashed
                    or v.get("alive") is False
                    or v.get("pid") in live_pids
                ):
                    continue
                self.emit(
                    "INFO",
                    "o11y.session.ended",
                    f"{v.get('name') or sid[:8]} ended",
                    {
                        "session.id": sid,
                        "session.role": v.get("role"),
                        "session.name": v.get("name"),
                    },
                    dedup=f"ended:{sid}",
                )
        alerted = self.state.setdefault("status_alerts", {})
        for s in fleet:
            since = s.status_since_epoch
            if s.alive is False or since is None:
                continue
            age = self.now - since
            key = f"{s.session_id}|{s.status}|{s.status_since}"
            if key in alerted:
                continue
            if (
                s.status == "waiting"
                and (s.waiting_for or "") in WAIT_REASONS
                and age > self.th["perm_wait_s"]
            ):
                alerted[key] = self.now
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
                )
            elif s.status == "idle" and age > self.th["idle_warn_s"]:
                alerted[key] = self.now
                self.emit(
                    "WARN",
                    "o11y.session.idle",
                    f"{s.label} idle for {int(age // 60)} min",
                    {"o11y.idle_since": s.status_since},
                    s,
                    dedup=f"idle:{key}",
                )
        for k in [k for k, t in alerted.items() if self.now - t > 7 * 86400]:
            del alerted[k]
        self.state["sessions"] = cur
        mains = [s for s in fleet if s.role == "main"]
        self.main_alive = any(s.alive is not False for s in mains) if mains else None
        return fleet

    # ------------------------------------------------------------------ histories
    def transcripts(self, fleet: list) -> dict:
        usage_all = self.state.setdefault("usage", {})
        comp_seen = self.state.setdefault("compactions_seen", [])
        api_alerted = self.state.setdefault("api_alerted", {})
        fatal_seen = self.state.setdefault("api_fatal_seen", [])
        steps = self.state.setdefault("token_steps", {})
        step = int(self.th["token_step"])
        for s in fleet:
            if not s.transcript_path or not Path(s.transcript_path).exists():
                continue
            tokens = {k: 0 for k in tr_src.USAGE_KEYS.values()}
            messages = 0
            first_ts = last_ts = None
            files = tr_src.session_files(s.transcript_path)
            for path in files:
                key = f"ingest:{path}"
                chunk = read_new(self.cursors.get(key, path))
                self._history_reset(path, chunk, s)
                extra = chunk.cursor.extra
                ing = tr_src.ingest_lines(path, chunk.lines, extra, self.problems)
                if ing.last_ts:
                    extra["last_ts"] = ing.last_ts
                self.pending_cursors.append((key, chunk.cursor))
                for k, v in extra.get("usage", {}).items():
                    tokens[k] = tokens.get(k, 0) + v
                messages += extra.get("messages", 0)
                if extra.get("first_ts") and (first_ts is None or extra["first_ts"] < first_ts):
                    first_ts = extra["first_ts"]
                if extra.get("last_ts") and (last_ts is None or extra["last_ts"] > last_ts):
                    last_ts = extra["last_ts"]
                for c in ing.compactions:
                    if c["uuid"] in comp_seen:
                        continue
                    comp_seen.append(c["uuid"])
                    ev = {
                        "path": path,
                        "offset": c["offset"],
                        **({"uuid": c["uuid"]} if c["uuid"] else {}),
                    }
                    self.emit(
                        "INFO",
                        "o11y.compaction.done",
                        f"{s.label} compacted ({c['trigger']}): {c['pre_tokens']} -> {c['post_tokens']} tokens",
                        {
                            "o11y.trigger": c["trigger"],
                            "o11y.pre_tokens": c["pre_tokens"],
                            "o11y.post_tokens": c["post_tokens"],
                            "o11y.at": c["ts"],
                            "evidence": [ev],
                        },
                        s,
                        dedup=f"compaction:{c['uuid']}",
                    )
                for e in ing.api_errors:
                    k = f"{path}:{e['offset']}"
                    if e["fatal"] and k not in fatal_seen:
                        fatal_seen.append(k)
                        self.emit(
                            "ERROR",
                            "o11y.api.error",
                            f"{s.label}: fatal API error ({e['error']}, {e['status']}): {e['text'][:200]}",
                            {
                                "o11y.error_kind": e["error"],
                                "o11y.status": e["status"],
                                "evidence": [
                                    {
                                        "path": path,
                                        "offset": e["offset"],
                                        **({"uuid": e["uuid"]} if e["uuid"] else {}),
                                    }
                                ],
                            },
                            s,
                            dedup=f"api_fatal:{k}",
                        )
                if ing.api_error_streak_max >= int(self.th["api_error_k"]) and not api_alerted.get(
                    path
                ):
                    api_alerted[path] = True
                    last = ing.api_errors[-1]
                    self.emit(
                        "ERROR",
                        "o11y.api.error_repeated",
                        f"{s.label}: {ing.api_error_streak_max} consecutive API errors, last {last['status']}: {last['text'][:200]}",
                        {
                            "o11y.count": ing.api_error_streak_max,
                            "o11y.status": last["status"],
                            "evidence": [
                                {
                                    "path": path,
                                    "offset": e["offset"],
                                    **({"uuid": e["uuid"]} if e["uuid"] else {}),
                                }
                                for e in ing.api_errors[-3:]
                            ],
                        },
                        s,
                        dedup=f"api:{path}:{last['offset']}",
                    )
                if extra.get("api_err_streak", 0) == 0:
                    api_alerted.pop(path, None)
            del comp_seen[:-500]
            del fatal_seen[:-500]
            new = tr_src.new_total(tokens)
            done = steps.get(s.session_id)
            n = new // step
            if done is None:
                steps[s.session_id] = n  # first sight: the baseline, not reported (spec §6.2)
            elif n > done:
                steps[s.session_id] = n
                self.emit(
                    "INFO",
                    "o11y.tokens.crossed",
                    f"{s.label} passed {n * step / 1e6:g}M new tokens (input + cache creation + output)",
                    {
                        "o11y.scope": "session",
                        "o11y.tokens_new": new,
                        "o11y.tokens_cache_read": tokens["cache_read"],
                        "o11y.step": step,
                    },
                    s,
                    dedup=f"tokens:{s.session_id}:{n}",
                )
            usage_all[s.session_id] = {
                "tokens": {**tokens, "new": new},
                "messages": messages,
                "subagent_files": len(files) - 1,
                "name": s.name,
                "role": s.role,
                "first_at": first_ts,
                "last_at": last_ts,
                "steps_crossed": steps[s.session_id],
            }
        fleet_ids = {s.session_id for s in fleet}
        ft = {k: 0 for k in [*tr_src.USAGE_KEYS.values(), "new"]}
        msgs = 0
        for sid, u in usage_all.items():
            if sid in fleet_ids:
                for k in ft:
                    ft[k] += u["tokens"].get(k, 0)
                msgs += u.get("messages", 0)
        n = ft["new"] // step
        done = steps.get("_fleet")
        if done is None:
            steps["_fleet"] = n
        elif n > done:
            steps["_fleet"] = n
            self.emit(
                "INFO",
                "o11y.tokens.crossed",
                f"fleet passed {n * step / 1e6:g}M new tokens",
                {
                    "o11y.scope": "fleet",
                    "o11y.tokens_new": ft["new"],
                    "o11y.tokens_cache_read": ft["cache_read"],
                    "o11y.step": step,
                },
                dedup=f"tokens:fleet:{n}",
            )
        return {
            "sessions": {k: v for k, v in usage_all.items() if k in fleet_ids},
            "fleet": {"tokens": ft, "messages": msgs, "steps_crossed": steps["_fleet"]},
        }

    # ------------------------------------------------------------------ machine pool
    def pool(self, fleet: list) -> list[dict]:
        pdir = self.cfg["sources"]["pool_dir"]
        if not pdir or not Path(pdir).is_dir():
            return []
        grants = pool_src.read_grants(pdir, self.problems)
        path = str(Path(pdir) / "outbox.jsonl")
        key = f"ingest:{path}"
        cur = self.cursors.get(key, path)
        first = not cur.inode
        chunk = read_new(cur)
        self.pending_cursors.append((key, chunk.cursor))
        pending = self.state.setdefault("pool_ended", {})  # req -> outbox record held back
        if first:
            return grants  # history before the first pass is the baseline
        req_notify = {g["id"]: g.get("notify") for g in grants}
        for e in pool_src.parse_outbox(path, chunk.lines, self.problems):
            notify = e.get("notify") or req_notify.get(e["req"]) or ""
            pid = reg_src.pid_of_notify(notify)
            sess = next(
                (s for s in fleet if (pid and s.pid == pid) or (notify and s.name == notify)), None
            )
            if sess is None or e["type"] not in pool_src.TYPE_MAP:
                continue
            if e["type"] == "released":
                pending.pop(e["req"], None)
            if e["type"] == "ended":
                pending[e["req"]] = {
                    **e,
                    "_session": sess.session_id,
                    "_seen": self.now,
                    "_path": path,
                }
                continue
            self._pool_event(e, sess, path)
        by_id = {s.session_id: s for s in fleet}
        for req, e in list(pending.items()):
            if self.now - e["_seen"] >= ENDED_GRACE_S:
                pending.pop(req)
                if e["_session"] in by_id:
                    self._pool_event(e, by_id[e["_session"]], e["_path"])
        return grants

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

    # ------------------------------------------------------------------ KernelForge
    def forge(self) -> list[dict]:
        ledger_roots = {
            r["campaign_root"]: r for r in core.query_ledger(self.root, kind="campaign")
        }
        dirs = forge_src.discover(
            list(ledger_roots), self.cfg["sources"].get("campaign_globs") or []
        )
        out = []
        stale_alerted = self.state.setdefault("forge_stale", {})
        for d in dirs:
            c = forge_src.read_campaign(
                d, self.problems, now=self.now, stale_s=self.th["forge_stale_s"]
            )
            if not c:
                continue
            c["ledger"] = next(
                (r for root, r in ledger_roots.items() if d.startswith(root.rstrip("/"))), None
            )
            out.append(c)
            k = c["campaign_id"]
            if c["stale"] and not stale_alerted.get(k):
                stale_alerted[k] = True
                self.emit(
                    "WARN",
                    "o11y.campaign.stale",
                    f"campaign {c['operator']} says running but events.jsonl is unchanged for {c['idle_s'] // 60} min",
                    {
                        "o11y.campaign_id": k,
                        "o11y.forge_dir": d,
                        "evidence": [{"path": d + "/events.jsonl"}],
                    },
                    dedup=f"campaign_stale:{k}",
                )
            elif not c["stale"]:
                stale_alerted.pop(k, None)
        for root in ledger_roots:
            if not any(
                c["forge_dir"].startswith(root.rstrip("/")) for c in out
            ) and not forge_src.resolve_forge_dir(root):
                self.problems.add(
                    "forge",
                    root,
                    "forge_experiments/run_state.json",
                    "ledger campaign_root has no readable forge_experiments state",
                )
        return out

    # ------------------------------------------------------------------ OTel (optional)
    def otel(self, fleet: list) -> dict:
        """`<otel_dir>/*.jsonl`, the collector's rotated set read by name. Cursors are keyed by inode so a
        rotated (renamed) file continues where it was read, and its successor starts at 0."""
        odir = self.cfg["sources"].get("otel_dir")
        if not odir or not Path(odir).is_dir():
            return {}
        sids = {s.session_id: s for s in fleet}
        cost = self.state.setdefault("otel_cost_usd", {})
        for p in sorted(Path(odir).glob("*.jsonl")):
            path = str(p)
            try:
                ino = os.stat(path).st_ino
            except OSError:
                continue
            key = f"ingest-otel:{ino}"
            cur = self.cursors.get(key, path, follow_rename=True)
            chunk = read_new(cur)
            self.pending_cursors.append((key, chunk.cursor))
            for r in otel_src.records(path, chunk.lines, self.problems):
                a = r["attrs"]
                s = sids.get(a.get("session.id"))
                if s is None:
                    continue
                if r["event"] == "api_request":
                    cost[s.session_id] = cost.get(s.session_id, 0.0) + float(a.get("cost_usd") or 0)
                elif r["event"] == "api_retries_exhausted":
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
                    )
        return {"cost_usd": {k: v for k, v in cost.items() if k in sids}}

    # ------------------------------------------------------------------ problems and invalid lines
    def report_problems(self, invalid_files: list[str] | None = None) -> None:
        seen = self.state.setdefault("mismatch_seen", {})
        reported = set()
        for p in self.problems.items:
            if p.key in reported:
                continue
            reported.add(p.key)
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
        for msg in invalid_files or []:
            k = "invalid_file|" + msg
            if seen.get(k):
                continue
            seen[k] = self.now
            self.emit("WARN", "o11y.protocol.invalid_file", msg[:500], {}, dedup=k)

    def report_invalid_lines(self) -> None:
        """Invalid protocol lines are reported when their count grows, once per file per pass."""
        counts = self.state.setdefault("invalid_lines", {})
        from o11y_core import store

        for path, n in (
            (self.root / store.LEDGER, core.ledger_invalid_lines(self.root)),
            (self.root / store.EVENTS, core.events_invalid_lines(self.root)),
        ):
            k = str(path)
            if n > counts.get(k, 0):
                self.emit(
                    "WARN",
                    "o11y.protocol.invalid_line",
                    f"{path.name}: {n - counts.get(k, 0)} new invalid line(s), {n} in total",
                    {"o11y.path": k, "o11y.count": n, "evidence": [{"path": k}]},
                    dedup=f"invalid_line:{k}:{n}",
                )
            counts[k] = n

    def run(self) -> dict:
        fleet = self.registry()
        return {
            "fleet": fleet,
            "usage": self.transcripts(fleet),
            "grants": self.pool(fleet),
            "campaigns": self.forge(),
            "otel": self.otel(fleet),
        }

    def finish(
        self, extra_problems: Problems | None = None, invalid_files: list[str] | None = None
    ) -> None:
        """Report source problems and invalid lines, then commit cursors and detector state."""
        if extra_problems is not None:
            self.problems.items.extend(extra_problems.items)
        self.report_problems(invalid_files)
        self.report_invalid_lines()
        self.commit()
