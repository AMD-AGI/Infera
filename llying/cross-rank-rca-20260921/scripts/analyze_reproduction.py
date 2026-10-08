#!/usr/bin/env python3
"""Correlate one AgentX run across client, Router, SGLang, Mooncake, and HCA."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path


IMPORTANT_COUNTERS = (
    "retry_excd",
    "ack_timeout",
    "retx",
    "rnr",
    "nak",
    "remote_access",
    "out_of_buffer",
    "invalid",
)
HARD_FAULT_COUNTERS = ("req_tx_retry_excd_err", "tx_rdma_ack_timeout")
# SGLang logs a request the client cancels while its KV is in flight as a
# transfer failure with this reason; it is not a transport failure.
CLIENT_ABORT = "aborted by abortreq"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OUTER_TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z")


def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def worker_log(run: Path, role: str) -> str:
    final = run / "logs" / f"{role}-final.log"
    if final.exists():
        return text(final)
    return text(run / "launch" / "server-logs" / f"{role}-0.log")


def timestamp_from_line(line: str) -> str | None:
    match = OUTER_TS_RE.search(line)
    return f"{match.group(1)}+00:00" if match else None


def epoch(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    # Python 3.10 fromisoformat needs 6 fraction digits and no "Z"; docker writes
    # RFC3339Nano ("...53.7Z", "...53.700000001Z").
    timestamp = re.sub(
        r"\.(\d+)", lambda m: "." + (m.group(1) + "000000")[:6], timestamp
    ).replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(timestamp).timestamp()
    except ValueError:
        return None


def rank_from_line(line: str) -> int | None:
    for pattern in (r"\brank[=:\s]+(\d+)", r"\bDP(\d+)\b"):
        match = re.search(pattern, line)
        if match:
            return int(match.group(1))
    return None


def request_id_from_line(line: str) -> str | None:
    match = re.search(r"(?:decode_req\.)?req\.rid='([^']+)'", line)
    return match.group(1) if match else None


def bootstrap_from_line(line: str) -> str | None:
    match = re.search(r"(?:bootstrap_room[=:\s]+|bootstrap room is\s+)(\d+)", line)
    return match.group(1) if match else None


def parse_transfer_lines(prefill: str, decode: str) -> list[dict]:
    by_bootstrap: dict[str, dict] = {}
    for role, body in (("prefill", prefill), ("decode", decode)):
        for line in body.splitlines():
            lower = line.lower()
            if "transfer failed" not in lower or CLIENT_ABORT in lower:
                continue
            bootstrap = bootstrap_from_line(line)
            if not bootstrap:
                continue
            record = by_bootstrap.setdefault(
                bootstrap, {"bootstrap_room": bootstrap, "lines": []}
            )
            rank = rank_from_line(line)
            if rank is not None:
                record[f"{role}_rank"] = rank
            request_id = request_id_from_line(line)
            if request_id:
                record[f"{role}_request_id"] = request_id
            timestamp = timestamp_from_line(line)
            if timestamp:
                record[f"{role}_timestamp"] = timestamp
            record["lines"].append({"role": role, "line": line})
            if role == "prefill":
                if "remote mooncake session" in lower and "not alive" in lower:
                    record["failure_class"] = "session-cascade"
                elif "failed to send" in lower:
                    record["failure_class"] = "primary-send"
    records = []
    for record in by_bootstrap.values():
        p_rank = record.get("prefill_rank")
        d_rank = record.get("decode_rank")
        if p_rank is not None and d_rank is not None:
            record["rank_class"] = "same-rank" if p_rank == d_rank else "cross-rank"
            record["numa_class"] = (
                "same-numa" if (p_rank < 4) == (d_rank < 4) else "cross-numa"
            )
        records.append(record)
    return sorted(records, key=lambda value: int(value["bootstrap_room"]))


def transfer_failure_lines(body: str, role: str) -> list[str]:
    return [line for line in body.lower().splitlines() if f"{role} transfer failed" in line]


def parse_session_failures(prefill: str) -> list[dict]:
    pattern = re.compile(r"\bDP(\d+)\b.*Session ([^:\s]+):(\d+) failed\.")
    records = []
    for line in prefill.splitlines():
        match = pattern.search(line)
        if match:
            records.append(
                {
                    "timestamp": timestamp_from_line(line),
                    "prefill_rank": int(match.group(1)),
                    "peer_ip": match.group(2),
                    "peer_port": int(match.group(3)),
                    "line": line,
                }
            )
    return records


def parse_router_pairs(router: str) -> list[dict]:
    pending: deque[int] = deque()
    pairs = []
    pick = re.compile(r"\brole=(Prefill|Decode)\b.*?\bpicked=\S+#dp(\d+)\b")
    for raw in router.splitlines():
        match = pick.search(ANSI_RE.sub("", raw))
        if not match:
            continue
        role, rank = match.group(1), int(match.group(2))
        if role == "Prefill":
            pending.append(rank)
        elif pending:
            p_rank = pending.popleft()
            pairs.append(
                {
                    "prefill_rank": p_rank,
                    "decode_rank": rank,
                    "rank_class": "same-rank" if p_rank == rank else "cross-rank",
                    "numa_class": (
                        "same-numa" if (p_rank < 4) == (rank < 4) else "cross-numa"
                    ),
                }
            )
    return pairs


def listening_ports(body: str) -> set[int]:
    pattern = re.compile(
        r"Transfer Engine RPC using P2P handshake, listening on [^:\s]+:(\d+)"
    )
    return {int(value) for value in pattern.findall(body)}


def decode_port_map(decode: str, transfers: list[dict]) -> tuple[dict[int, int], dict]:
    votes: dict[int, Counter] = defaultdict(Counter)
    for record in transfers:
        d_rank = record.get("decode_rank")
        if d_rank is None:
            continue
        for item in record["lines"]:
            if item["role"] != "prefill":
                continue
            match = re.search(r"remote mooncake session [^:\s]+:(\d+)", item["line"])
            if match:
                votes[int(match.group(1))][d_rank] += 1
    mapping = {}
    evidence = {}
    for port, counts in votes.items():
        rank, count = counts.most_common(1)[0]
        if len(counts) != 1:
            raise ValueError(f"conflicting rank evidence for decode port {port}: {counts}")
        mapping[port] = rank
        evidence[str(port)] = {"rank": rank, "method": "session-cascade", "samples": count}
    ports = listening_ports(decode)
    missing_ports = ports - set(mapping)
    missing_ranks = set(range(8)) - set(mapping.values())
    if len(missing_ports) == len(missing_ranks) == 1:
        port = missing_ports.pop()
        rank = missing_ranks.pop()
        mapping[port] = rank
        evidence[str(port)] = {"rank": rank, "method": "elimination", "samples": 0}
    return mapping, evidence


def parse_cqe_events(prefill: str, port_map: dict[int, int]) -> list[dict]:
    pattern = re.compile(
        r"Worker: Process failed for slice "
        r"\(opcode: (?P<opcode>\d+), source_addr: (?P<source>0x[0-9a-f]+), "
        r"length: (?P<length>\d+), dest_addr: (?P<dest>0x[0-9a-f]+), "
        r"local_nic: (?P<hca>ionic_\d+), "
        r"peer_nic: (?P<peer_ip>[^:\s]+):(?P<peer_port>\d+)@(?P<peer_hca>ionic_\d+).*"
        r"transport retry counter exceeded"
    )
    events = []
    for line in prefill.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        values = match.groupdict()
        port = int(values["peer_port"])
        events.append(
            {
                "timestamp": timestamp_from_line(line),
                "opcode": int(values["opcode"]),
                "length": int(values["length"]),
                "source_addr": values["source"],
                "destination_addr": values["dest"],
                "local_hca": values["hca"],
                "peer_ip": values["peer_ip"],
                "peer_port": port,
                "peer_hca": values["peer_hca"],
                "destination_rank": port_map.get(port),
                "same_named_hca": values["hca"] == values["peer_hca"],
                "line": line,
            }
        )
    return events


def flatten_counters(payload: dict) -> dict[tuple[str, str, str, str], int]:
    flat = {}
    for node, hcas in payload.get("nodes", {}).items():
        for hca, record in hcas.items():
            for source in ("sysfs", "ethtool"):
                values = record.get(source, {})
                if source == "ethtool":
                    values = {
                        f"{netdev}:{name}": value
                        for netdev, stats in values.items()
                        for name, value in stats.items()
                    }
                for name, value in values.items():
                    if isinstance(value, int):
                        flat[(node, hca, source, name)] = value
    return flat


def counter_deltas(run: Path) -> list[dict]:
    before_path = run / "counters" / "before.json"
    after_path = run / "counters" / "after.json"
    if not before_path.exists() or not after_path.exists():
        return []
    before = flatten_counters(json.loads(text(before_path)))
    after = flatten_counters(json.loads(text(after_path)))
    rows = []
    for key in sorted(set(before) & set(after)):
        delta = after[key] - before[key]
        if delta and any(token in key[3].lower() for token in IMPORTANT_COUNTERS):
            rows.append(
                {
                    "node": key[0],
                    "hca": key[1],
                    "source": key[2],
                    "counter": key[3],
                    "before": before[key],
                    "after": after[key],
                    "delta": delta,
                }
            )
    return rows


def timeseries_fault_events(run: Path) -> list[dict]:
    path = run / "counters" / "timeseries.jsonl"
    if not path.exists():
        return []
    previous: dict[tuple[str, str, str], int] = {}
    events = []
    for raw in path.open(encoding="utf-8", errors="replace"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        timestamp = payload.get("captured_at")
        for node, hcas in payload.get("nodes", {}).items():
            for hca, record in hcas.items():
                for counter in HARD_FAULT_COUNTERS:
                    value = record.get("sysfs", {}).get(counter)
                    if not isinstance(value, int):
                        continue
                    key = (node, hca, counter)
                    if key in previous and value != previous[key]:
                        events.append(
                            {
                                "timestamp": timestamp,
                                "node": node,
                                "hca": hca,
                                "counter": counter,
                                "before": previous[key],
                                "after": value,
                                "delta": value - previous[key],
                            }
                        )
                    previous[key] = value
    return events


def client_errors(run: Path) -> list[dict]:
    records = []
    for path in (run / "bench").glob("**/profile_export.jsonl"):
        for raw in path.open(encoding="utf-8", errors="replace"):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if record.get("error") or record.get("error_type") or record.get("success") is False:
                records.append(record)
    return records


def correlate_client_errors(
    errors: list[dict], transfers: list[dict], window_seconds: float = 5.0
) -> list[dict]:
    server = []
    for index, record in enumerate(transfers):
        when = epoch(record.get("decode_timestamp"))
        if when is not None:
            server.append((index, when))
    unused = {index for index, _ in server}
    server_time = dict(server)
    matches = []
    ordered_errors = sorted(
        enumerate(errors),
        key=lambda item: item[1].get("metadata", {}).get("request_end_ns", 0),
    )
    for error_index, record in ordered_errors:
        end_ns = record.get("metadata", {}).get("request_end_ns")
        if not isinstance(end_ns, int) or not unused:
            continue
        ended = end_ns / 1e9
        candidate = min(unused, key=lambda index: abs(server_time[index] - ended))
        delta = server_time[candidate] - ended
        if abs(delta) <= window_seconds:
            unused.remove(candidate)
            matches.append(
                {
                    "client_error_index": error_index,
                    "client_request_id": record.get("metadata", {}).get("x_request_id"),
                    "bootstrap_room": transfers[candidate]["bootstrap_room"],
                    "prefill_rank": transfers[candidate].get("prefill_rank"),
                    "decode_rank": transfers[candidate].get("decode_rank"),
                    "delta_ms": round(delta * 1000, 3),
                }
            )
    return matches


def benchmark_summary(run: Path) -> dict:
    paths = sorted((run / "bench").glob("agentx_conc*.json"))
    if not paths:
        return {}
    try:
        payload = json.loads(text(paths[-1]))
    except json.JSONDecodeError:
        return {}
    accounting = payload.get("request_accounting", {})
    throughput = payload.get("request_metrics", {}).get("throughput", {})
    cancelled = re.findall(
        r"Phase profiling \(profiling\) complete \|.*?cancelled=([\d,]+)",
        text(run / "bench" / "benchmark.log"),
    )
    return {
        "cancelled_requests": int(cancelled[-1].replace(",", "")) if cancelled else None,
        "records_total": accounting.get("records_total"),
        "records_profiled": accounting.get("records_profiled"),
        "records_error_dropped": accounting.get("records_error_dropped"),
        "error_categories": accounting.get("error_categories", {}),
        "duration_seconds": throughput.get("duration_seconds"),
        "input_tokens_per_second": throughput.get("input", {}).get("tokens_per_second"),
        "output_tokens_per_second": throughput.get("output", {}).get("tokens_per_second"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    analysis = run / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    prefill = worker_log(run, "prefill")
    decode = worker_log(run, "decode")
    router = worker_log(run, "router")
    transfer_records = parse_transfer_lines(prefill, decode)
    prefill_transfer_lines = transfer_failure_lines(prefill, "prefill")
    decode_transfer_lines = transfer_failure_lines(decode, "decode")
    session_failures = parse_session_failures(prefill)
    router_pairs = parse_router_pairs(router)
    deltas = counter_deltas(run)
    fault_events = timeseries_fault_events(run)
    errors = client_errors(run)
    client_matches = correlate_client_errors(errors, transfer_records)
    port_map, port_evidence = decode_port_map(decode, transfer_records)
    cqe_events = parse_cqe_events(prefill, port_map)
    paired_failures = [
        record
        for record in transfer_records
        if record.get("prefill_rank") is not None and record.get("decode_rank") is not None
    ]
    failure_pairs = Counter(
        (record["prefill_rank"], record["decode_rank"]) for record in paired_failures
    )
    router_classes = Counter(
        (pair["rank_class"], pair["numa_class"]) for pair in router_pairs
    )
    hard_counter_totals = Counter()
    hard_counter_hcas = defaultdict(Counter)
    hard_counter_nodes = defaultdict(Counter)
    for row in deltas:
        if row["counter"] in HARD_FAULT_COUNTERS:
            hard_counter_totals[row["counter"]] += row["delta"]
            hard_counter_hcas[row["counter"]][row["hca"]] += row["delta"]
            hard_counter_nodes[row["counter"]][row["node"]] += row["delta"]
    routed_cross_numa = router_classes[("cross-rank", "cross-numa")]
    failed_cross_numa = sum(
        record.get("numa_class") == "cross-numa" for record in paired_failures
    )
    summary = {
        "run": str(run),
        "benchmark": benchmark_summary(run),
        "signatures": {
            "cqe_error_12": prefill.count("cqe with error 12")
            + decode.count("cqe with error 12"),
            "retry_exhausted": prefill.count("transport retry counter exceeded")
            + decode.count("transport retry counter exceeded"),
            "prefill_transfer_failed": sum(
                CLIENT_ABORT not in line for line in prefill_transfer_lines
            ),
            "decode_transfer_failed": sum(
                CLIENT_ABORT not in line for line in decode_transfer_lines
            ),
            "transfer_client_aborted": sum(
                CLIENT_ABORT in line
                for line in prefill_transfer_lines + decode_transfer_lines
            ),
            "session_failed": len(session_failures),
            "session_not_alive": prefill.lower().count("remote mooncake session")
            + decode.lower().count("remote mooncake session"),
            "client_errors": len(errors),
        },
        "client_server_correlation": {
            "matched_within_5s": len(client_matches),
            "max_abs_delta_ms": max(
                (abs(record["delta_ms"]) for record in client_matches), default=None
            ),
            "unmatched_client_errors": len(errors) - len(client_matches),
            "unmatched_decode_failures": len(transfer_records) - len(client_matches),
        },
        "transfer_failure_records": len(transfer_records),
        "paired_transfer_failures": len(paired_failures),
        "paired_failure_rank_classes": dict(
            Counter(record["rank_class"] for record in paired_failures)
        ),
        "paired_failure_numa_classes": dict(
            Counter(record["numa_class"] for record in paired_failures)
        ),
        "paired_failure_causes": dict(
            Counter(record.get("failure_class", "decode-only") for record in transfer_records)
        ),
        "failure_rank_pairs": {
            f"P{p_rank}->D{d_rank}": count
            for (p_rank, d_rank), count in sorted(failure_pairs.items())
        },
        "router_pair_count": len(router_pairs),
        "router_pair_classes": {
            f"{rank_class}/{numa_class}": count
            for (rank_class, numa_class), count in sorted(router_classes.items())
        },
        "cross_numa_failure_rate": {
            "failed": failed_cross_numa,
            "routed": routed_cross_numa,
            "percent": (
                round(failed_cross_numa / routed_cross_numa * 100, 3)
                if routed_cross_numa
                else None
            ),
            "paired_same_numa_failures": sum(
                record.get("numa_class") == "same-numa" for record in paired_failures
            ),
        },
        "cqe_hcas": dict(Counter(record["local_hca"] for record in cqe_events)),
        "cqe_destination_ranks": dict(
            sorted(
                Counter(
                    str(record["destination_rank"])
                    for record in cqe_events
                    if record["destination_rank"] is not None
                ).items()
            )
        ),
        "cqe_same_named_hca": sum(record["same_named_hca"] for record in cqe_events),
        "decode_port_rank_evidence": port_evidence,
        "hca_hard_counter_totals": dict(hard_counter_totals),
        "hca_hard_counter_by_hca": {
            name: dict(values) for name, values in hard_counter_hcas.items()
        },
        "hca_hard_counter_by_node": {
            name: dict(values) for name, values in hard_counter_nodes.items()
        },
        "hca_fault_delta_rows": len(deltas),
        "hca_timeseries_fault_events": len(fault_events),
        "hca_timeseries_fault_events_by_counter": dict(
            Counter(record["counter"] for record in fault_events)
        ),
        "root_cause_status": "not-closed; single-variable pair matrix pending",
    }
    summary["reproduced"] = bool(
        summary["signatures"]["cqe_error_12"]
        and summary["signatures"]["retry_exhausted"]
        and summary["paired_failure_numa_classes"].get("cross-numa")
    )
    (analysis / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, payload in (
        ("transfer-failures.json", transfer_records),
        ("session-failures.json", session_failures),
        ("router-pairs.json", router_pairs),
        ("client-errors.json", errors),
        ("client-server-matches.json", client_matches),
        ("cqe-events.json", cqe_events),
    ):
        (analysis / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    pair_rows = [
        {"prefill_rank": p_rank, "decode_rank": d_rank, "failures": count}
        for (p_rank, d_rank), count in sorted(failure_pairs.items())
    ]
    write_csv(analysis / "failure-rank-pairs.csv", pair_rows)
    write_csv(analysis / "hca-counter-deltas.csv", deltas)
    write_csv(analysis / "hca-timeseries-fault-events.csv", fault_events)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
