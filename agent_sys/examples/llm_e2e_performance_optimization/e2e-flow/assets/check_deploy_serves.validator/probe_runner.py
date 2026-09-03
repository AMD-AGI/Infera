#!/usr/bin/env python3
"""Execute a resolved probe plan against a live deployment and write the results.

**RUNS ON THE COMPUTE NODE. Standard library only.** That constraint is the
reason this file exists as a separate program rather than as functions inside
`check.py`: the endpoint under test is bound on the node, the validator body runs
on the login node, and the node's `python3` is whatever the image happens to
carry — so nothing here may import `yaml`, `requests` or anything else. The
plan arrives as JSON, already resolved by `check.py` from `probes.yaml`, and the
results leave as JSON.

    probe_runner.py --plan <plan.json> --out <results.json>

The plan is a list of probe objects; the expectation vocabulary is the one
`probes.yaml` documents. This file adds no probe and no criterion of its own —
if a probe is not in the yaml it is not sent, which is what mission M1.2.3.3's
*不允许临时发挥* reduces to once it has to be enforced rather than intended.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

#: `choices[].message.content` — a path with a list hop. The hop means "every
#: element", so `equals` on it is a claim about all of them; an empty list fails,
#: because "all of nothing" passing is how a probe reports success against a
#: server that returned no choices at all.
_HOP = re.compile(r"([^.\[\]]+)(\[\])?")


def walk(document, path: str) -> tuple[list, bool]:
    """Every value at `path`. Returns `(values, reached)`.

    `reached` is False when a key along the way is simply absent, which is a
    different fact from "present and empty" and is reported differently.
    """
    cursor = [document]
    for name, listy in _HOP.findall(path):
        nxt = []
        for item in cursor:
            if not isinstance(item, dict) or name not in item:
                return [], False
            value = item[name]
            if listy:
                if not isinstance(value, list):
                    return [], False
                nxt.extend(value)
            else:
                nxt.append(value)
        cursor = nxt
    return cursor, True


def send(probe: dict) -> dict:
    """One HTTP request. Never raises: a transport failure is a result."""
    request = probe["request"]
    timeout = request.get("timeout", 60)
    body = None
    headers = {}
    if "json" in request:
        body = json.dumps(request["json"]).encode()
        headers["Content-Type"] = "application/json"
    if request.get("stream"):
        headers["Accept"] = "text/event-stream"

    req = urllib.request.Request(
        request["url"], data=body, headers=headers, method=request["method"]
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.getcode()
    except urllib.error.HTTPError as exc:
        # A 4xx is an *answer*, and several probes are satisfied by one. Reading
        # the body matters: SGLang puts the reason there.
        raw = exc.read().decode("utf-8", "replace")
        status = exc.code
    except Exception as exc:  # URLError, timeout, connection reset, socket error
        return {
            "status": None,
            "body": "",
            "elapsed_s": round(time.time() - started, 3),
            "transport_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": status,
        "body": raw,
        "elapsed_s": round(time.time() - started, 3),
        "transport_error": None,
    }


def sse_chunks(body: str) -> list[str]:
    """The `data:` payloads of a server-sent-event stream, `[DONE]` included."""
    return [
        line[len("data:"):].strip()
        for line in body.splitlines()
        if line.startswith("data:")
    ]


def judge(probe: dict, response: dict) -> list[str]:
    """Every way this response fails this probe's expectations."""
    expect = probe.get("expect") or {}
    faults: list[str] = []

    if response["transport_error"]:
        return [f"no answer: {response['transport_error']}"]

    # `any_of` is satisfied by one of several alternative shapes; used where
    # upstream is still changing which one it emits.
    if "any_of" in expect:
        for option in expect["any_of"]:
            if not judge({"expect": option}, response):
                return []
        return [
            "none of the accepted refusal shapes: "
            + json.dumps(expect["any_of"])
            + f" (got status {response['status']})"
        ]

    if "status" in expect and response["status"] != expect["status"]:
        faults.append(f"status {response['status']}, expected {expect['status']}")
    if "status_in" in expect and response["status"] not in expect["status_in"]:
        faults.append(f"status {response['status']}, expected one of {expect['status_in']}")

    pattern = expect.get("body_matches")
    if pattern and not re.search(pattern, response["body"]):
        faults.append(f"body does not match {pattern!r}")

    needs_json = any(
        key in expect for key in ("json_nonempty", "json_all", "json_present")
    )
    document = None
    if needs_json:
        try:
            document = json.loads(response["body"])
        except ValueError as exc:
            return faults + [f"body is not JSON: {exc}"]

    name = expect.get("json_nonempty")
    if name:
        values, reached = walk(document, name)
        if not reached:
            faults.append(f"no {name} in the response")
        elif not values or not values[0]:
            faults.append(f"{name} is empty")

    for path in expect.get("json_present") or []:
        _, reached = walk(document, path)
        if not reached:
            faults.append(f"{path} is absent")

    for rule in expect.get("json_all") or []:
        path = rule["path"]
        values, reached = walk(document, path)
        if not reached:
            faults.append(f"{path} is absent")
            continue
        if not values:
            faults.append(f"{path} matched nothing — the list it walks is empty")
            continue
        for value in values:
            if "equals" in rule and value != rule["equals"]:
                faults.append(f"{path} is {value!r}, expected {rule['equals']!r}")
            if "nonempty" in rule and not value:
                faults.append(f"{path} is empty")
            if "not_matches" in rule and re.search(rule["not_matches"], str(value)):
                faults.append(f"{path} is {value!r}, which matches the forbidden {rule['not_matches']!r}")

    # ---- streaming ---------------------------------------------------------
    # The researched trap (sgl-project/sglang#19996): SGLang returns HTTP 200
    # carrying an error payload for a streaming request that vLLM refuses with a
    # 400. A judgement made on the status line alone reports that as a success,
    # so the stream is judged on its body.
    if any(key.startswith("sse_") for key in expect):
        chunks = sse_chunks(response["body"])
        payload = [c for c in chunks if c and c != "[DONE]"]
        errors = []
        for chunk in payload:
            try:
                parsed = json.loads(chunk)
            except ValueError:
                continue
            if isinstance(parsed, dict) and parsed.get("error"):
                errors.append(parsed["error"])

        floor = expect.get("sse_min_chunks")
        if floor is not None and len(payload) < floor:
            faults.append(f"{len(payload)} data chunk(s), expected at least {floor}")
        if expect.get("sse_terminates") and "[DONE]" not in chunks:
            faults.append("the stream never sent `data: [DONE]`; it was cut, not finished")
        if expect.get("sse_no_error") and errors:
            faults.append(
                f"the stream carried an error payload under HTTP {response['status']}: "
                f"{json.dumps(errors[0])[:300]}"
            )
        if expect.get("sse_carries_error") and not errors:
            faults.append("the stream carried no error payload")

    return faults


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(open(args.plan).read())
    results = []
    short_circuit = False

    for probe in plan["probes"]:
        if short_circuit:
            results.append({
                "name": probe["name"],
                "severity": probe["severity"],
                "skipped": "an earlier fatal probe short-circuited the set",
                "faults": [],
                "passed": None,
            })
            continue

        response = send(probe)
        faults = judge(probe, response)
        results.append({
            "name": probe["name"],
            "severity": probe["severity"],
            "direction": probe.get("direction", "").strip(),
            "url": probe["request"]["url"],
            "status": response["status"],
            "elapsed_s": response["elapsed_s"],
            # Bounded: a /metrics body is megabytes and a probe result is read by
            # a person. The judgement was made on the whole body.
            "body_head": response["body"][:2000],
            "faults": faults,
            "passed": not faults,
        })
        if faults and probe.get("short_circuit"):
            short_circuit = True

    payload = {
        "probes": results,
        "failed": [r["name"] for r in results if r["passed"] is False and r["severity"] == "fail"],
        "warned": [r["name"] for r in results if r["passed"] is False and r["severity"] == "warn"],
        "skipped": [r["name"] for r in results if r["passed"] is None],
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    for row in results:
        mark = "SKIP" if row["passed"] is None else ("ok" if row["passed"] else row["severity"].upper())
        print(f"  {mark:5} {row['name']}")
        for fault in row["faults"]:
            print(f"        {fault}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
