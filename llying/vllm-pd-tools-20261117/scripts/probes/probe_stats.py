#!/usr/bin/env python3
"""Counts of one GSM8K probe run (gsm8k_probe.py output), errors kept apart.

bad_start: reasoning corrupted right after the first token. probe_badstart.py's rule ("The" + a non-space
character, CJK in the first 60 characters, or under 25 characters) also caught English words ("They",
"There") and the empty text of failed requests; here failed requests are excluded and those words are
not counted. garbled: the probe's own flag (repetition / think tags). Every flagged sample is written to
REVIEW for reading by hand.

usage: probe_stats.py PROBE.jsonl [--review REVIEW.txt]
"""
import argparse
import json
import re

CJK = re.compile(r"[\u4e00-\u9fff]")
WORDS = {"They", "There", "Then", "These", "Their", "Theirs", "Them", "Theme", "Themes", "Theory",
         "Therefore", "Thereby", "Thereafter", "Theater", "Theatre", "Theo", "Thea", "Theodore"}


def bad_start(text):
    s = text or ""
    m = re.match(r"The\S*", s)
    if m and m.group(0) != "The" and m.group(0).rstrip(".,:;!?") not in WORDS:
        return True
    return bool(CJK.search(s[:60]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--review")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.probe)]
    ok = [r for r in rows if not r.get("error")]
    bad = [r for r in ok if bad_start(r.get("reasoning"))]
    gar = [r for r in ok if r.get("garbled")]
    cap = [r for r in ok if r.get("finish_reason") == "length"]
    short = [r for r in ok if len((r.get("reasoning") or "").strip()) < 25 and not r.get("content")]
    print(f"n={len(rows)} errors={len(rows) - len(ok)} correct={sum(r['correct'] for r in ok)}/{len(ok)} "
          f"({sum(r['correct'] for r in ok) / max(1, len(ok)):.3f}) bad_start={len(bad)} garbled={len(gar)} "
          f"hit_cap={len(cap)} empty={len(short)}")
    if a.review:
        with open(a.review, "w") as f:
            for tag, group in (("BAD_START", bad), ("GARBLED", gar), ("EMPTY", short)):
                for r in group:
                    s = r.get("reasoning") or ""
                    f.write(f"{tag} doc={r.get('doc_id')} fin={r.get('finish_reason')} tok={r.get('completion_tokens')} "
                            f"correct={r.get('correct')}\n  head: {s[:200]!r}\n  tail: {s[-160:]!r}\n")


if __name__ == "__main__":
    main()
