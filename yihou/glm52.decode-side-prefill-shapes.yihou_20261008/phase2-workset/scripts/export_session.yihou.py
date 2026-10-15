#!/usr/bin/env python3
"""Render a Claude Code session .jsonl into readable markdown, streaming (constant memory).

Usage: export_session.yihou.py SESSION.jsonl OUT.md [--max-tool-chars 1500]
Keeps user messages, assistant text, tool calls (input) and tool results (truncated).
Thinking blocks are omitted.
"""
import argparse
import json


def text_of(content, limit):
    if isinstance(content, str):
        return content[:limit] + (" …[truncated]" if len(content) > limit else "")
    parts = []
    for c in content or []:
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(c.get("text", ""))
    s = "\n".join(parts)
    return s[:limit] + (" …[truncated]" if len(s) > limit else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--max-tool-chars", type=int, default=1500)
    a = ap.parse_args()
    lim = a.max_tool_chars
    with open(a.session) as src, open(a.out, "w") as out:
        out.write(f"# Session export — {a.session}\n\n")
        for line in src:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                if rec.get("type") == "summary":
                    out.write(f"\n> **[summary]** {rec.get('summary', '')}\n")
                continue
            role, ts = msg.get("role"), rec.get("timestamp", "")
            content = msg.get("content")
            if isinstance(content, str):
                out.write(f"\n## {role} · {ts}\n\n{content}\n")
                continue
            for c in content or []:
                t = c.get("type")
                if t == "text" and c.get("text", "").strip():
                    out.write(f"\n## {role} · {ts}\n\n{c['text']}\n")
                elif t == "tool_use":
                    inp = json.dumps(c.get("input", {}), ensure_ascii=False)
                    out.write(f"\n### tool_use `{c.get('name')}` · {ts}\n\n```\n{inp[:lim]}"
                              f"{' …[truncated]' if len(inp) > lim else ''}\n```\n")
                elif t == "tool_result":
                    out.write(f"\n### tool_result · {ts}\n\n```\n{text_of(c.get('content'), lim)}\n```\n")


if __name__ == "__main__":
    main()
