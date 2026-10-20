#!/usr/bin/env python3
"""Progress figure (pure-Python SVG): throughput per round, per-request time breakdown, a1-spf utilisation.
Numbers come from RESULT.zh-CN.md §2.5-2.9 (client JSON + sampler windows + prefill_rank_analysis.txt).
Usage: make_progress_svg.py OUT.svg
"""
import re
import sys

W, H = 1200, 650
out = []
add = out.append


CJK = re.compile(r"([\u2e80-\u9fff\u3000-\u303f\uff00-\uffef\u2460-\u24ff\u2190-\u21ff\u00b7]+)")


def text(x, y, s, size=14, anchor="start", weight="normal", color="#222"):
    # One tspan per script run: CJK in Droid Sans Fallback, the rest in DejaVu Sans (renderers here do not fall back).
    spans = []
    for part in CJK.split(s):
        if not part:
            continue
        fam = "DejaVu Sans"
        part = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        spans.append(f'<tspan font-family="{fam}">{part}</tspan>')
    add(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{color}">'
        + "".join(spans) + "</text>")


def rect(x, y, w, h, color, opacity=1.0):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}" fill-opacity="{opacity}"/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
rect(0, 0, W, H, "#ffffff")
text(W / 2, 32, "GLM-5.2 2P(TP4)1D(TP8) on AUS | AgentX C176 | 10-20 05:01", 20, "middle", "bold")

# Panel 1: throughput per round.
text(40, 72, "(1) Throughput, tok/s/GPU (AgentX C176)", 16, weight="bold")
bars = [("3P1D best C192\n(crsuse2, 20 GPUs)", 31829, "#9e9e9e"), ("t5b 2P1D\n(crsuse2)", 28900, "#bdbdbd"),
        ("a1-wt2 baseline\nwrite_through", 25695, "#90caf9"), ("a1-wb\n+write_back", 30659, "#42a5f5"),
        ("a1-spf\n+SPF scheduling", 35377, "#1565c0")]
x0, y0, bh, ymax = 60, 330, 220, 40000
for i, (lab, v, c) in enumerate(bars):
    x = x0 + i * 105
    h = v / ymax * bh
    rect(x, y0 - h, 70, h, c)
    text(x + 35, y0 - h - 6, f"{v:,}", 13, "middle", "bold")
    for k, part in enumerate(lab.split("\n")):
        text(x + 35, y0 + 18 + k * 15, part, 11, "middle")
text(x0 + 3 * 105 + 35, y0 - 30659 / ymax * bh - 24, "+19%", 12, "middle", color="#c62828")
text(x0 + 4 * 105 + 35, y0 - 35377 / ymax * bh - 24, "+15%", 12, "middle", color="#c62828")

# Panel 2: per-request cycle breakdown.
text(620, 72, "(2) One lane cycle = idle + TTFT + decode (seconds)", 16, weight="bold")
segs = [("idle (trace gap)", "#e0e0e0"), ("prefill queue", "#ef5350"), ("bootstrap / alloc wait", "#ffb74d"),
        ("prefill compute", "#ffd54f"), ("KV transfer", "#aed581"), ("decode", "#64b5f6")]
rows = [("a1-wt2", [7.3, 16.5, 2.0, 4.8, 1.7, 18.6], 50.9), ("a1-wb", [8.6, 10.9, 0.5, 4.5, 2.1, 18.0], 44.6),
        ("a1-spf", [11.0, 3.3, 0.2, 5.0, 2.3, 18.7], 40.5)]
sx, sy, scale = 680, 110, 8.2
for r, (name, vals, cyc) in enumerate(rows):
    y = sy + r * 60
    text(sx - 10, y + 22, name, 13, "end", "bold")
    x = sx
    for (lab, c), v in zip(segs, vals):
        rect(x, y, v * scale, 34, c)
        if v >= 2.5:
            text(x + v * scale / 2, y + 22, f"{v:.1f}", 11, "middle")
        x += v * scale
    text(x + 8, y + 22, f"= {cyc} s", 12)
for k, (lab, c) in enumerate(segs):
    lx, ly = 690 + (k % 3) * 160, 300 + (k // 3) * 22
    rect(lx, ly - 11, 14, 14, c)
    text(lx + 20, ly, lab, 12)
text(690, 352, "176 lanes / cycle: 50.9 -> 44.6 -> 40.5 s = 3.46 -> 3.94 -> 4.35 req/s", 12, color="#555")

# Panel 3: a1-spf utilisation.
text(40, 410, "(3) a1-spf: how loaded is each side?", 16, weight="bold")
text(60, 440, "prefill: busy share per DP rank (mean ~68%; prefill-1 busier than prefill-0)", 13)
busy = [("P0.DP0", 65), ("P0.DP1", 60), ("P0.DP2", 59), ("P0.DP3", 57), ("P1.DP0", 76), ("P1.DP1", 77),
        ("P1.DP2", 75), ("P1.DP3", 79)]
for i, (lab, v) in enumerate(busy):
    x = 60 + i * 66
    rect(x, 600 - v * 1.4, 44, v * 1.4, "#ffb74d" if lab.startswith("P0") else "#fb8c00")
    text(x + 22, 600 - v * 1.4 - 5, f"{v}%", 11, "middle")
    text(x + 22, 616, lab, 10, "middle")
add('<line x1="55" y1="460" x2="590" y2="460" stroke="#c62828" stroke-dasharray="4,3"/>')
text(592, 464, "100%", 10, color="#c62828")

text(640, 440, "decode: distance to its limits (per DP rank)", 13)
gauges = [("running", 9.79, 11.0, "reqs (about 10-11 at ITL 20 ms)"), ("output tok/s", 530, 680, "(plateau ~650-720 in 3P1D)"),
          ("KV usage", 0.74, 1.0, "(>90% for 27.5% of the time)"), ("ITL p50", 20.0, 20.0, "ms (P50 interactivity 50.0: at the 50 line)")]
for i, (lab, v, cap, note) in enumerate(gauges):
    y = 470 + i * 44
    text(640, y + 16, lab, 12, weight="bold")
    rect(740, y, 300, 22, "#eeeeee")
    rect(740, y, 300 * min(v / cap, 1.0), 22, "#1e88e5" if v / cap < 0.95 else "#e53935")
    text(1046, y + 16, f"{v:g}", 12, weight="bold")
    text(740, y + 36, note, 10, color="#555")

add("</svg>")
open(sys.argv[1], "w").write("\n".join(out))
print("wrote", sys.argv[1])
