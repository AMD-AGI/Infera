#!/usr/bin/env python3
"""Plot GLM-5.2 AgentX P90 interactivity against token throughput per chip.

Combines the InferenceX CSV export with local AgentX sweeps
(<results>/c*/agentx_conc*.json), using the InferenceX definitions:

* P90 interactivity = 1 / P90 full-response ITL in seconds (plain ITL when a
  result has no full-response block).
* Token throughput per chip = input (including cached) + output tokens/s,
  divided by all prefill and decode chips of the deployment.

Only the Pareto-optimal points of each series are drawn (--all-points adds the
dominated ones as hollow markers). Writes a PNG, an SVG and a CSV of all points
with their frontier flag.

    python3 plot_p90_interactivity.py
    python3 plot_p90_interactivity.py --run "LABEL=/path/to/results" --run ...
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import StrMethodFormatter
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INFERENCEX_CSV = ROOT / "refs_performance/20260922/InferenceX_GLM-5.2_interactivity.csv"
DEFAULT_RUNS = (
    ("MI355X SGLang 2P1D, TileLang DSA (09-21)", ROOT / "2p1d-sweep/results"),
    ("MI355X SGLang 2P1D, Triton DSA (09-22, new)", ROOT / "2p1d-sweep-triton-dsa-20260922/results"),
)
FRAMEWORKS = {"sglang": "SGLang", "vllm": "vLLM", "atom": "ATOM", "trt": "TRT-LLM"}
INFERENCEX_COLORS = {"b300_sglang": "#2ca02c", "b200_sglang": "#8c564b",
                     "mi355x_atom": "#d62728", "mi355x_sglang": "#9467bd"}
EXTRA_COLORS = ("#bcbd22", "#7f7f7f", "#e377c2", "#17becf")
RUN_COLORS = ("#1f77b4", "#ff7f0e", "#17becf", "#e377c2", "#7f7f7f")
TP_MARKERS = {8: "o", 4: "s", 2: "^", 1: "v"}
FIELDS = ("Series", "Source", "Hardware", "Framework", "Precision", "SpecDecoding", "Topology",
          "Chips", "Concurrency", "P90Interactivity", "TokenThroughputPerChip",
          "InputThroughputPerChip", "OutputThroughputPerChip", "P90ITLSeconds",
          "MedianInteractivity", "OnParetoFrontier", "Reference", "Provenance")
ATTRIBUTION = ("InferenceX data: https://github.com/SemiAnalysisAI/InferenceX, "
               "Copyright 2026 SemiAnalysis LLC, Apache License 2.0")
DIRECTIONS = ((0, 1, "center", "bottom"), (0, -1, "center", "top"),
              (1, 0, "left", "center"), (-1, 0, "right", "center"),
              (0.7, 0.7, "left", "bottom"), (-0.7, 0.7, "right", "bottom"),
              (0.7, -0.7, "left", "top"), (-0.7, -0.7, "right", "top"))


def load_inferencex(path):
    with path.open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    colors, extra = {}, iter(EXTRA_COLORS)
    points = []
    for row in rows:
        key, tp = row["Hardware Key"], int(row["TP"])
        chips = int(row["Physical Chips"] or tp)
        interactivity = float(row["P90 Interactivity (tok/s/user)"])
        framework = FRAMEWORKS.get(row["Framework"], row["Framework"])
        disagg = " disagg" if row["Disaggregated"] == "true" else ""
        if key not in colors:
            colors[key] = INFERENCEX_COLORS.get(key) or next(extra)
        points.append({
            "Series": f'{row["Hardware"].upper()} {framework}{disagg} (InferenceX)',
            "Source": "InferenceX", "Hardware": row["Hardware"], "Framework": row["Framework"],
            "Precision": row["Precision"], "SpecDecoding": row["Spec Decoding"],
            "Topology": f"TP{tp}", "Chips": chips, "Concurrency": int(row["Concurrency"]),
            "P90Interactivity": interactivity,
            "TokenThroughputPerChip": float(row["Token Throughput per Chip (tok/s/chip)"]),
            "InputThroughputPerChip": float(row["Input Throughput/Chip (tok/s)"]),
            "OutputThroughputPerChip": float(row["Output Throughput/Chip (tok/s)"]),
            "P90ITLSeconds": round(1 / interactivity, 6),
            "MedianInteractivity": float(row["Median Interactivity (tok/s/user)"]),
            "Reference": row["Date"], "Provenance": row["Run URL"],
            "color": colors[key], "marker": TP_MARKERS.get(tp, "^"),
            "shape": f"single node TP{tp} ({chips} chips)",
        })
    return points


def load_run(label, directory):
    """Return the sweep points and its (duration, simulated acceptance) from run.json."""
    run_id, condition = directory.parent.name, None
    manifest = directory / "run.json"
    if manifest.exists():
        run = json.loads(manifest.read_text())
        if run["status"] != "complete":
            raise ValueError(f"sweep is not complete: {directory}")
        run_id = run["run_id"]
        condition = (run.get("requested_duration_s"), run.get("simulation_accept_length"))
    points = []
    for path in sorted(directory.glob("c*/agentx_conc*.json")):
        result = json.loads(path.read_text())
        latency = result["request_metrics"]["latency"]
        throughput = result["request_metrics"]["throughput"]
        itl = latency.get("full_response_itl") or latency["itl"]
        prefill, decode = int(result["num_prefill_gpu"]), int(result["num_decode_gpu"])
        chips = prefill + decode
        total = throughput["total"]["tokens_per_second"] / chips
        claimed = throughput.get("per_gpu", {}).get("total_tput_tps")
        if claimed is not None and not math.isclose(total, claimed, rel_tol=1e-5):
            raise ValueError(f"{path}: per-chip throughput is not normalized by {chips} chips")
        topology = f'{result["prefill_num_workers"]}P{result["decode_num_workers"]}D'
        points.append({
            "Series": label, "Source": "local", "Hardware": result["hw"],
            "Framework": result["framework"], "Precision": result["precision"],
            "SpecDecoding": result["spec_decoding"], "Topology": topology, "Chips": chips,
            "Concurrency": int(result["conc"]), "P90Interactivity": 1 / itl["p90"],
            "TokenThroughputPerChip": total,
            "InputThroughputPerChip": throughput["input"]["tokens_per_second"] / chips,
            "OutputThroughputPerChip": throughput["output"]["tokens_per_second"] / chips,
            "P90ITLSeconds": itl["p90"], "MedianInteractivity": 1 / itl["p50"],
            "Reference": run_id, "Provenance": str(path.resolve()), "marker": "D",
            "shape": f"{topology} ({chips} chips: {prefill} prefill + {decode} decode)",
        })
    if not points:
        raise ValueError(f"no agentx_conc*.json under {directory}")
    return sorted(points, key=lambda point: point["Concurrency"]), condition


def mark_pareto(points):
    best = -math.inf
    for point in sorted(points, key=lambda p: (-p["P90Interactivity"], -p["TokenThroughputPerChip"])):
        point["OnParetoFrontier"] = point["TokenThroughputPerChip"] > best
        best = max(best, point["TokenThroughputPerChip"])


def is_local(point):
    return point["Source"] == "local"


def marker_size(point):
    return 8 if is_local(point) else 6.5


def draw_series(ax, series):
    for members in series.values():
        local, color = is_local(members[0]), members[0]["color"]
        frontier = sorted((p for p in members if p["OnParetoFrontier"]),
                          key=lambda p: p["P90Interactivity"])
        ax.plot([p["P90Interactivity"] for p in frontier],
                [p["TokenThroughputPerChip"] for p in frontier],
                color=color, linewidth=2.6 if local else 1.6, zorder=4 if local else 2)
        for p in members:
            ax.plot(p["P90Interactivity"], p["TokenThroughputPerChip"], linestyle="none",
                    marker=p["marker"], markersize=marker_size(p), markeredgewidth=1.5,
                    markeredgecolor=color,
                    markerfacecolor=color if p["OnParetoFrontier"] else "white",
                    zorder=5 if local else 3)


def in_view(ax, point):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return x0 <= point["P90Interactivity"] <= x1 and y0 <= point["TokenThroughputPerChip"] <= y1


def marker_boxes(ax, points):
    boxes = []
    for p in points:
        if in_view(ax, p):
            x, y = ax.transData.transform((p["P90Interactivity"], p["TokenThroughputPerChip"]))
            r = (marker_size(p) / 2 + 1) * ax.figure.dpi / 72
            boxes.append(Bbox.from_extents(x - r, y - r, x + r, y + r))
    return boxes


def label_points(ax, points, obstacles):
    """Greedily place concurrency labels next to markers without overlaps.

    Local points always get a label (with a leader line when moved further out);
    InferenceX labels are skipped when no free position exists.
    """
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    taken = list(obstacles)
    for p in points:
        local = is_local(p)
        first = DIRECTIONS[1] if local and p["run_index"] % 2 == 0 else DIRECTIONS[0]
        order = [first] + [d for d in DIRECTIONS if d != first]
        style = {"fontsize": 10 if local else 8.5, "fontweight": "bold" if local else "normal",
                 "color": p["color"], "zorder": 6, "textcoords": "offset points",
                 "bbox": {"boxstyle": "square,pad=0.1", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.75}}
        xy, text = (p["P90Interactivity"], p["TokenThroughputPerChip"]), f'c{p["Concurrency"]}'
        chosen = None
        for distance in (7, 17):
            for dx, dy, ha, va in order:
                probe = ax.annotate(text, xy, xytext=(dx * distance, dy * distance),
                                    ha=ha, va=va, **style)
                box = probe.get_window_extent(renderer).padded(2)
                probe.remove()
                if (frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0
                        and box.y1 <= frame.y1 and not any(box.overlaps(o) for o in taken)):
                    chosen = (distance, dx, dy, ha, va, box)
                    break
            if chosen:
                break
        if chosen is None:
            if not local:
                continue
            chosen = (17, *order[0], None)
        distance, dx, dy, ha, va, box = chosen
        arrow = {"arrowstyle": "-", "color": p["color"], "linewidth": 0.7} if distance > 7 else None
        ax.annotate(text, xy, xytext=(dx * distance, dy * distance), ha=ha, va=va,
                    arrowprops=arrow, **style)
        if box is not None:
            taken.append(box)


def parse_run(text):
    label, separator, directory = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected LABEL=RESULTS_DIR")
    return label, Path(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inferencex", type=Path, default=INFERENCEX_CSV,
                        help="InferenceX interactivity CSV export")
    parser.add_argument("--run", dest="runs", type=parse_run, action="append",
                        metavar="LABEL=RESULTS_DIR",
                        help="local sweep to overlay, repeatable "
                             "(default: the TileLang and Triton DSA 2P1D sweeps)")
    parser.add_argument("--zoom", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                        help="right-panel limits (default: around the local sweeps)")
    parser.add_argument("--all-points", action="store_true",
                        help="also draw points dominated within their series (hollow markers)")
    parser.add_argument("--output", type=Path,
                        default=HERE / "glm52_p90_interactivity_vs_throughput.png",
                        help="PNG path; the SVG and *_points.csv are written next to it")
    args = parser.parse_args()

    points = load_inferencex(args.inferencex)
    inferencex = list(points)
    conditions = set()
    for index, (label, directory) in enumerate(args.runs or DEFAULT_RUNS):
        run_points, condition = load_run(label, directory)
        for p in run_points:
            p["color"], p["run_index"] = RUN_COLORS[index % len(RUN_COLORS)], index
        points += run_points
        conditions.add(condition)
    series = {}
    for p in points:
        series.setdefault(p["Series"], []).append(p)
    for members in series.values():
        mark_pareto(members)
    plotted = {}
    for p in points:
        if args.all_points or p["OnParetoFrontier"]:
            plotted.setdefault(p["Series"], []).append(p)
    shown = [p for members in plotted.values() for p in members]

    if args.zoom:
        x0, x1, y0, y1 = args.zoom
    else:
        xs = [p["P90Interactivity"] for p in shown if is_local(p)]
        x0 = max(0, 5 * math.floor((min(xs) - 12) / 5))
        x1 = 5 * math.ceil((max(xs) + 20) / 5)
        y0 = 0
        y1 = 1000 * math.ceil(1.08 * max(p["TokenThroughputPerChip"] for p in shown
                                         if x0 <= p["P90Interactivity"] <= x1) / 1000)
    xmax = 20 * math.ceil(1.05 * max(p["P90Interactivity"] for p in shown) / 20)
    ymax = 1000 * math.ceil(1.08 * max(p["TokenThroughputPerChip"] for p in shown) / 1000)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5})
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(21, 9), layout="constrained",
                                     width_ratios=(1, 1.35))
    for ax in (full, zoom):
        draw_series(ax, plotted)
        ax.set(xlabel="P90 interactivity (tok/s/user)",
               ylabel="Token throughput per chip (tok/s/chip)")
        ax.grid(alpha=0.25)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
    full.set(xlim=(0, xmax), ylim=(0, ymax), title="Full range")
    zoom.set(xlim=(x0, x1), ylim=(y0, y1),
             title=f"Zoom around the local sweeps: {x0:g}–{x1:g} tok/s/user")
    full.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linestyle="--",
                             edgecolor="0.45", linewidth=1, zorder=1))

    handles = [Line2D([], [], color=m[0]["color"], linewidth=2.6 if is_local(m[0]) else 1.6,
                      label=name) for name, m in plotted.items()]
    shapes = {}
    for p in shown:
        shapes.setdefault(p["shape"], p["marker"])
    handles += [Line2D([], [], linestyle="none", marker=marker, markersize=7, color="0.35",
                       label=shape) for shape, marker in shapes.items()]
    if args.all_points:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=7,
                              markerfacecolor="white", markeredgecolor="0.35",
                              markeredgewidth=1.5, label="hollow: dominated within its series"))
    legend = full.legend(handles=handles, loc="upper right", fontsize=9.5, framealpha=0.95)

    dates = ", ".join(sorted({p["Reference"] for p in inferencex}))
    configs = ", ".join(sorted({f'{p["Precision"].upper()} {p["SpecDecoding"].upper()}'
                                for p in inferencex}))
    notes = [
        "P90 interactivity = 1 / P90 full-response inter-token latency (InferenceX definition). "
        "Token throughput per chip = (input incl. cached + output) tok/s divided by all chips "
        "of the deployment.",
        ("Lines: Pareto frontier of each series; hollow markers: points dominated within their "
         "series" if args.all_points else
         "Pareto-optimal points only: points beaten on both axes by another point of the same "
         "series are omitted") + "; labels: AgentX concurrency.",
        f"InferenceX rows: single node, {configs}, dated {dates}.",
    ]
    if len(conditions) == 1 and None not in conditions:
        duration, accept = next(iter(conditions))
        notes[-1] += (f" Local sweeps: {float(duration):,.0f} s per point, "
                      f"simulated MTP acceptance length {accept}.")
    notes.append(ATTRIBUTION + ".")
    fig.suptitle("GLM-5.2 AgentX (agentic coding): P90 interactivity vs token throughput per chip",
                 fontsize=16, fontweight="bold")
    fig.supxlabel("\n".join(notes), fontsize=9.5)

    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    priority = sorted(shown, key=lambda p: (not is_local(p), -p.get("run_index", 0)))
    inside = [p for p in priority if in_view(zoom, p)]
    label_points(zoom, inside, marker_boxes(zoom, shown))
    outside = [p for p in priority if in_view(full, p) and not in_view(zoom, p)]
    label_points(full, outside, marker_boxes(full, shown) + [legend.get_window_extent(renderer)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        path = args.output.with_suffix(suffix)
        fig.savefig(path, dpi=200)
        print(path)
    plt.close(fig)
    table = args.output.with_name(args.output.stem + "_points.csv")
    with table.open("w", newline="") as stream:
        stream.write(f"# {ATTRIBUTION}; rows with Source=InferenceX are derived from it.\n")
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)
    print(table)
    for p in filter(is_local, points):
        print(f'{p["Series"]:46s} c{p["Concurrency"]:<4d} P90 {p["P90Interactivity"]:6.2f} tok/s/user  '
              f'{p["TokenThroughputPerChip"]:9,.0f} tok/s/chip  '
              f'{"frontier" if p["OnParetoFrontier"] else "dominated"}')


if __name__ == "__main__":
    main()
