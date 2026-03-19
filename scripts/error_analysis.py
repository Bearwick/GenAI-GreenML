#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


ITERATION_PREFIX = "failed_generated_code_iteration_"
ANALYSIS_FILENAME = "analysis"

TOTAL_ERRORS_LINE_RE = re.compile(r"^Total Errors occured:\s*(\d+)$")
MODE_ERRORS_LINE_RE = re.compile(r"^Total Errors by (assisted|original|autonomous):\s*(\d+)$")
LLM_TOTAL_LINE_RE = re.compile(r"^Total Errors per LLM:\s*(.*)$")
LLM_BY_MODE_LINE_RE = re.compile(r"^Total Errors per LLM by (assisted|original|autonomous):\s*(.*)$")
TYPE_BY_MODE_LINE_RE = re.compile(r"^Type of Error by (assisted|original|autonomous):\s*(.*)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build cross-iteration error analysis visualizations from failed_generated_code_iteration_*"
            " analysis files."
        )
    )
    p.add_argument(
        "--root-dir",
        default=None,
        help="Repository root containing failed_generated_code_iteration_* (default: project root).",
    )
    p.add_argument(
        "--output-dir",
        default="error_analysis",
        help="Output directory for plots (default: ./error_analysis).",
    )
    p.add_argument(
        "--max-error-types",
        type=int,
        default=0,
        help="Max number of error types in the temporal plot (0 = include all; default: 0).",
    )
    return p.parse_args()


def import_plot_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError as e:
        pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
        raise SystemExit(
            f"Missing dependency: {pkg}. Install with: python3 -m pip install matplotlib"
        )


def normalize_error_type(name: str) -> str:
    name = name.strip()
    if "." in name:
        return name.rsplit(".", 1)[-1]
    return name


def parse_counter(text: str) -> Dict[str, int]:
    text = (text or "").strip()
    if not text or text.lower() == "none":
        return {}

    out: Dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = normalize_error_type(key)
        try:
            out[key] = int(val.strip())
        except ValueError:
            out[key] = 0
    return out


def parse_iteration_index(name: str) -> int:
    suffix = name[len(ITERATION_PREFIX) :]
    try:
        return int(suffix)
    except ValueError:
        return 10**9


def parse_analysis_file(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    data = {
        "name": path.parent.name,
        "iteration": path.parent.name,
        "iteration_index": parse_iteration_index(path.parent.name),
        "total_errors": 0,
        "mode_totals": {"assisted": 0, "original": 0, "autonomous": 0},
        "llm_total": Counter(),
        "llm_by_mode": {
            "assisted": Counter(),
            "original": Counter(),
            "autonomous": Counter(),
        },
        "error_types_by_mode": {
            "assisted": Counter(),
            "original": Counter(),
            "autonomous": Counter(),
        },
        "error_types": Counter(),
    }

    in_error_type_list = False
    for line in text:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Error type with list of projects:script"):
            in_error_type_list = True
            continue
        if in_error_type_list and line.startswith("Error type with exact causes"):
            break
        if in_error_type_list and ":" in line and not line.startswith("none"):
            key, entries = line.split(":", 1)
            key = normalize_error_type(key)
            if key:
                count = 0
                entries = entries.strip()
                if entries and entries != "none":
                    count = len([v for v in entries.split(",") if v.strip()])
                data["error_types"][key] += count
            continue

        m = MODE_ERRORS_LINE_RE.match(line)
        if m:
            mode = m.group(1)
            data["mode_totals"][mode] = int(m.group(2))
            continue

        m = LLM_TOTAL_LINE_RE.match(line)
        if m:
            data["llm_total"].update(parse_counter(m.group(1)))
            continue

        m = LLM_BY_MODE_LINE_RE.match(line)
        if m:
            mode = m.group(1)
            data["llm_by_mode"][mode].update(parse_counter(m.group(2)))
            continue

        m = TYPE_BY_MODE_LINE_RE.match(line)
        if m:
            mode = m.group(1)
            data["error_types_by_mode"][mode].update(parse_counter(m.group(2)))
            continue

        m = TOTAL_ERRORS_LINE_RE.match(line)
        if m:
            data["total_errors"] = int(m.group(1))

    return data


def iteration_labels(iterations: List[Dict[str, object]]) -> Tuple[List[int], List[str]]:
    numbers: List[int] = []
    labels: List[str] = []
    for it in iterations:
        idx = int(it["iteration_index"]) if isinstance(it["iteration_index"], int) else 0
        if idx >= 10**8:
            num = 0
            label = str(it["iteration"])
        else:
            num = idx
            label = str(idx)
        numbers.append(num)
        labels.append(label)
    return numbers, labels


def should_include_original(iterations: List[Dict[str, object]]) -> bool:
    return any(int(it["mode_totals"]["original"]) > 0 for it in iterations)


def find_iterations(root_dir: Path) -> List[Path]:
    dirs = [
        p
        for p in root_dir.iterdir()
        if p.is_dir() and p.name.startswith(ITERATION_PREFIX)
    ]
    if not dirs:
        raise SystemExit(f"No iteration folders found in {root_dir}")

    def sort_key(p: Path) -> Tuple[int, str]:
        suffix = p.name[len(ITERATION_PREFIX) :]
        try:
            return (int(suffix), suffix)
        except ValueError:
            return (10**9, suffix)

    return sorted(dirs, key=sort_key)


def plot_llm_per_mode(iterations: List[Dict[str, object]], out_dir: Path, plt):
    xvals, labels = iteration_labels(iterations)
    mode_order = ["assisted", "autonomous"]
    if should_include_original(iterations):
        mode_order.append("original")

    llms = set()
    for mode in mode_order:
        for it in iterations:
            llms.update(it["llm_by_mode"][mode].keys())  # type: ignore[index]
    llms = sorted(llms)

    if not mode_order:
        return None

    fig, axes = plt.subplots(1, len(mode_order), figsize=(5 * len(mode_order), 5), sharey=True)
    if len(mode_order) == 1:
        axes = [axes]

    for i, mode in enumerate(mode_order):
        ax = axes[i]
        for llm in llms:
            ys = [it["llm_by_mode"][mode].get(llm, 0) for it in iterations]
            ax.plot(xvals, ys, marker="o", label=llm)
        ax.set_title(f"{mode.title()} LLM errors")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error count")
        ax.set_xticks(xvals)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, alpha=0.25)
        if llms:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = out_dir / "llm_errors_per_mode_over_iterations.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_llm_overall(iterations: List[Dict[str, object]], out_dir: Path, plt):
    xvals, labels = iteration_labels(iterations)
    all_llms = sorted({llm for it in iterations for llm in it["llm_total"]})
    fig, ax = plt.subplots(figsize=(10, 6))
    combined = []
    for llm in all_llms:
        ys = [it["llm_total"].get(llm, 0) for it in iterations]
        combined = [sum(pair) for pair in zip(combined, ys)] if combined else ys[:]
        ax.plot(xvals, ys, marker="o", label=llm)

    if iterations:
        all_totals = [sum(it["llm_total"].values()) for it in iterations]
        ax.plot(xvals, all_totals, marker="x", linestyle="--", linewidth=2, label="Combined")

    ax.set_title("LLM errors over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error count")
    ax.set_xticks(xvals)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "llm_errors_over_iterations.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_error_types_over_iterations(iterations: List[Dict[str, object]], out_dir: Path, plt, max_types: int):
    xvals, labels = iteration_labels(iterations)
    global_type_order = sorted(
        ((k, int(v)) for it in iterations for k, v in it["error_types"].items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_types = []
    seen = set()
    for err, _ in global_type_order:
        if err in seen:
            continue
        seen.add(err)
        top_types.append(err)
    if max_types > 0:
        top_types = top_types[:max_types]

    if not top_types:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = []
    for err in top_types:
        ys = [d["error_types"].get(err, 0) for d in iterations]
        line, = ax.plot(xvals, ys, marker="o")
        plotted.append((line, err))

    ax.set_title("Error type trend over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error count")
    ax.set_xticks(xvals)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, alpha=0.25)
    handles = [h for h, _ in plotted]
    display_labels = [lbl for _, lbl in plotted]
    ax.legend(handles, display_labels, loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    out = out_dir / "error_type_over_iterations.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_last_iteration_error_types(iterations: List[Dict[str, object]], out_dir: Path, plt):
    if not iterations:
        return None
    last = iterations[-1]
    assisted = last["error_types_by_mode"]["assisted"]  # type: ignore[index]
    autonomous = last["error_types_by_mode"]["autonomous"]  # type: ignore[index]
    errs = sorted(set(assisted) | set(autonomous))
    if not errs:
        return None

    idx = list(range(len(errs)))
    width = 0.35
    auto_vals = [autonomous.get(e, 0) for e in errs]
    assist_vals = [assisted.get(e, 0) for e in errs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = [0 for _ in errs]
    ax.bar(idx, auto_vals, width=0.5, label="autonomous")
    ax.bar(idx, assist_vals, width=0.5, bottom=auto_vals, label="assisted")
    for i in idx:
        total = auto_vals[i] + assist_vals[i]
        if total > 0:
            ax.text(
                i,
                total + 0.2,
                str(total),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(f"Error types by mode (final iteration)")
    ax.set_xlabel("Error type")
    ax.set_ylabel("Error count")
    ax.set_xticks(idx)
    ax.set_xticklabels(errs, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = out_dir / f"error_types_last_iteration_{last['iteration']}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_mode_totals(iterations: List[Dict[str, object]], out_dir: Path, plt):
    xvals, labels = iteration_labels(iterations)
    assisted = [d["mode_totals"]["assisted"] for d in iterations]
    autonomous = [d["mode_totals"]["autonomous"] for d in iterations]
    include_original = should_include_original(iterations)
    original = [d["mode_totals"]["original"] for d in iterations] if include_original else []

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xvals, autonomous, marker="o", label="autonomous")
    ax.plot(xvals, assisted, marker="o", label="assisted")

    if include_original:
        ax.plot(xvals, original, marker="o", label="original")
    ax.set_title("Errors by mode over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error count")
    ax.set_xticks(xvals)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "errors_by_mode_over_iterations.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    plt = import_plot_libs()

    root_dir = Path(args.root_dir).resolve() if args.root_dir else Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    iteration_dirs = find_iterations(root_dir)
    parsed: List[Dict[str, object]] = []
    for it in iteration_dirs:
        analysis_path = it / ANALYSIS_FILENAME
        if not analysis_path.is_file():
            print(f"[warn] Missing analysis file: {analysis_path}")
            continue
        parsed.append(parse_analysis_file(analysis_path))

    if not parsed:
        raise SystemExit("No analysis files found to analyze.")

    parsed = sorted(parsed, key=lambda d: int(d["iteration_index"]))

    plot_paths = [
        plot_mode_totals(parsed, out_dir, plt),
        plot_llm_per_mode(parsed, out_dir, plt),
        plot_llm_overall(parsed, out_dir, plt),
        plot_error_types_over_iterations(parsed, out_dir, plt, args.max_error_types),
        plot_last_iteration_error_types(parsed, out_dir, plt),
    ]

    for p in plot_paths:
        if p:
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
