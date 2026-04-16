#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPOS_DIR = REPO_ROOT / "repos"
OUTPUT_DIR = REPO_ROOT / "results" / "file_size_analysis"
GENERATED_PREFIX = "GENAIGREENML"
GENERATED_SUFFIX = ".py"
MODES = ("assisted", "autonomous", "original", "unknown")
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 16
PLOT_TITLE_FONTSIZE = 18
IGNORE_DIR_NAMES = {
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
}


@dataclass
class SizeRecord:
    project: str
    relative_path: str
    file: str
    mode: str
    llm: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze GENAIGREENML file sizes in repos/* and write summaries to "
            "results/file_size_analysis"
        )
    )
    p.add_argument(
        "--repos-dir",
        default=str(DEFAULT_REPOS_DIR),
        help="Directory containing projects to scan (default: ./repos).",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory for all file-size analysis artefacts.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many rows to include for top/bottom largest files (default: 10).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all plotting; still create csv/txt outputs.",
    )
    return p.parse_args()


def normalize_size(size_bytes: int) -> float:
    return round(size_bytes / 1024, 3)


def detect_mode(file_name: str) -> str:
    n = file_name.lower()
    if "assisted" in n:
        return "assisted"
    if "autonomous" in n:
        return "autonomous"
    if "original" in n:
        return "original"
    return "unknown"


def detect_llm(file_name: str) -> str:
    stem = Path(file_name).stem
    if "_" not in stem:
        return "unknown"
    return stem.rsplit("_", 1)[-1].lower()


def is_generated_file(name: str) -> bool:
    return name.startswith(GENERATED_PREFIX) and name.endswith(GENERATED_SUFFIX)


def iter_generated_files(repos_dir: Path):
    for root, dirs, files in os.walk(repos_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES and not d.startswith(".")]
        for file in files:
            if is_generated_file(file):
                yield Path(root) / file


def safe_stdev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return stdev(values)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return float(values[0])
    if p >= 100:
        return float(values[-1])
    idx = (len(values) - 1) * (p / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(values[int(idx)])
    weight = idx - lo
    return float(values[lo] * (1 - weight) + values[hi] * weight)


def summarise(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}

    sorted_vals = sorted(values)
    m = mean(sorted_vals)
    med = median(sorted_vals)
    sd = safe_stdev([float(v) for v in sorted_vals])
    return {
        "count": float(len(sorted_vals)),
        "mean": float(m),
        "median": float(med),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "std_dev": sd if sd is not None else float("nan"),
        "p10": percentile(sorted_vals, 10),
        "p25": percentile(sorted_vals, 25),
        "p50": percentile(sorted_vals, 50),
        "p75": percentile(sorted_vals, 75),
        "p90": percentile(sorted_vals, 90),
    }


def format_number(v: float | int) -> str:
    if isinstance(v, int):
        return f"{v}"
    if isinstance(v, float) and math.isfinite(v):
        if v.is_integer():
            return f"{int(v)}"
        return f"{v:.3f}"
    return "n/a"


def kb(v: float) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(fv):
        return "n/a"
    return f"{fv/1024:.3f} KB"


def build_records(repos_dir: Path) -> List[SizeRecord]:
    if not repos_dir.is_dir():
        raise SystemExit(f"Repos directory does not exist: {repos_dir}")

    records: List[SizeRecord] = []
    for file_path in iter_generated_files(repos_dir):
        try:
            rel = file_path.relative_to(repos_dir)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 2:
            continue
        project = parts[0]
        file_name = file_path.name
        mode = detect_mode(file_name)
        llm = detect_llm(file_name)
        size = file_path.stat().st_size
        records.append(
            SizeRecord(
                project=project,
                relative_path=str(rel),
                file=file_path.name,
                mode=mode,
                llm=llm,
                size_bytes=size,
            )
        )
    return records


def import_plot_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams.update(
            {
                "axes.labelsize": AXIS_LABEL_FONTSIZE,
                "xtick.labelsize": TICK_LABEL_FONTSIZE,
                "ytick.labelsize": TICK_LABEL_FONTSIZE,
                "axes.titlesize": PLOT_TITLE_FONTSIZE,
            }
        )
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError as e:
        pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
        raise RuntimeError(
            f"Plotting requested but dependency missing: {pkg}. Install with: python3 -m pip install matplotlib"
        )


def save_raw_csv(path: Path, records: List[SizeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["project", "relative_path", "mode", "llm", "size_bytes", "size_kb"])
        for r in sorted(records, key=lambda x: (x.project, x.mode, x.llm, x.file)):
            kb_size = normalize_size(r.size_bytes)
            writer.writerow([r.project, r.relative_path, r.mode, r.llm, r.size_bytes, f"{kb_size:.3f}"])


def save_summary_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    header = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in header})


def non_generated_project_sizes(repos_dir: Path) -> Dict[str, int]:
    project_sizes: Dict[str, int] = defaultdict(int)
    for root, dirs, files in os.walk(repos_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES and not d.startswith(".")]
        for file_name in files:
            if is_generated_file(file_name):
                continue
            file_path = Path(root) / file_name
            rel = file_path.relative_to(repos_dir)
            if len(rel.parts) < 2:
                continue
            project = rel.parts[0]
            try:
                project_sizes[project] += file_path.stat().st_size
            except OSError:
                continue
    return project_sizes


def collect_group_sizes(records: List[SizeRecord], key):
    grouped: Dict[str, List[int]] = defaultdict(list)
    for r in records:
        grouped[key(r)].append(r.size_bytes)
    return grouped


def group_stats_to_rows(group_name: str, grouped: Dict[str, List[int]]) -> List[Dict[str, str]]:
    rows = []
    for group_key, vals in sorted(grouped.items(), key=lambda kv: (str(kv[0]))):
        stats = summarise(vals)
        if stats["count"] == 0:
            continue
        rows.append(
            {
                "group_type": group_name,
                "group_key": group_key,
                "count": format_number(stats["count"]),
                "mean_bytes": format_number(stats["mean"]),
                "median_bytes": format_number(stats["median"]),
                "std_dev": format_number(stats["std_dev"]),
                "min_bytes": format_number(stats["min"]),
                "p10_bytes": format_number(stats["p10"]),
                "p25_bytes": format_number(stats["p25"]),
                "p50_bytes": format_number(stats["p50"]),
                "p75_bytes": format_number(stats["p75"]),
                "p90_bytes": format_number(stats["p90"]),
                "max_bytes": format_number(stats["max"]),
            }
        )
    return rows


def add_mode_llm_project_deltas(records: List[SizeRecord]) -> Dict[str, Dict[str, float]]:
    # Compare generated files to project-level originals with same LLM
    original_sizes = {}
    for r in records:
        if r.mode != "original":
            continue
        original_sizes[(r.project, r.llm)] = r.size_bytes

    deltas_per_mode_llm = {
        "assisted": defaultdict(list),
        "autonomous": defaultdict(list),
    }
    matched = {"assisted": 0, "autonomous": 0}
    unmatched = {"assisted": 0, "autonomous": 0}

    for r in records:
        if r.mode not in ("assisted", "autonomous"):
            continue
        ref = original_sizes.get((r.project, r.llm))
        if ref is None:
            unmatched[r.mode] += 1
            continue
        matched[r.mode] += 1
        deltas_per_mode_llm[r.mode][r.llm].append(r.size_bytes - ref)

    out: Dict[str, Dict[str, float]] = {}
    for mode, by_llm in deltas_per_mode_llm.items():
        for llm, vals in by_llm.items():
            out[f"{mode}/{llm}"] = summarise(vals)
    return {
        "matched_counts": matched,
        "unmatched_counts": unmatched,
        "delta_stats": out,
    }


def plot_box_by_group(records: List[SizeRecord], output_path: Path, by: str) -> List[Path]:
    plt = import_plot_libs()
    if not records:
        return []

    if by == "mode":
        labels = MODES[:3]
        groups = [[normalize_size(r.size_bytes) for r in records if r.mode == m] for m in labels]
        path = output_path / "boxplot_size_by_mode.png"
        #title = "File size by mode"
        x_labels = [f"{label} (n={len(g)})" for label, g in zip(labels, groups)]
        fig, ax = plt.subplots(figsize=(10, 6))
        if any(groups):
            ax.boxplot([g for g in groups if g], labels=[x for g, x in zip(groups, x_labels) if g])
            ax.set_xticks(range(1, 1 + len([g for g in groups if g])))
            ax.set_xticklabels([x for g, x in zip(groups, x_labels) if g], rotation=20, ha="right")
       # ax.set_title(title)
        ax.set_ylabel("Size (KB)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return [path]

    llms = sorted({r.llm for r in records})
    path = output_path / "boxplot_size_by_llm.png"
    groups = [[normalize_size(r.size_bytes) for r in records if r.llm == llm] for llm in llms]
    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(llms)), 6))
    if any(groups):
        ax.boxplot(groups, labels=llms)
    #ax.set_title("File size by LLM")
    ax.set_ylabel("Size (KB)")
    ax.set_xticklabels([f"{llm}\n(n={len(g)})" for llm, g in zip(llms, groups)], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def plot_histogram(records: List[SizeRecord], output_path: Path) -> Path | None:
    if not records:
        return None
    plt = import_plot_libs()
    sizes = [r.size_bytes / 1024 for r in records]
    path = output_path / "hist_size_kb.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sizes, bins=25, alpha=0.9)
    ax.set_title("Distribution of GENAIGREENML file sizes")
    ax.set_xlabel("File size (KB)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    repos_dir = Path(args.repos_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(repos_dir)
    if not records:
        raise SystemExit(f"No GENAIGREENML .py files found in {repos_dir}")
    nongenerated_project_bytes = non_generated_project_sizes(repos_dir)

    records_sorted = sorted(records, key=lambda r: (r.project, r.mode, r.llm, r.file))
    by_mode = collect_group_sizes(records_sorted, lambda r: r.mode)
    by_llm = collect_group_sizes(records_sorted, lambda r: r.llm)
    by_mode_llm = collect_group_sizes(records_sorted, lambda r: f"{r.mode}|{r.llm}")
    by_project = collect_group_sizes(records_sorted, lambda r: r.project)

    stats_overall = summarise([r.size_bytes for r in records_sorted])
    stats_original = summarise([r.size_bytes for r in records_sorted if r.mode == "original"])
    stats_assisted = summarise([r.size_bytes for r in records_sorted if r.mode == "assisted"])
    stats_autonomous = summarise([r.size_bytes for r in records_sorted if r.mode == "autonomous"])

    mode_rows = group_stats_to_rows("mode", by_mode)
    llm_rows = group_stats_to_rows("llm", by_llm)
    mode_llm_rows = group_stats_to_rows("mode_llm", by_mode_llm)
    project_rows = group_stats_to_rows("project", by_project)

    all_summary_rows: List[Dict[str, str]] = [
        {
            "group_type": "overall",
            "group_key": "all",
            **{k: format_number(v) for k, v in {
                "count": stats_overall["count"],
                "mean_bytes": stats_overall["mean"],
                "median_bytes": stats_overall["median"],
                "std_dev": stats_overall["std_dev"],
                "min_bytes": stats_overall["min"],
                "p10_bytes": stats_overall["p10"],
                "p25_bytes": stats_overall["p25"],
                "p50_bytes": stats_overall["p50"],
                "p75_bytes": stats_overall["p75"],
                "p90_bytes": stats_overall["p90"],
                "max_bytes": stats_overall["max"],
            }.items()},
        }
    ]

    for r in mode_rows:
        all_summary_rows.append(r)
    for r in llm_rows:
        all_summary_rows.append(r)
    for r in mode_llm_rows:
        all_summary_rows.append(r)
    for r in project_rows:
        all_summary_rows.append(r)

    summary_csv = output_dir / "file_size_summary_stats.csv"
    save_summary_csv(summary_csv, all_summary_rows)

    raw_csv = output_dir / "file_size_records.csv"
    save_raw_csv(raw_csv, records_sorted)

    biggest = max(records_sorted, key=lambda r: r.size_bytes)
    sorted_by_size = sorted(records_sorted, key=lambda r: r.size_bytes, reverse=True)
    smallest = sorted(records_sorted, key=lambda r: r.size_bytes)
    deltas = add_mode_llm_project_deltas(records_sorted)

    lines = []
    lines.append("GENAIGREENML File Size Analysis")
    lines.append("=" * 34)
    lines.append(f"Repos directory: {repos_dir}")
    lines.append(f"Projects scanned: {len(set(r.project for r in records_sorted))}")
    lines.append(f"Total files analyzed: {len(records_sorted)}")
    lines.append("")
    lines.append("Requested metrics")
    lines.append(f"Average original file size: {kb(stats_original['mean'])}")
    lines.append(f"Median  original file size: {kb(stats_original['median'])}")
    lines.append("")
    lines.append("Average file size per LLM:")
    for llm, vs in sorted(by_llm.items()):
        if llm == "unknown":
            continue
        lines.append(f"  {llm}: {kb(summarise(vs)['mean'])} (n={len(vs)})")
    lines.append("Median file size per LLM:")
    for llm, vs in sorted(by_llm.items()):
        if llm == "unknown":
            continue
        lines.append(f"  {llm}: {kb(summarise(vs)['median'])} (n={len(vs)})")
    lines.append(f"Average file size per mode: assisted={kb(stats_assisted['mean'])}, autonomous={kb(stats_autonomous['mean'])}, original={kb(stats_original['mean'])}")
    lines.append(f"Median file size per mode: assisted={kb(stats_assisted['median'])}, autonomous={kb(stats_autonomous['median'])}, original={kb(stats_original['median'])}")

    lines.append("Average file size per mode per LLM:")
    for mode in ("assisted", "autonomous", "original", "unknown"):
        for llm in sorted({r.llm for r in records_sorted if r.mode == mode}):
            vals = by_mode_llm.get(f"{mode}|{llm}", [])
            if not vals:
                continue
            s = summarise(vals)
            lines.append(f"  {mode}/{llm}: mean={kb(s['mean'])}, median={kb(s['median'])}, n={int(s['count'])}")

    lines.append("")
    lines.append("Additional analyses")
    total_size = sum(r.size_bytes for r in records_sorted)
    lines.append("Share of total bytes by mode:")
    for mode, vals in sorted(by_mode.items()):
        if not vals:
            continue
        share = sum(vals) / total_size * 100 if total_size else 0.0
        lines.append(f"  {mode}: {share:.2f}% (n={len(vals)})")
    lines.append("Share of total bytes by LLM:")
    for llm, vals in sorted(by_llm.items()):
        if not vals:
            continue
        share = sum(vals) / total_size * 100 if total_size else 0.0
        lines.append(f"  {llm}: {share:.2f}% (n={len(vals)})")
    lines.append("Top projects by total bytes (top 10)")
    project_bytes = {p: sum(v) for p, v in by_project.items()}
    for project, total in sorted(project_bytes.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        lines.append(f"  {project}: {kb(total)}")
    top_count = max(1, args.top_k)
    lines.append(f"Top {top_count} largest files")
    for i, r in enumerate(sorted_by_size[:top_count], start=1):
        lines.append(f"  {i:>2}. {r.project}/{r.relative_path} — {kb(r.size_bytes)} ({r.mode}/{r.llm})")
    lines.append(f"Top {top_count} smallest files")
    for i, r in enumerate(smallest[:top_count], start=1):
        lines.append(f"  {i:>2}. {r.project}/{r.relative_path} — {kb(r.size_bytes)} ({r.mode}/{r.llm})")

    lines.append("")
    lines.append(f"Biggest file: {biggest.project}/{biggest.relative_path}")
    lines.append(f"  bytes={biggest.size_bytes} ({kb(biggest.size_bytes)}) mode={biggest.mode} llm={biggest.llm}")
    lines.append("")
    lines.append(f"Mode counts: { {k: len(v) for k, v in by_mode.items() if v} }")
    lines.append(f"LLM counts: { {k: len(v) for k, v in by_llm.items() if v} }")

    lines.append("")
    lines.append("Within-project comparison (generated vs original, same LLM)")
    lines.append(f"Matched files: assisted={deltas['matched_counts']['assisted']}, autonomous={deltas['matched_counts']['autonomous']}")
    lines.append(f"Unmatched files: assisted={deltas['unmatched_counts']['assisted']}, autonomous={deltas['unmatched_counts']['autonomous']}")
    for k, v in sorted(deltas["delta_stats"].items()):
        if not math.isfinite(v["mean"]):
            lines.append(f"  {k}: n={int(v['count'])}, delta stats unavailable (insufficient numeric data)")
            continue
        lines.append(
            f"  {k}: n={int(v['count'])}, "
            f"mean delta={kb(v['mean']) if math.isfinite(v['mean']) else 'n/a'}, "
            f"median delta={kb(v['median']) if math.isfinite(v['median']) else 'n/a'}, "
            f"p90 delta={kb(v['p90']) if math.isfinite(v['p90']) else 'n/a'}, "
            f"p10 delta={kb(v['p10']) if math.isfinite(v['p10']) else 'n/a'}, "
            f"min={kb(v['min']) if math.isfinite(v['min']) else 'n/a'}, "
            f"max={kb(v['max']) if math.isfinite(v['max']) else 'n/a'}"
        )

    lines.append("")
    lines.append("Project-level summary (top 15 mean file sizes)")
    project_mean = sorted(
        ((project, mean(vs)) for project, vs in by_project.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for project, avg_bytes in project_mean[: min(15, len(project_mean))]:
        lines.append(f"  {project}: {kb(avg_bytes)}")

    lines.append("")
    lines.append(f"Output files")
    lines.append(f"- records: {raw_csv}")
    lines.append(f"- summary: {summary_csv}")
    non_generated_csv = output_dir / "non_generated_project_sizes.csv"
    with non_generated_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["project", "total_bytes", "total_kb", "total_mb"])
        for project, total_bytes in sorted(nongenerated_project_bytes.items(), key=lambda kv: kv[1], reverse=True):
            writer.writerow([project, total_bytes, f"{total_bytes / 1024:.3f}", f"{total_bytes / 1024 / 1024:.3f}"])
    lines.append(f"- non-generated project sizes: {non_generated_csv}")
    lines.append("")
    lines.append("Top 10 biggest projects excluding GENAIGREENML files")
    for i, (project, total_bytes) in enumerate(
        sorted(nongenerated_project_bytes.items(), key=lambda kv: kv[1], reverse=True)[:10],
        start=1,
    ):
        lines.append(f"{i:>2}. {project}: {kb(total_bytes)}")

    if not args.no_plots:
        plot_paths: List[Path] = []
        try:
            plot_paths.extend(plot_box_by_group(records_sorted, output_dir, "mode"))
            plot_paths.extend(plot_box_by_group(records_sorted, output_dir, "llm"))
            h = plot_histogram(records_sorted, output_dir)
            if h:
                plot_paths.append(h)
        except RuntimeError as e:
            lines.append("")
            lines.append(f"Plotting skipped: {e}")
        if plot_paths:
            lines.append("Plots:")
            for p in plot_paths:
                lines.append(f"- {p}")

    txt_path = output_dir / "file_size_analysis.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {txt_path}")
    print(f"Wrote {raw_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
