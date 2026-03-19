#!/usr/bin/env python3
import argparse
import csv
import warnings
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
METRICS = ("accuracy", "exec_time", "energy")
MODES = ("assisted", "autonomous")

# Positive deltas represent improvements over original.
METRIC_CONFIG = {
    "accuracy": {"column": "accuracy", "better": "higher"},
    "exec_time": {"column": "exec_time_s", "better": "lower"},
    "energy": {"column": "energy_j", "better": "lower"},
}


def import_external_libs():
    try:
        import numpy as np  # type: ignore
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore
        from scipy import stats  # type: ignore
        from statsmodels.stats.multicomp import pairwise_tukeyhsd  # type: ignore
        from statsmodels.stats.multitest import multipletests  # type: ignore
        return np, plt, Line2D, stats, pairwise_tukeyhsd, multipletests
    except ModuleNotFoundError as e:
        pkg = str(e).split("'")[-2] if "'" in str(e) else str(e)
        raise SystemExit(
            "Missing dependency for quantitative analysis: "
            f"{pkg}. Install required packages with: "
            "python3 -m pip install numpy scipy matplotlib statsmodels"
        )


def latest_results_file() -> Path:
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"Results directory not found: {RESULTS_DIR}")
    files = [p for p in RESULTS_DIR.iterdir() if p.is_file() and p.name.startswith("results_") and p.suffix == ".csv"]
    if not files:
        raise SystemExit("No results_*.csv files found")
    return max(files, key=lambda p: p.stat().st_mtime)


def resolve_results_file(results_file: Optional[str]) -> Path:
    if not results_file:
        return latest_results_file()

    candidate = Path(results_file)
    if not candidate.is_absolute():
        candidate = RESULTS_DIR / candidate
    candidate = candidate.resolve()
    results_dir_resolved = RESULTS_DIR.resolve()

    if not candidate.is_file():
        raise SystemExit(f"Results file not found: {candidate}")
    if results_dir_resolved not in candidate.parents:
        raise SystemExit(f"Results file must be inside {RESULTS_DIR}/: {candidate}")
    if candidate.suffix.lower() != ".csv":
        raise SystemExit(f"Results file must be a .csv file: {candidate}")
    return candidate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze benchmark results in results/*.csv")
    p.add_argument(
        "--results-file",
        default=None,
        help="Optional CSV filename/path inside results/ (default: newest results_*.csv).",
    )
    return p.parse_args()


def to_float(val):
    try:
        return float(val)
    except Exception:
        return None


def rounded(val, places):
    if val is None:
        return None
    return round(val, places)


def paired_cohens_d(diff):
    arr = np.asarray(diff, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float("nan")
    denom = arr.std(ddof=1)
    if denom == 0:
        return float("nan")
    return float(arr.mean() / denom)


def paired_ci_mean(diff, alpha=0.05):
    arr = np.asarray(diff, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))

    mean_diff = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return (mean_diff, mean_diff)

    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    half_width = t_crit * se
    return float(mean_diff - half_width), float(mean_diff + half_width)


def holm_adjust_pvalues(p_values):
    arr = np.asarray(p_values, dtype=float)
    if arr.size == 0:
        return []

    out = np.full(arr.shape, np.nan, dtype=float)
    valid = ~np.isnan(arr)
    if np.any(valid):
        out[valid] = multipletests(arr[valid], alpha=0.05, method="holm")[1]
    return out.tolist()


def save_performance_category_plots(analysis_dir: Path, latest: Path, counts: dict, plt, np) -> list[Path]:
    mode_names = [m.title() for m in MODES]
    x = np.arange(len(MODES))
    width = 0.55

    metric_labels = {
        "accuracy": "Accuracy",
        "exec_time": "Execution time",
        "energy": "Energy",
    }

    color_inc = "#2ca02c"
    color_dec = "#d62728"
    color_eq = "#7f7f7f"

    stacked_path = analysis_dir / f"{latest.name}_analysis_mode_performance_stacked.png"
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5), sharey=False)
    if len(METRICS) == 1:
        axes = [axes]
    for ax, metric in zip(axes, METRICS):
        inc = np.array([counts[m][metric]["inc"] for m in MODES], dtype=float)
        dec = np.array([counts[m][metric]["dec"] for m in MODES], dtype=float)
        eq = np.array([counts[m][metric]["eq"] for m in MODES], dtype=float)

        ax.bar(x, inc, width, color=color_inc, label="inc")
        ax.bar(x, dec, width, bottom=inc, color=color_dec, label="dec")
        ax.bar(x, eq, width, bottom=inc + dec, color=color_eq, label="eq")

        for i in range(len(MODES)):
            total = inc[i] + dec[i] + eq[i]
            if total > 0:
                ax.text(x[i], total + max(1, 0.02 * total), f"{int(total)}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(mode_names)
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_ylabel("Projects/rows")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("LLM performance relative to original (count, stacked)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(stacked_path, dpi=160)
    plt.close(fig)

    percent_path = analysis_dir / f"{latest.name}_analysis_mode_performance_stacked_pct.png"
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5), sharey=False)
    if len(METRICS) == 1:
        axes = [axes]
    for ax, metric in zip(axes, METRICS):
        inc = np.array([counts[m][metric]["inc"] for m in MODES], dtype=float)
        dec = np.array([counts[m][metric]["dec"] for m in MODES], dtype=float)
        eq = np.array([counts[m][metric]["eq"] for m in MODES], dtype=float)
        totals = inc + dec + eq
        with np.errstate(divide="ignore", invalid="ignore"):
            inc_pct = np.where(totals > 0, inc / totals * 100.0, 0.0)
            dec_pct = np.where(totals > 0, dec / totals * 100.0, 0.0)
            eq_pct = np.where(totals > 0, eq / totals * 100.0, 0.0)

        ax.bar(x, inc_pct, width, color=color_inc, label="inc")
        ax.bar(x, dec_pct, width, bottom=inc_pct, color=color_dec, label="dec")
        ax.bar(x, eq_pct, width, bottom=inc_pct + dec_pct, color=color_eq, label="eq")

        for i in range(len(MODES)):
            if totals[i] > 0:
                ax.text(
                    x[i],
                    101,
                    f"n={int(totals[i])}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(mode_names)
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_ylabel("Percentage")
        ax.tick_params(axis="x", rotation=0)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("LLM performance relative to original (percentage by mode)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(percent_path, dpi=160)
    plt.close(fig)

    net_path = analysis_dir / f"{latest.name}_analysis_mode_performance_net_inc_minus_dec.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    n_metrics = len(METRICS)
    ind = np.arange(n_metrics)
    bar_width = 0.35

    assisted_net = np.array([counts["assisted"][metric]["inc"] - counts["assisted"][metric]["dec"] for metric in METRICS], dtype=float)
    autonomous_net = np.array([counts["autonomous"][metric]["inc"] - counts["autonomous"][metric]["dec"] for metric in METRICS], dtype=float)

    ax.axhline(0, color="black", linewidth=1)
    ax.bar(ind - bar_width / 2, assisted_net, bar_width, label="assisted", color="#1f77b4")
    ax.bar(ind + bar_width / 2, autonomous_net, bar_width, label="autonomous", color="#ff7f0e")

    for i in range(n_metrics):
        ax.text(
            ind[i] - bar_width / 2,
            assisted_net[i] + (0.1 if assisted_net[i] >= 0 else -0.3),
            str(int(assisted_net[i])),
            ha="center",
            va="bottom" if assisted_net[i] >= 0 else "top",
            fontsize=8,
        )
        ax.text(
            ind[i] + bar_width / 2,
            autonomous_net[i] + (0.1 if autonomous_net[i] >= 0 else -0.3),
            str(int(autonomous_net[i])),
            ha="center",
            va="bottom" if autonomous_net[i] >= 0 else "top",
            fontsize=8,
        )

    ax.set_xticks(ind)
    ax.set_xticklabels([metric_labels.get(m, m) for m in METRICS])
    ax.set_title("LLM performance net score (inc - dec)")
    ax.set_ylabel("Net score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(net_path, dpi=160)
    plt.close(fig)

    return [stacked_path, percent_path, net_path]


def detect_mode(script: str) -> Optional[str]:
    s = script.lower()
    if "assisted" in s:
        return "assisted"
    if "autonomous" in s:
        return "autonomous"
    return None


def detect_llm(script: str) -> str:
    stem = Path(script).stem
    if "_" not in stem:
        return "unknown"
    return stem.rsplit("_", 1)[-1].lower()


def compute_delta(metric: str, gen: float, orig: float) -> float:
    better = METRIC_CONFIG[metric]["better"]
    if better == "higher":
        return gen - orig
    return orig - gen


def main():
    np, plt, Line2D, stats, pairwise_tukeyhsd, multipletests = import_external_libs()
    globals().update({"np": np, "stats": stats, "multipletests": multipletests})

    args = parse_args()
    latest = resolve_results_file(args.results_file)
    analysis_dir = RESULTS_DIR / f"{latest.name}_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_output = analysis_dir / f"{latest.name}_analysis.txt"

    all_rows = []
    ok_rows = []
    failed = []
    with latest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
            if row.get("status", "").strip().upper() != "OK":
                project = row.get("project", "").strip()
                script = row.get("script", "").strip()
                if project or script:
                    failed.append(f"{project}: {script}")
                continue
            ok_rows.append(row)

    projects_all = {r.get("project", "").strip() for r in all_rows if r.get("project", "").strip()}
    scripts_all = {
        (r.get("project", "").strip(), r.get("script", "").strip())
        for r in all_rows
        if r.get("project", "").strip() and r.get("script", "").strip()
    }
    assisted_autonomous_all = {
        (r.get("project", "").strip(), r.get("script", "").strip())
        for r in all_rows
        if r.get("project", "").strip()
        and r.get("script", "").strip()
        and ("assisted" in r.get("script", "") or "autonomous" in r.get("script", ""))
    }
    scripts_ok = {
        (r.get("project", "").strip(), r.get("script", "").strip())
        for r in ok_rows
        if r.get("project", "").strip() and r.get("script", "").strip()
    }
    assisted_autonomous_ok = {
        (r.get("project", "").strip(), r.get("script", "").strip())
        for r in ok_rows
        if r.get("project", "").strip()
        and r.get("script", "").strip()
        and ("assisted" in r.get("script", "") or "autonomous" in r.get("script", ""))
    }

    original_by_project = {}
    for r in ok_rows:
        script = r.get("script", "")
        project = r.get("project", "")
        if "original_telemetry" in script and project not in original_by_project:
            original_by_project[project] = r

    rows = [r for r in ok_rows if r.get("project") in original_by_project]
    aa_ok_with_comparable_original = sum(
        1
        for r in rows
        if "assisted" in r.get("script", "") or "autonomous" in r.get("script", "")
    )

    lines = []
    lines.append(f"Results file: {latest}")
    lines.append(f"ML Projects: {len(projects_all)}")
    lines.append(f"ML Projects with original OK: {len(original_by_project)}")
    lines.append(f"Distinct code files: {len(scripts_all)}")
    lines.append(f"Distinct code files (OK): {len(scripts_ok)}")
    lines.append(f"Assisted + autonomous code files: {len(assisted_autonomous_all)}")
    lines.append(f"Assisted + autonomous code files (OK): {len(assisted_autonomous_ok)}")
    lines.append(f"A+A OK with comparable original: {aa_ok_with_comparable_original}")

    def init_counts():
        return {"inc": 0, "dec": 0, "eq": 0}

    counts = {
        "assisted": {"accuracy": init_counts(), "exec_time": init_counts(), "energy": init_counts()},
        "autonomous": {"accuracy": init_counts(), "exec_time": init_counts(), "energy": init_counts()},
    }
    counts_by_llm = {}

    deltas_by_mode_metric_llm = {
        mode: {metric: {} for metric in METRICS}
        for mode in MODES
    }
    raw_by_mode_metric_llm = {
        mode: {metric: {} for metric in METRICS}
        for mode in MODES
    }
    paired_vs_original = {
        mode: {metric: {} for metric in METRICS}
        for mode in MODES
    }
    paired_lookup = {}
    raw_paired_lookup = {}
    raw_original = {metric: [] for metric in METRICS}

    for r in rows:
        script = r.get("script", "")
        project = r.get("project", "")
        mode = detect_mode(script)
        if mode not in MODES:
            continue

        llm = detect_llm(script)
        orig = original_by_project.get(project)
        if not orig:
            continue

        metric_vals = {}
        for metric in METRICS:
            col = METRIC_CONFIG[metric]["column"]
            gen_val = to_float(r.get(col, ""))
            orig_val = to_float(orig.get(col, ""))
            metric_vals[metric] = (gen_val, orig_val)

        acc = rounded(metric_vals["accuracy"][0], 2)
        acc0 = rounded(metric_vals["accuracy"][1], 2)
        ex = rounded(metric_vals["exec_time"][0], 1)
        ex0 = rounded(metric_vals["exec_time"][1], 1)
        en = rounded(metric_vals["energy"][0], 1)
        en0 = rounded(metric_vals["energy"][1], 1)

        def update(metric, a, b):
            if a is None or b is None:
                return
            if a > b:
                counts[mode][metric]["inc"] += 1
            elif a < b:
                counts[mode][metric]["dec"] += 1
            else:
                counts[mode][metric]["eq"] += 1

        update("accuracy", acc, acc0)
        update("exec_time", ex, ex0)
        update("energy", en, en0)

        if llm not in counts_by_llm:
            counts_by_llm[llm] = {
                "assisted": {"accuracy": init_counts(), "exec_time": init_counts(), "energy": init_counts()},
                "autonomous": {"accuracy": init_counts(), "exec_time": init_counts(), "energy": init_counts()},
            }

        def update_llm(metric, a, b):
            if a is None or b is None:
                return
            if a > b:
                counts_by_llm[llm][mode][metric]["inc"] += 1
            elif a < b:
                counts_by_llm[llm][mode][metric]["dec"] += 1
            else:
                counts_by_llm[llm][mode][metric]["eq"] += 1

        update_llm("accuracy", acc, acc0)
        update_llm("exec_time", ex, ex0)
        update_llm("energy", en, en0)

        for metric in METRICS:
            gen_val, orig_val = metric_vals[metric]
            if gen_val is None or orig_val is None:
                continue

            delta = compute_delta(metric, gen_val, orig_val)
            deltas_by_mode_metric_llm[mode][metric].setdefault(llm, []).append(delta)
            paired_vs_original[mode][metric].setdefault(llm, []).append((gen_val, orig_val))
            raw_by_mode_metric_llm[mode][metric].setdefault(llm, []).append(gen_val)

            key = (llm, metric, project)
            paired_lookup.setdefault(key, {})[mode] = delta
            raw_paired_lookup.setdefault(key, {})[mode] = gen_val

    for r in original_by_project.values():
        for metric in METRICS:
            col = METRIC_CONFIG[metric]["column"]
            original_val = to_float(r.get(col, ""))
            if original_val is None:
                continue
            raw_original[metric].append(original_val)

    for mode in MODES:
        lines.append(f"\n{mode.capitalize()} vs Original")
        for metric in METRICS:
            c = counts[mode][metric]
            lines.append(f"{metric}: inc={c['inc']} dec={c['dec']} eq={c['eq']}")

    combined = {metric: init_counts() for metric in METRICS}
    for metric in METRICS:
        for mode in MODES:
            combined[metric]["inc"] += counts[mode][metric]["inc"]
            combined[metric]["dec"] += counts[mode][metric]["dec"]
            combined[metric]["eq"] += counts[mode][metric]["eq"]

    lines.append("\nAssisted + Autonomous (combined)")
    for metric in METRICS:
        c = combined[metric]
        lines.append(f"{metric}: inc={c['inc']} dec={c['dec']} eq={c['eq']}")

    if counts_by_llm:
        lines.append("\nBy LLM")
        for llm in sorted(counts_by_llm.keys()):
            lines.append(f"{llm}:")
            for mode in MODES:
                parts = []
                for metric in METRICS:
                    c = counts_by_llm[llm][mode][metric]
                    parts.append(f"{metric} inc={c['inc']} dec={c['dec']} eq={c['eq']}")
                lines.append(f"{mode}: {' | '.join(parts)}")

    perf_plot_paths = save_performance_category_plots(analysis_dir, latest, counts, plt, np)
    lines.append("\nPerformance category visuals")
    for path in perf_plot_paths:
        lines.append(str(path))

    lines.append("\nQuantitative Analysis")
    lines.append("Delta definition: accuracy = generated-original; exec_time/energy = original-generated (positive means improvement)")

    delta_plot_paths = []
    raw_plot_paths = []
    ttest_tvalue_plot_paths = []

    def save_tvalue_plot(section_label: str, variant: str, metric: str, entries: list[dict]):
        section_slug = section_label.replace(" ", "_").lower()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"{section_label} [t-values, {variant}] - {metric}")
        ax.set_ylabel("t statistic")
        if not entries:
            ax.text(0.5, 0.5, "No paired data", ha="center", va="center", transform=ax.transAxes)
        else:
            x = np.arange(len(entries))
            llms = [e["llm"] for e in entries]
            t_values = [e["t"] for e in entries]
            colors = ["tab:blue" if t >= 0 else "tab:orange" for t in t_values]
            bars = ax.bar(x, t_values, color=colors, edgecolor="black", linewidth=0.7)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xticks(x, llms)
            ax.tick_params(axis="x", rotation=35)
            ax.set_ylim(-max(abs(np.max(t_values)), abs(np.min(t_values)), 1.0) * 1.4, max(abs(np.max(t_values)), abs(np.min(t_values)), 1.0) * 1.4)
            for bar, entry in zip(bars, entries):
                bar_height = bar.get_height()
                x_pos = bar.get_x() + bar.get_width() / 2
                ax.text(
                    x_pos,
                    bar_height + (0.05 if bar_height >= 0 else -0.05),
                    f"n={entry['n']}\np={entry['p_adj']:.4f}",
                    ha="center",
                    va="bottom" if bar_height >= 0 else "top",
                    fontsize=8,
                    rotation=0,
                )
            ax.legend(
                handles=[
                    Line2D([0], [0], color="tab:blue", linewidth=2, label="t >= 0"),
                    Line2D([0], [0], color="tab:orange", linewidth=2, label="t < 0"),
                ],
                loc="upper right",
                frameon=True,
            )

        plot_path = analysis_dir / f"{latest.name}_analysis_tvalues_{variant}_{section_slug}_{metric}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        ttest_tvalue_plot_paths.append((variant, str(plot_path)))

    lines.append("\nBox Plots")
    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
        plot_metric_label = "accuracy (%)" if metric == "accuracy" else metric
        fig.suptitle(f"{plot_metric_label} delta vs original (positive = better)")

        for ax, mode in zip(axes, MODES):
            by_llm = deltas_by_mode_metric_llm[mode][metric]
            llms = sorted(by_llm.keys())
            if metric == "accuracy":
                data = [[v * 100.0 for v in by_llm[llm]] for llm in llms if by_llm[llm]]
            else:
                data = [by_llm[llm] for llm in llms if by_llm[llm]]
            labels = [llm for llm in llms if by_llm[llm]]

            if not data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(mode)
                ax.set_xticks([])
            else:
                ax.boxplot(
                    data,
                    tick_labels=labels,
                    showfliers=False,  # Hide outlier dots for cleaner, standard line view.
                    showmeans=False,   # Keep default median line only.
                    medianprops={"color": "green", "linewidth": 2},
                    whiskerprops={"linewidth": 1.5},
                    capprops={"linewidth": 1.5},
                    boxprops={"linewidth": 1.5},
                )
                ax.set_title(mode)
                ax.tick_params(axis="x", rotation=35)
                if metric == "accuracy":
                    ax.set_ylabel("Delta percentage points (positive = better)")
                else:
                    ax.set_ylabel("Delta (positive = better)")
                ax.legend(
                    handles=[
                        Line2D([0], [0], color="green", linewidth=2, label="Median"),
                        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="No change (0)"),
                    ],
                    loc="upper right",
                    frameon=True,
                )
                ax.axhline(0, color="gray", linestyle="--", linewidth=1)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = analysis_dir / f"{latest.name}_analysis_boxplot_{metric}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        delta_plot_paths.append(plot_path)

    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
        metric_original = raw_original[metric]
        metric_original_plot = list(metric_original) if metric == "accuracy" else list(metric_original)
        plot_metric_label = "Accuracy" if metric == "accuracy" else metric
        fig.suptitle(f"{plot_metric_label} by mode and source (raw)")

        all_values_for_axis = list(metric_original_plot)

        for ax, mode in zip(axes, MODES):
            by_llm = raw_by_mode_metric_llm[mode][metric]
            llms = sorted(by_llm.keys())
            data = []
            labels = []

            for llm in llms:
                values = by_llm[llm]
                if not values:
                    continue
                data.append(values)
                labels.append(llm)

            if metric_original_plot:
                data.append(metric_original_plot)
                labels.append("original")

            for vals in data:
                all_values_for_axis.extend(vals)

            if metric == "accuracy":
                ylabel = "Accuracy"
            else:
                ylabel = plot_metric_label

            if not data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(mode)
                ax.set_xticks([])
            else:
                ax.boxplot(
                    data,
                    tick_labels=labels,
                    showfliers=False,  # Hide outlier dots for cleaner, standard line view.
                    showmeans=False,   # Keep default median line only.
                    medianprops={"color": "green", "linewidth": 2},
                    whiskerprops={"linewidth": 1.5},
                    capprops={"linewidth": 1.5},
                    boxprops={"linewidth": 1.5},
                )
                ax.set_title(mode)
                ax.tick_params(axis="x", rotation=35)
                ax.set_ylabel(ylabel)
                ax.legend(
                    handles=[
                        Line2D([0], [0], color="green", linewidth=2, label="Median"),
                        Line2D([0], [0], color="blue", linestyle="--", linewidth=1, label="Original"),
                    ],
                    loc="upper right",
                    frameon=True,
                )
                ax.axhline(
                    np.median(metric_original_plot) if metric_original_plot else 0,
                    color="blue",
                    linestyle="--",
                    linewidth=1,
                )

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = analysis_dir / f"{latest.name}_analysis_boxplot_raw_{metric}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        raw_plot_paths.append(plot_path)

    lines.append("\nBox Plots")
    lines.append("Delta-based (existing behavior):")
    for p in delta_plot_paths:
        lines.append(str(p))
    lines.append("Raw (including original):")
    for p in raw_plot_paths:
        lines.append(str(p))

    lines.append("\nPaired t-tests [delta]: Assisted vs Autonomous by model")
    lines.append("(paired by project, tested on delta values)")
    lines.append("Holm correction: per metric, across LLMs in this section")
    section_tvals = {}
    for metric in METRICS:
        lines.append(f"\nMetric: {metric}")
        all_llms = sorted(set(deltas_by_mode_metric_llm["assisted"][metric].keys()) | set(deltas_by_mode_metric_llm["autonomous"][metric].keys()))
        rows = []
        metric_tvals = []
        for llm in all_llms:
            assisted = []
            autonomous = []
            for (k_llm, k_metric, _project), mode_map in paired_lookup.items():
                if k_llm != llm or k_metric != metric:
                    continue
                if "assisted" in mode_map and "autonomous" in mode_map:
                    assisted.append(mode_map["assisted"])
                    autonomous.append(mode_map["autonomous"])

            n_pairs = len(assisted)
            if n_pairs == 0:
                continue

            row = {"llm": llm, "n": n_pairs}
            if n_pairs < 2:
                row["status"] = "insufficient_pairs"
                rows.append(row)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat, p_val = stats.ttest_rel(np.array(assisted), np.array(autonomous), nan_policy="omit")

            row["t_stat"] = float(t_stat)
            if np.isnan(t_stat) or np.isnan(p_val):
                row["status"] = "insufficient_variance"
                rows.append(row)
            else:
                diff = np.array(assisted) - np.array(autonomous)
                mean_diff = float(np.mean(diff))
                cohen_d = paired_cohens_d(diff)
                ci_low, ci_high = paired_ci_mean(diff)
                row["p_raw"] = float(p_val)
                row["mean"] = mean_diff
                row["cohen_d"] = cohen_d
                row["ci_low"] = ci_low
                row["ci_high"] = ci_high
                rows.append(row)

        p_raw = [r["p_raw"] for r in rows if "p_raw" in r]
        p_adj = holm_adjust_pvalues(p_raw)
        p_adj_iter = iter(p_adj)

        for r in rows:
            llm = r["llm"]
            n_pairs = r["n"]
            if "status" in r:
                if r["status"] == "insufficient_pairs":
                    lines.append(f"{llm}: n={n_pairs} | insufficient pairs for t-test")
                elif r["status"] == "insufficient_variance":
                    lines.append(f"{llm}: n={n_pairs} | insufficient variance for t-test")
                continue
            p_adj_val = next(p_adj_iter)
            lines.append(
                f"{llm}: n={n_pairs} | t={r['t_stat']:.5f} | p_ttest={r['p_raw']:.6f} | "
                f"p_ttest_holm={p_adj_val:.6f} | mean(assisted-autonomous)={r['mean']:.2f} | "
                f"cohen_d={r['cohen_d']:.4f} | 95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
            )
            metric_tvals.append({"llm": llm, "n": n_pairs, "t": r["t_stat"], "p_adj": p_adj_val})

        if len(rows) == 0:
            lines.append("No paired data.")
        else:
            section_tvals[metric] = metric_tvals
    for metric in METRICS:
        save_tvalue_plot("Assisted vs Autonomous", "delta", metric, section_tvals.get(metric, []))

    lines.append("\nPaired t-tests [delta]: Original vs Assisted by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: per metric, across LLMs in this section")
    section_tvals = {}
    for metric in METRICS:
        lines.append(f"\nMetric: {metric}")
        rows = []
        metric_tvals = []
        for llm in sorted(paired_vs_original["assisted"][metric].keys()):
            pairs = paired_vs_original["assisted"][metric][llm]
            generated = np.array([g for g, _ in pairs], dtype=float)
            original = np.array([o for _, o in pairs], dtype=float)
            n_pairs = len(generated)
            if n_pairs == 0:
                continue

            row = {"llm": llm, "n": n_pairs}
            if n_pairs < 2:
                row["status"] = "insufficient_pairs"
                rows.append(row)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat, p_val = stats.ttest_rel(generated, original, nan_policy="omit")

            row["t_stat"] = float(t_stat)
            if np.isnan(t_stat) or np.isnan(p_val):
                row["status"] = "insufficient_variance"
                rows.append(row)
                continue
            diff = generated - original if metric == "accuracy" else original - generated
            mean_delta = float(np.mean(diff))
            cohen_d = paired_cohens_d(diff)
            ci_low, ci_high = paired_ci_mean(diff)
            row["p_raw"] = float(p_val)
            row["mean"] = mean_delta
            row["cohen_d"] = cohen_d
            row["ci_low"] = ci_low
            row["ci_high"] = ci_high
            rows.append(row)

        p_raw = [r["p_raw"] for r in rows if "p_raw" in r]
        p_adj = holm_adjust_pvalues(p_raw)
        p_adj_iter = iter(p_adj)

        for r in rows:
            llm = r["llm"]
            n_pairs = r["n"]
            if "status" in r:
                if r["status"] == "insufficient_pairs":
                    lines.append(f"{llm}: n={n_pairs} | insufficient pairs for t-test")
                elif r["status"] == "insufficient_variance":
                    lines.append(f"{llm}: n={n_pairs} | insufficient variance for t-test")
                continue

            p_adj_val = next(p_adj_iter)
            if metric == "accuracy":
                lines.append(
                    f"{llm}: n={n_pairs} | t={r['t_stat']:.5f} | p_ttest={r['p_raw']:.6f} | "
                    f"p_ttest_holm={p_adj_val:.6f} | mean(generated-original)={r['mean']:.2f} | "
                    f"cohen_d={r['cohen_d']:.4f} | 95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
                )
            else:
                lines.append(
                    f"{llm}: n={n_pairs} | t={r['t_stat']:.5f} | p_ttest={r['p_raw']:.6f} | "
                    f"p_ttest_holm={p_adj_val:.6f} | mean(original-generated)={r['mean']:.2f} | "
                    f"cohen_d={r['cohen_d']:.4f} | 95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
                )
            metric_tvals.append({"llm": llm, "n": n_pairs, "t": r["t_stat"], "p_adj": p_adj_val})

        if len(rows) == 0:
            lines.append("No paired data.")
        else:
            section_tvals[metric] = metric_tvals
    for metric in METRICS:
        save_tvalue_plot("Original vs Assisted", "delta", metric, section_tvals.get(metric, []))

    lines.append("\nPaired t-tests [delta]: Original vs Autonomous by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: per metric, across LLMs in this section")
    section_tvals = {}
    for metric in METRICS:
        lines.append(f"\nMetric: {metric}")
        rows = []
        metric_tvals = []
        for llm in sorted(paired_vs_original["autonomous"][metric].keys()):
            pairs = paired_vs_original["autonomous"][metric][llm]
            generated = np.array([g for g, _ in pairs], dtype=float)
            original = np.array([o for _, o in pairs], dtype=float)
            n_pairs = len(generated)
            if n_pairs == 0:
                continue

            row = {"llm": llm, "n": n_pairs}
            if n_pairs < 2:
                row["status"] = "insufficient_pairs"
                rows.append(row)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat, p_val = stats.ttest_rel(generated, original, nan_policy="omit")

            row["t_stat"] = float(t_stat)
            if np.isnan(t_stat) or np.isnan(p_val):
                row["status"] = "insufficient_variance"
                rows.append(row)
                continue
            diff = generated - original if metric == "accuracy" else original - generated
            mean_delta = float(np.mean(diff))
            cohen_d = paired_cohens_d(diff)
            ci_low, ci_high = paired_ci_mean(diff)
            row["p_raw"] = float(p_val)
            row["mean"] = mean_delta
            row["cohen_d"] = cohen_d
            row["ci_low"] = ci_low
            row["ci_high"] = ci_high
            rows.append(row)

        p_raw = [r["p_raw"] for r in rows if "p_raw" in r]
        p_adj = holm_adjust_pvalues(p_raw)
        p_adj_iter = iter(p_adj)

        for r in rows:
            llm = r["llm"]
            n_pairs = r["n"]
            if "status" in r:
                if r["status"] == "insufficient_pairs":
                    lines.append(f"{llm}: n={n_pairs} | insufficient pairs for t-test")
                elif r["status"] == "insufficient_variance":
                    lines.append(f"{llm}: n={n_pairs} | insufficient variance for t-test")
                continue

            p_adj_val = next(p_adj_iter)
            if metric == "accuracy":
                lines.append(
                    f"{llm}: n={n_pairs} | t={r['t_stat']:.5f} | p_ttest={r['p_raw']:.6f} | "
                    f"p_ttest_holm={p_adj_val:.6f} | mean(generated-original)={r['mean']:.2f} | "
                    f"cohen_d={r['cohen_d']:.4f} | 95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
                )
            else:
                lines.append(
                    f"{llm}: n={n_pairs} | t={r['t_stat']:.5f} | p_ttest={r['p_raw']:.6f} | "
                    f"p_ttest_holm={p_adj_val:.6f} | mean(original-generated)={r['mean']:.2f} | "
                    f"cohen_d={r['cohen_d']:.4f} | 95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
                )
            metric_tvals.append({"llm": llm, "n": n_pairs, "t": r["t_stat"], "p_adj": p_adj_val})

        if len(rows) == 0:
            lines.append("No paired data.")
        else:
            section_tvals[metric] = metric_tvals
    for metric in METRICS:
        save_tvalue_plot("Original vs Autonomous", "delta", metric, section_tvals.get(metric, []))

    lines.append("\nPaired t-tests [raw]: Assisted vs Autonomous by model")
    lines.append("(paired by project, tested on raw generated metric values)")
    lines.append("Holm correction: per metric, across LLMs in this section")
    section_tvals = {}
    for metric in METRICS:
        lines.append(f"\nMetric: {metric}")
        all_llms = sorted(
            set(k_llm for k_llm, k_metric, _ in raw_paired_lookup.keys() if k_metric == metric)
        )
        rows = []
        metric_tvals = []
        for llm in all_llms:
            assisted = []
            autonomous = []
            for (k_llm, k_metric, _project), mode_map in raw_paired_lookup.items():
                if k_llm != llm or k_metric != metric:
                    continue
                if "assisted" in mode_map and "autonomous" in mode_map:
                    assisted.append(mode_map["assisted"])
                    autonomous.append(mode_map["autonomous"])

            n_pairs = len(assisted)
            if n_pairs == 0:
                continue

            row = {"llm": llm, "n": n_pairs}
            if n_pairs < 2:
                row["status"] = "insufficient_pairs"
                rows.append(row)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat, p_val = stats.ttest_rel(np.array(assisted), np.array(autonomous), nan_policy="omit")

            row["t_stat"] = float(t_stat)
            if np.isnan(t_stat) or np.isnan(p_val):
                row["status"] = "insufficient_variance"
                rows.append(row)
            else:
                diff = np.array(assisted) - np.array(autonomous)
                mean_diff = float(np.mean(diff))
                cohen_d = paired_cohens_d(diff)
                ci_low, ci_high = paired_ci_mean(diff)
                row["p_raw"] = float(p_val)
                row["mean"] = mean_diff
                row["cohen_d"] = cohen_d
                row["ci_low"] = ci_low
                row["ci_high"] = ci_high
                rows.append(row)

        p_raw = [r["p_raw"] for r in rows if "p_raw" in r]
        p_adj = holm_adjust_pvalues(p_raw)
        p_adj_iter = iter(p_adj)

        for r in rows:
            llm = r["llm"]
            n_pairs = r["n"]
            if "status" in r:
                if r["status"] == "insufficient_pairs":
                    lines.append(f"{llm}: n={n_pairs} | insufficient pairs for t-test")
                elif r["status"] == "insufficient_variance":
                    lines.append(f"{llm}: n={n_pairs} | insufficient variance for t-test")
                continue

            p_adj_val = next(p_adj_iter)
            lines.append(
                f"{llm}: n={n_pairs} | t={r['t_stat']:.5f} | p_ttest={r['p_raw']:.6f} | "
                f"p_ttest_holm={p_adj_val:.6f} | mean(assisted-autonomous)={r['mean']:.4f} | "
                f"cohen_d={r['cohen_d']:.4f} | 95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            )
            metric_tvals.append({"llm": llm, "n": n_pairs, "t": r["t_stat"], "p_adj": p_adj_val})

        if len(rows) == 0:
            lines.append("No paired data.")
        else:
            section_tvals[metric] = metric_tvals
    for metric in METRICS:
        save_tvalue_plot("Assisted vs Autonomous", "raw", metric, section_tvals.get(metric, []))

    lines.append("\nPaired t-tests [raw]: Original vs Assisted by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: per metric, across LLMs in this section")
    section_tvals = {}
    for metric in METRICS:
        lines.append(f"\nMetric: {metric}")
        rows = []
        metric_tvals = []
        for llm in sorted(paired_vs_original["assisted"][metric].keys()):
            pairs = paired_vs_original["assisted"][metric][llm]
            generated = np.array([g for g, _ in pairs], dtype=float)
            original = np.array([o for _, o in pairs], dtype=float)
            n_pairs = len(generated)
            if n_pairs == 0:
                continue

            row = {"llm": llm, "n": n_pairs}
            if n_pairs < 2:
                row["status"] = "insufficient_pairs"
                rows.append(row)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat, p_val = stats.ttest_rel(generated, original, nan_policy="omit")

            row["t_stat"] = float(t_stat)
            if np.isnan(t_stat) or np.isnan(p_val):
                row["status"] = "insufficient_variance"
                rows.append(row)
            else:
                diff = generated - original
                mean_diff = float(np.mean(diff))
                cohen_d = paired_cohens_d(diff)
                ci_low, ci_high = paired_ci_mean(diff)
                row["p_raw"] = float(p_val)
                row["mean"] = mean_diff
                row["cohen_d"] = cohen_d
                row["ci_low"] = ci_low
                row["ci_high"] = ci_high
                rows.append(row)

        p_raw = [r["p_raw"] for r in rows if "p_raw" in r]
        p_adj = holm_adjust_pvalues(p_raw)
        p_adj_iter = iter(p_adj)

        for r in rows:
            llm = r["llm"]
            n_pairs = r["n"]
            if "status" in r:
                if r["status"] == "insufficient_pairs":
                    lines.append(f"{llm}: n={n_pairs} | insufficient pairs for t-test")
                elif r["status"] == "insufficient_variance":
                    lines.append(f"{llm}: n={n_pairs} | insufficient variance for t-test")
                continue

            p_adj_val = next(p_adj_iter)
            lines.append(
                f"{llm}: n={n_pairs} | t={r['t_stat']:.5f} | p_ttest={r['p_raw']:.6f} | "
                f"p_ttest_holm={p_adj_val:.6f} | mean(generated-original)={r['mean']:.4f} | "
                f"cohen_d={r['cohen_d']:.4f} | 95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            )
            metric_tvals.append({"llm": llm, "n": n_pairs, "t": r["t_stat"], "p_adj": p_adj_val})

        if len(rows) == 0:
            lines.append("No paired data.")
        else:
            section_tvals[metric] = metric_tvals
    for metric in METRICS:
        save_tvalue_plot("Original vs Assisted", "raw", metric, section_tvals.get(metric, []))

    lines.append("\nPaired t-tests [raw]: Original vs Autonomous by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: per metric, across LLMs in this section")
    section_tvals = {}
    for metric in METRICS:
        lines.append(f"\nMetric: {metric}")
        rows = []
        metric_tvals = []
        for llm in sorted(paired_vs_original["autonomous"][metric].keys()):
            pairs = paired_vs_original["autonomous"][metric][llm]
            generated = np.array([g for g, _ in pairs], dtype=float)
            original = np.array([o for _, o in pairs], dtype=float)
            n_pairs = len(generated)
            if n_pairs == 0:
                continue

            row = {"llm": llm, "n": n_pairs}
            if n_pairs < 2:
                row["status"] = "insufficient_pairs"
                rows.append(row)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat, p_val = stats.ttest_rel(generated, original, nan_policy="omit")

            row["t_stat"] = float(t_stat)
            if np.isnan(t_stat) or np.isnan(p_val):
                row["status"] = "insufficient_variance"
                rows.append(row)
            else:
                diff = generated - original
                mean_diff = float(np.mean(diff))
                cohen_d = paired_cohens_d(diff)
                ci_low, ci_high = paired_ci_mean(diff)
                row["p_raw"] = float(p_val)
                row["mean"] = mean_diff
                row["cohen_d"] = cohen_d
                row["ci_low"] = ci_low
                row["ci_high"] = ci_high
                rows.append(row)

        p_raw = [r["p_raw"] for r in rows if "p_raw" in r]
        p_adj = holm_adjust_pvalues(p_raw)
        p_adj_iter = iter(p_adj)

        for r in rows:
            llm = r["llm"]
            n_pairs = r["n"]
            if "status" in r:
                if r["status"] == "insufficient_pairs":
                    lines.append(f"{llm}: n={n_pairs} | insufficient pairs for t-test")
                elif r["status"] == "insufficient_variance":
                    lines.append(f"{llm}: n={n_pairs} | insufficient variance for t-test")
                continue

            p_adj_val = next(p_adj_iter)
            lines.append(
                f"{llm}: n={n_pairs} | t={r['t_stat']:.5f} | p_ttest={r['p_raw']:.6f} | "
                f"p_ttest_holm={p_adj_val:.6f} | mean(generated-original)={r['mean']:.4f} | "
                f"cohen_d={r['cohen_d']:.4f} | 95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            )
            metric_tvals.append({"llm": llm, "n": n_pairs, "t": r["t_stat"], "p_adj": p_adj_val})

        if len(rows) == 0:
            lines.append("No paired data.")
        else:
            section_tvals[metric] = metric_tvals
    for metric in METRICS:
        save_tvalue_plot("Original vs Autonomous", "raw", metric, section_tvals.get(metric, []))

    if ttest_tvalue_plot_paths:
        lines.append("\nT-value Plots")
        for variant, p in ttest_tvalue_plot_paths:
            lines.append(f"{variant}: {p}")

    lines.append("\nANOVA by mode per metric (factor: model) [delta]")
    lines.append("(one-way ANOVA + Tukey HSD post-hoc)")
    for mode in MODES:
        lines.append(f"\nMode: {mode}")
        for metric in METRICS:
            by_llm = deltas_by_mode_metric_llm[mode][metric]
            llms = sorted(by_llm.keys())
            groups = [np.array(by_llm[llm], dtype=float) for llm in llms if len(by_llm[llm]) > 0]
            valid_llms = [llm for llm in llms if len(by_llm[llm]) > 0]

            if len(groups) < 2:
                lines.append(f"{metric}: insufficient data for ANOVA")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_stat, p_val = stats.f_oneway(*groups)

            if np.isnan(f_stat) or np.isnan(p_val):
                lines.append(f"{metric}: ANOVA undefined (constant/insufficient variance)")
            else:
                lines.append(f"{metric}: F={f_stat:.5f} p_anova={p_val:.6f}")

            values = []
            labels = []
            for llm in valid_llms:
                for v in by_llm[llm]:
                    values.append(v)
                    labels.append(llm)

            if len(set(labels)) < 2:
                lines.append(f"{metric} Tukey HSD: insufficient grouped data")
                continue

            tukey = pairwise_tukeyhsd(endog=np.array(values, dtype=float), groups=np.array(labels), alpha=0.05)
            lines.append(f"{metric} Tukey HSD:")
            for row in tukey.summary().as_text().splitlines():
                lines.append(row)

    lines.append("\nANOVA by mode per metric (factor: model) [raw]")
    lines.append("(one-way ANOVA + Tukey HSD post-hoc)")
    for mode in MODES:
        lines.append(f"\nMode: {mode}")
        for metric in METRICS:
            by_llm = raw_by_mode_metric_llm[mode][metric]
            llms = sorted(by_llm.keys())
            groups = [np.array(by_llm[llm], dtype=float) for llm in llms if len(by_llm[llm]) > 0]
            valid_llms = [llm for llm in llms if len(by_llm[llm]) > 0]

            if len(groups) < 2:
                lines.append(f"{metric}: insufficient data for ANOVA")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_stat, p_val = stats.f_oneway(*groups)

            if np.isnan(f_stat) or np.isnan(p_val):
                lines.append(f"{metric}: ANOVA undefined (constant/insufficient variance)")
            else:
                lines.append(f"{metric}: F={f_stat:.5f} p_anova={p_val:.6f}")

            values = []
            labels = []
            for llm in valid_llms:
                for v in by_llm[llm]:
                    values.append(v)
                    labels.append(llm)

            if len(set(labels)) < 2:
                lines.append(f"{metric} Tukey HSD: insufficient grouped data")
                continue

            tukey = pairwise_tukeyhsd(endog=np.array(values, dtype=float), groups=np.array(labels), alpha=0.05)
            lines.append(f"{metric} Tukey HSD:")
            for row in tukey.summary().as_text().splitlines():
                lines.append(row)

    lines.append(f"\nFailed runs: {len(failed)}")
    if failed:
        lines.append("Failed list:")
        for item in failed:
            lines.append(item)

    analysis_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis: {analysis_output}")


if __name__ == "__main__":
    main()
