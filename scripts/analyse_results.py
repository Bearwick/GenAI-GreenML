#!/usr/bin/env python3
import argparse
import csv
import warnings
from pathlib import Path


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
        return np, plt, Line2D, stats, pairwise_tukeyhsd
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


def resolve_results_file(results_file: str | None) -> Path:
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


def detect_mode(script: str) -> str | None:
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
    np, plt, Line2D, stats, pairwise_tukeyhsd = import_external_libs()

    args = parse_args()
    latest = resolve_results_file(args.results_file)
    analysis_output = RESULTS_DIR / f"{latest.name}_analysis"

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
    paired_vs_original = {
        mode: {metric: {} for metric in METRICS}
        for mode in MODES
    }
    paired_lookup = {}

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

            key = (llm, metric, project)
            paired_lookup.setdefault(key, {})[mode] = delta

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

    lines.append("\nQuantitative Analysis")
    lines.append("Delta definition: accuracy = generated-original; exec_time/energy = original-generated (positive means improvement)")

    plot_paths = []
    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 12))
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
                ax.axhline(0, color="gray", linestyle="--", linewidth=1)
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

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = RESULTS_DIR / f"{latest.name}_analysis_boxplot_{metric}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(plot_path)

    lines.append("\nBox Plots")
    for p in plot_paths:
        lines.append(str(p))

    lines.append("\nPaired t-tests: Assisted vs Autonomous by model")
    lines.append("(paired by project, tested on delta values)")
    for metric in METRICS:
        lines.append(f"\nMetric: {metric}")
        all_llms = sorted(set(deltas_by_mode_metric_llm["assisted"][metric].keys()) | set(deltas_by_mode_metric_llm["autonomous"][metric].keys()))
        any_row = False
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
            any_row = True
            if n_pairs < 2:
                lines.append(f"{llm}: n={n_pairs} | insufficient pairs for t-test")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat, p_val = stats.ttest_rel(np.array(assisted), np.array(autonomous), nan_policy="omit")

            if np.isnan(t_stat) or np.isnan(p_val):
                lines.append(f"{llm}: n={n_pairs} | insufficient variance for t-test")
            else:
                mean_diff = float(np.mean(np.array(assisted) - np.array(autonomous)))
                lines.append(
                    f"{llm}: n={n_pairs} | t={t_stat:.5f} | p_ttest={p_val:.6f} | mean(assisted-autonomous)={mean_diff:.6f}"
                )
        if not any_row:
            lines.append("No paired data.")

    lines.append("\nPaired t-tests: Original vs Assisted by model")
    lines.append("(paired by project and model)")
    for metric in METRICS:
        lines.append(f"\nMetric: {metric}")
        any_row = False
        for llm in sorted(paired_vs_original["assisted"][metric].keys()):
            pairs = paired_vs_original["assisted"][metric][llm]
            generated = np.array([g for g, _ in pairs], dtype=float)
            original = np.array([o for _, o in pairs], dtype=float)
            n_pairs = len(generated)
            if n_pairs == 0:
                continue
            any_row = True
            if n_pairs < 2:
                lines.append(f"{llm}: n={n_pairs} | insufficient pairs for t-test")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat, p_val = stats.ttest_rel(generated, original, nan_policy="omit")

            if np.isnan(t_stat) or np.isnan(p_val):
                lines.append(f"{llm}: n={n_pairs} | insufficient variance for t-test")
                continue

            if metric == "accuracy":
                mean_delta = float(np.mean(generated - original))
                lines.append(
                    f"{llm}: n={n_pairs} | t={t_stat:.5f} | p_ttest={p_val:.6f} | mean(generated-original)={mean_delta:.6f}"
                )
            else:
                mean_delta = float(np.mean(original - generated))
                lines.append(
                    f"{llm}: n={n_pairs} | t={t_stat:.5f} | p_ttest={p_val:.6f} | mean(original-generated)={mean_delta:.6f}"
                )
        if not any_row:
            lines.append("No paired data.")

    lines.append("\nPaired t-tests: Original vs Autonomous by model")
    lines.append("(paired by project and model)")
    for metric in METRICS:
        lines.append(f"\nMetric: {metric}")
        any_row = False
        for llm in sorted(paired_vs_original["autonomous"][metric].keys()):
            pairs = paired_vs_original["autonomous"][metric][llm]
            generated = np.array([g for g, _ in pairs], dtype=float)
            original = np.array([o for _, o in pairs], dtype=float)
            n_pairs = len(generated)
            if n_pairs == 0:
                continue
            any_row = True
            if n_pairs < 2:
                lines.append(f"{llm}: n={n_pairs} | insufficient pairs for t-test")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_stat, p_val = stats.ttest_rel(generated, original, nan_policy="omit")

            if np.isnan(t_stat) or np.isnan(p_val):
                lines.append(f"{llm}: n={n_pairs} | insufficient variance for t-test")
                continue

            if metric == "accuracy":
                mean_delta = float(np.mean(generated - original))
                lines.append(
                    f"{llm}: n={n_pairs} | t={t_stat:.5f} | p_ttest={p_val:.6f} | mean(generated-original)={mean_delta:.6f}"
                )
            else:
                mean_delta = float(np.mean(original - generated))
                lines.append(
                    f"{llm}: n={n_pairs} | t={t_stat:.5f} | p_ttest={p_val:.6f} | mean(original-generated)={mean_delta:.6f}"
                )
        if not any_row:
            lines.append("No paired data.")

    lines.append("\nANOVA by mode per metric (factor: model)")
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

    lines.append(f"\nFailed runs: {len(failed)}")
    if failed:
        lines.append("Failed list:")
        for item in failed:
            lines.append(item)

    analysis_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis: {analysis_output}")


if __name__ == "__main__":
    main()
