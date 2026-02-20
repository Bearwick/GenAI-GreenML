#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


RESULTS_DIR = Path("results")


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


def main():
    args = parse_args()
    latest = resolve_results_file(args.results_file)
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

    # Summary counts should reflect all executed files (OK + failed).
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

    # Build original lookup per project (only original_telemetry) from successful runs.
    original_by_project = {}
    for r in ok_rows:
        script = r.get("script", "")
        project = r.get("project", "")
        if "original_telemetry" in script:
            if project not in original_by_project:
                original_by_project[project] = r

    # Keep only successful rows from projects that have successful original_telemetry
    rows = [r for r in ok_rows if r.get("project") in original_by_project]
    aa_ok_with_comparable_original = sum(
        1
        for r in rows
        if "assisted" in r.get("script", "") or "autonomous" in r.get("script", "")
    )

    print(f"Results file: {latest}")
    print(f"ML Projects: {len(projects_all)}")
    print(f"ML Projects with original OK: {len(original_by_project)}")
    print(f"Distinct code files: {len(scripts_all)}")
    print(f"Distinct code files (OK): {len(scripts_ok)}")
    print(f"Assisted + autonomous code files: {len(assisted_autonomous_all)}")
    print(f"Assisted + autonomous code files (OK): {len(assisted_autonomous_ok)}")
    print(f"A+A OK with comparable original: {aa_ok_with_comparable_original}")

    def init_counts():
        return {"inc": 0, "dec": 0, "eq": 0}

    counts = {
        "assisted": {"accuracy": init_counts(), "exec_time": init_counts(), "energy": init_counts()},
        "autonomous": {"accuracy": init_counts(), "exec_time": init_counts(), "energy": init_counts()},
    }
    counts_by_llm = {}

    for r in rows:
        script = r.get("script", "")
        project = r.get("project", "")
        if "assisted" in script:
            mode = "assisted"
        elif "autonomous" in script:
            mode = "autonomous"
        else:
            continue

        orig = original_by_project.get(project)
        if not orig:
            continue

        acc = rounded(to_float(r.get("accuracy", "")), 2)
        acc0 = rounded(to_float(orig.get("accuracy", "")), 2)
        exec_t = rounded(to_float(r.get("exec_time_s", "")), 1)
        exec_t0 = rounded(to_float(orig.get("exec_time_s", "")), 1)
        energy = rounded(to_float(r.get("energy_j", "")), 1)
        energy0 = rounded(to_float(orig.get("energy_j", "")), 1)

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
        update("exec_time", exec_t, exec_t0)
        update("energy", energy, energy0)

        llm = script.rsplit("_", 1)[-1].replace(".py", "")
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
        update_llm("exec_time", exec_t, exec_t0)
        update_llm("energy", energy, energy0)

    for mode in ("assisted", "autonomous"):
        print(f"\n{mode.capitalize()} vs Original")
        for metric in ("accuracy", "exec_time", "energy"):
            c = counts[mode][metric]
            print(f"{metric}: inc={c['inc']} dec={c['dec']} eq={c['eq']}")

    combined = {"accuracy": init_counts(), "exec_time": init_counts(), "energy": init_counts()}
    for metric in combined:
        for mode in ("assisted", "autonomous"):
            combined[metric]["inc"] += counts[mode][metric]["inc"]
            combined[metric]["dec"] += counts[mode][metric]["dec"]
            combined[metric]["eq"] += counts[mode][metric]["eq"]

    print("\nAssisted + Autonomous (combined)")
    for metric in ("accuracy", "exec_time", "energy"):
        c = combined[metric]
        print(f"{metric}: inc={c['inc']} dec={c['dec']} eq={c['eq']}")

    if counts_by_llm:
        print("\nBy LLM")
        for llm in sorted(counts_by_llm.keys()):
            print(f"{llm}:")
            for mode in ("assisted", "autonomous"):
                print(f"{mode}: ", end="")
                parts = []
                for metric in ("accuracy", "exec_time", "energy"):
                    c = counts_by_llm[llm][mode][metric]
                    parts.append(f"{metric} inc={c['inc']} dec={c['dec']} eq={c['eq']}")
                print(" | ".join(parts))

    print(f"\nFailed runs: {len(failed)}")
    if failed:
        print("Failed list:")
        for item in failed:
            print(item)


if __name__ == "__main__":
    main()
