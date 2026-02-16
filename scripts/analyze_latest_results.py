#!/usr/bin/env python3
import csv
import os
from pathlib import Path


RESULTS_DIR = Path("results")


def latest_results_file() -> Path:
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"Results directory not found: {RESULTS_DIR}")
    files = [p for p in RESULTS_DIR.iterdir() if p.is_file() and p.name.startswith("results_") and p.suffix == ".csv"]
    if not files:
        raise SystemExit("No results_*.csv files found")
    return max(files, key=lambda p: p.stat().st_mtime)


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
    latest = latest_results_file()
    rows = []
    with latest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "").strip().upper() != "OK":
                continue
            rows.append(row)

    # Build original lookup per project (only original_telemetry)
    original_by_project = {}
    for r in rows:
        script = r.get("script", "")
        project = r.get("project", "")
        if "original_telemetry" in script:
            if project not in original_by_project:
                original_by_project[project] = r

    # Keep only rows from projects that have original_telemetry
    rows = [r for r in rows if r.get("project") in original_by_project]

    projects = {r.get("project", "") for r in rows if r.get("project")}

    scripts = [r for r in rows if r.get("project") and r.get("script")]

    print(f"Latest results: {latest}")
    print(f"ML Projects: {len(projects)}")
    print(f"Distinct code files: {len(scripts)}")
    print(f"Assisted + autonomous code files: {len(script)-len(projects)}")

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


if __name__ == "__main__":
    main()
