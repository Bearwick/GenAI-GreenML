#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Set, Tuple


DEFAULT_REPOS_DIR = "repos"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUTPUT_PREFIX = "failed_generated_code_iteration_"
GENERATED_PREFIX = "GENAIGREENML"
GENERATED_SUFFIX = ".py"

DEFAULT_IGNORE_DIR_NAMES = {
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
}

DELETE_SCAN_SKIP_DIR_NAMES = DEFAULT_IGNORE_DIR_NAMES | {"venv", ".venv"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Copy projects with failed generated scripts into "
            "failed_generated_code_iteration_N at repo root."
        )
    )
    p.add_argument(
        "--results-file",
        default=None,
        help="Optional results CSV path or filename in results/ (default: newest results_*.csv).",
    )
    p.add_argument(
        "--repos-dir",
        default=DEFAULT_REPOS_DIR,
        help=f"Projects directory (default: {DEFAULT_REPOS_DIR}).",
    )
    p.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR}).",
    )
    p.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Root output prefix (default: {DEFAULT_OUTPUT_PREFIX}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    return p.parse_args()


def latest_results_file(results_dir: Path) -> Path:
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")
    files = [
        p
        for p in results_dir.iterdir()
        if p.is_file() and p.name.startswith("results_") and p.suffix.lower() == ".csv"
    ]
    if not files:
        raise SystemExit(f"No results_*.csv files found in: {results_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def resolve_results_file(results_file: str | None, results_dir: Path) -> Path:
    if not results_file:
        return latest_results_file(results_dir)

    candidate = Path(results_file)
    if not candidate.is_absolute():
        candidate = results_dir / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise SystemExit(f"Results file not found: {candidate}")
    if candidate.suffix.lower() != ".csv":
        raise SystemExit(f"Results file must be a .csv file: {candidate}")
    return candidate


def next_output_dir(root_dir: Path, prefix: str) -> Path:
    i = 1
    while True:
        candidate = root_dir / f"{prefix}{i}"
        if not candidate.exists():
            return candidate
        i += 1


def is_generated_script(filename: str) -> bool:
    return filename.startswith(GENERATED_PREFIX) and filename.endswith(GENERATED_SUFFIX)


def collect_last_status_per_script(results_csv: Path) -> Dict[Tuple[str, str], str]:
    last_status: Dict[Tuple[str, str], str] = {}
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            project = (row.get("project") or "").strip()
            script = (row.get("script") or "").strip()
            status = (row.get("status") or "").strip().upper()
            if not project or not script:
                continue
            if not is_generated_script(script):
                continue
            last_status[(project, script)] = status
    return last_status


def project_copy_ignore_factory(
    failed_scripts: Set[str],
) -> Callable[[str, list[str]], Set[str]]:
    def _ignore(directory: str, names: list[str]) -> Set[str]:
        ignored: Set[str] = set()
        for name in names:
            path = Path(directory) / name
            if path.is_dir() and name in DEFAULT_IGNORE_DIR_NAMES:
                ignored.add(name)
                continue

            if path.is_file() and is_generated_script(name):
                if name not in failed_scripts:
                    ignored.add(name)
        return ignored

    return _ignore


def find_script_paths_for_delete(project_dir: Path, script_name: str) -> Set[Path]:
    """
    Results rows store only script filename (not relative path), so we find matches
    inside the project while skipping heavy/irrelevant directories.
    """
    matches: Set[Path] = set()

    direct = project_dir / script_name
    if direct.is_file():
        matches.add(direct)
        return matches

    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in DELETE_SCAN_SKIP_DIR_NAMES and not d.startswith(".")]
        if script_name in files:
            matches.add(Path(root) / script_name)

    return matches


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    repos_dir = (root_dir / args.repos_dir).resolve()
    results_dir = (root_dir / args.results_dir).resolve()
    results_csv = resolve_results_file(args.results_file, results_dir)

    if not repos_dir.is_dir():
        raise SystemExit(f"Repos directory not found: {repos_dir}")

    last_status = collect_last_status_per_script(results_csv)
    failed_by_project: Dict[str, Set[str]] = {}

    for (project, script), status in last_status.items():
        if status != "OK":
            failed_by_project.setdefault(project, set()).add(script)

    if not failed_by_project:
        print(f"No failed generated scripts found in: {results_csv}")
        return

    output_dir = next_output_dir(root_dir, args.output_prefix)

    projects_copied = 0
    projects_missing = 0
    total_failed_scripts = sum(len(v) for v in failed_by_project.values())
    deleted_files = 0
    missing_failed_files = 0

    print(f"Results file: {results_csv}")
    print(f"Failed projects: {len(failed_by_project)}")
    print(f"Failed generated scripts: {total_failed_scripts}")
    print(f"Output folder: {output_dir}")
    if args.dry_run:
        print("Mode: dry-run (no files written)")
        for project, failed_scripts in sorted(failed_by_project.items()):
            src = repos_dir / project
            if not src.is_dir():
                print(f"[missing] {project} (not found in {repos_dir})")
                continue
            for script_name in sorted(failed_scripts):
                paths = find_script_paths_for_delete(src, script_name)
                if not paths:
                    print(f"[dry-run delete missing] {project}/{script_name}")
                for p in sorted(paths):
                    print(f"[dry-run delete] {p}")
        return

    output_dir.mkdir(parents=True, exist_ok=False)

    for project, failed_scripts in sorted(failed_by_project.items()):
        src = repos_dir / project
        if not src.is_dir():
            projects_missing += 1
            print(f"[missing] {project} (not found in {repos_dir})")
            continue

        dst = output_dir / project
        shutil.copytree(
            src,
            dst,
            ignore=project_copy_ignore_factory(failed_scripts),
        )
        projects_copied += 1

        for script_name in sorted(failed_scripts):
            paths = find_script_paths_for_delete(src, script_name)
            if not paths:
                missing_failed_files += 1
                print(f"[delete missing] {project}/{script_name}")
                continue
            for p in sorted(paths):
                p.unlink(missing_ok=True)
                deleted_files += 1

    print(f"Copied projects: {projects_copied}")
    print(f"Missing projects: {projects_missing}")
    print(f"Deleted failed scripts from repos: {deleted_files}")
    print(f"Failed scripts not found for delete: {missing_failed_files}")


if __name__ == "__main__":
    main()
