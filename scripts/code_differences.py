#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPOS_DIR = REPO_ROOT / "repos"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "code_diffs"
ORIGINAL_FILE = "GENAIGREENML_original_telemetry_chatgpt.py"
MODES = ("assisted", "autonomous")
LLMS = ("chatgpt", "gemini")
DEFAULT_IGNORE_DIR_NAMES = {
    ".git",
    ".svn",
    ".hg",
    ".venv",
    "venv",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create patch files comparing each project's original telemetry script "
            "against assisted/autonomous ChatGPT and Gemini outputs."
        )
    )
    parser.add_argument(
        "--repos-dir",
        default=str(DEFAULT_REPOS_DIR),
        help="Directory containing project folders to scan (default: ./repos).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where patch files are written (default: ./results/code_diffs).",
    )
    return parser.parse_args()


def iter_projects(repos_dir: Path) -> list[Path]:
    if not repos_dir.is_dir():
        raise SystemExit(f"Repos directory does not exist: {repos_dir}")
    return sorted(path for path in repos_dir.iterdir() if path.is_dir())


def should_skip_dir(dir_path: Path, ignore_names: set[str]) -> bool:
    return any(part in ignore_names for part in dir_path.parts)


def discover_project_scripts(project_dir: Path, ignore_names: set[str]) -> dict[str, list[Path]]:
    scripts_by_name: dict[str, list[Path]] = defaultdict(list)
    for root, dirs, files in os.walk(project_dir):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in ignore_names and not d.startswith(".")]

        if should_skip_dir(root_path, ignore_names):
            continue

        for file_name in files:
            if file_name.startswith("GENAIGREENML") and file_name.endswith(".py"):
                scripts_by_name[file_name].append(root_path / file_name)

    for matches in scripts_by_name.values():
        matches.sort(key=lambda p: (len(p.relative_to(project_dir).parts), str(p.relative_to(project_dir))))
    return scripts_by_name


def run_diff(original_file: Path, target_file: Path, output_file: Path) -> None:
    try:
        original_arg = str(original_file.relative_to(REPO_ROOT))
    except ValueError:
        original_arg = str(original_file)
    try:
        target_arg = str(target_file.relative_to(REPO_ROOT))
    except ValueError:
        target_arg = str(target_file)

    with output_file.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            ["git", "diff", "--no-index", original_arg, target_arg],
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git diff failed for {original_arg} vs {target_arg}: {result.stderr.strip()}"
        )


def print_skipped_details(
    skipped_original_projects: list[str],
    skipped_target_files: dict[str, list[str]],
) -> None:
    print("Skipped items:")
    if not skipped_original_projects and not skipped_target_files:
        print("  None")
        return

    if skipped_original_projects:
        print("  Projects missing original telemetry:")
        for project in skipped_original_projects:
            print(f"    {project}: {ORIGINAL_FILE}")

    if skipped_target_files:
        print("  Missing comparison files:")
        for project in sorted(skipped_target_files):
            files = ", ".join(sorted(skipped_target_files[project]))
            print(f"    {project}: {files}")


def main() -> None:
    args = parse_args()
    repos_dir = Path(args.repos_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    files_added = 0
    skipped_missing_original = 0
    skipped_missing_target = 0
    skipped_original_projects: list[str] = []
    skipped_target_files: dict[str, list[str]] = defaultdict(list)

    for project_dir in iter_projects(repos_dir):
        scripts_by_name = discover_project_scripts(project_dir, DEFAULT_IGNORE_DIR_NAMES)

        original_matches = scripts_by_name.get(ORIGINAL_FILE, [])
        if not original_matches:
            skipped_missing_original += 1
            skipped_original_projects.append(project_dir.name)
            continue
        original_file = original_matches[0]

        for mode in MODES:
            for llm in LLMS:
                target_name = f"GENAIGREENML_{mode}_{llm}.py"
                target_matches = scripts_by_name.get(target_name, [])
                if not target_matches:
                    skipped_missing_target += 1
                    skipped_target_files[project_dir.name].append(target_name)
                    continue
                target_file = target_matches[0]

                patch_name = f"{project_dir.name}_original_{mode}_{llm}.patch"
                output_file = output_dir / patch_name
                if not output_file.exists():
                    files_added += 1
                run_diff(original_file, target_file, output_file)
                generated += 1

    print(f"Output directory: {output_dir}")
    print(f"Generated patch files: {generated}")
    print(f"Patch files added: {files_added}")
    print(f"Skipped projects missing original telemetry: {skipped_missing_original}")
    print(f"Skipped comparisons missing assisted/autonomous target: {skipped_missing_target}")
    print_skipped_details(skipped_original_projects, skipped_target_files)


if __name__ == "__main__":
    main()
