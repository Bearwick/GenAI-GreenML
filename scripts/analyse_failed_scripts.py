#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple


GENERATED_PREFIX = "GENAIGREENML"
GENERATED_SUFFIX = ".py"
ITERATION_PREFIX = "failed_generated_code_iteration_"
ANALYSIS_FILENAME = "analysis"

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
}

TRACEBACK_ERR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\.]*)(?::|\s*$)")


@dataclass
class ScriptFailure:
    project: str
    script: str
    mode: str
    llm: str
    error_type: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run generated scripts inside failed_generated_code_iteration_* folders "
            "and write per-iteration failure analyses."
        )
    )
    p.add_argument(
        "--root-dir",
        default=None,
        help="Repo root containing failed_generated_code_iteration_* (default: project root).",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="Timeout in seconds per script run (default: 300).",
    )
    p.add_argument(
        "--iteration",
        default=None,
        help="Analyze only one iteration folder name (e.g., failed_generated_code_iteration_1).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


def is_generated_script(name: str) -> bool:
    return name.startswith(GENERATED_PREFIX) and name.endswith(GENERATED_SUFFIX)


def detect_mode(script_name: str) -> str:
    n = script_name.lower()
    if "assisted" in n:
        return "assisted"
    if "autonomous" in n:
        return "autonomous"
    if "original" in n:
        return "original"
    return "unknown"


def detect_llm(script_name: str) -> str:
    stem = Path(script_name).stem
    if "_" not in stem:
        return "unknown"
    return stem.rsplit("_", 1)[-1].lower()


def extract_error_type(output: str, returncode: int | None, timed_out: bool) -> str:
    if timed_out:
        return "TimeoutExpired"

    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    for line in reversed(lines):
        m = TRACEBACK_ERR_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        # Avoid classifying file paths or plain words as exceptions.
        if "." in name or name.endswith("Error") or name.endswith("Exception"):
            return name

    if returncode is None:
        return "UnknownError"
    return f"NonZeroExit({returncode})"


def iter_generated_scripts(project_dir: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES and not d.startswith(".")]
        for f in files:
            if is_generated_script(f):
                yield Path(root) / f


def select_python(project_dir: Path) -> Path:
    venv_python = project_dir / "venv" / "bin" / "python"
    if venv_python.is_file():
        return venv_python
    return Path(sys.executable)


def run_script(project_dir: Path, script_path: Path, timeout_s: float) -> Tuple[bool, str]:
    py = select_python(project_dir)
    timed_out = False
    rc: int | None = None
    out = ""

    try:
        proc = subprocess.run(
            [str(py), str(script_path)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        rc = proc.returncode
        out = f"{proc.stdout}\n{proc.stderr}"
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out = f"{e.stdout or ''}\n{e.stderr or ''}"
    except Exception as e:
        return False, type(e).__name__

    if rc == 0 and not timed_out:
        return True, ""

    return False, extract_error_type(out, rc, timed_out)


def find_iteration_dirs(root_dir: Path, only_name: str | None) -> List[Path]:
    if only_name:
        p = root_dir / only_name
        if not p.is_dir():
            raise SystemExit(f"Iteration folder not found: {p}")
        return [p]

    dirs = [
        p
        for p in root_dir.iterdir()
        if p.is_dir() and p.name.startswith(ITERATION_PREFIX)
    ]
    return sorted(dirs, key=lambda p: p.name)


def format_analysis(
    folder_name: str,
    failures: List[ScriptFailure],
) -> str:
    total_errors = len(failures)

    by_mode_count: Counter[str] = Counter()
    by_mode_types: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    by_llm_count: Counter[str] = Counter()
    by_llm_types: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    by_error_projects: DefaultDict[str, List[str]] = defaultdict(list)

    for f in failures:
        by_mode_count[f.mode] += 1
        by_mode_types[f.mode][f.error_type] += 1
        by_llm_count[f.llm] += 1
        by_llm_types[f.llm][f.error_type] += 1
        by_error_projects[f.error_type].append(f"{f.project}:{f.script}")

    def fmt_counter(counter: Counter[str]) -> str:
        if not counter:
            return "none"
        return ", ".join(f"{k}={v}" for k, v in sorted(counter.items(), key=lambda x: (-x[1], x[0])))

    lines: List[str] = []
    lines.append(f"Iteration Folder: {folder_name}")
    lines.append(f"Total Errors occured: {total_errors}")
    lines.append(f"Total Errors by assisted: {by_mode_count.get('assisted', 0)}")
    lines.append(f"Total Errors by original: {by_mode_count.get('original', 0)}")
    lines.append(f"Total Errors by autonomous: {by_mode_count.get('autonomous', 0)}")
    lines.append(f"Type of Error by original: {fmt_counter(by_mode_types.get('original', Counter()))}")
    lines.append(f"Type of Error by assisted: {fmt_counter(by_mode_types.get('assisted', Counter()))}")
    lines.append(f"Type of Error by autonomous: {fmt_counter(by_mode_types.get('autonomous', Counter()))}")
    lines.append(f"Total Errors per LLM: {fmt_counter(by_llm_count)}")

    llm_type_parts: List[str] = []
    for llm in sorted(by_llm_types.keys()):
        llm_type_parts.append(f"{llm} -> {fmt_counter(by_llm_types[llm])}")
    lines.append(
        "Type of Error per LLM: "
        + (" | ".join(llm_type_parts) if llm_type_parts else "none")
    )

    lines.append("Error type with list of projects:script")
    if not by_error_projects:
        lines.append("none")
    else:
        for err in sorted(by_error_projects.keys(), key=lambda k: (-len(by_error_projects[k]), k)):
            entries = sorted(by_error_projects[err])
            lines.append(f"{err}: {', '.join(entries)}")

    lines.append("")
    return "\n".join(lines)


def analyze_iteration(iteration_dir: Path, timeout_s: float) -> Tuple[int, int]:
    failures: List[ScriptFailure] = []
    total_scripts = 0

    project_dirs = [p for p in iteration_dir.iterdir() if p.is_dir()]
    for project_dir in sorted(project_dirs, key=lambda p: p.name):
        scripts = sorted(iter_generated_scripts(project_dir), key=lambda p: p.name)
        for script_path in scripts:
            total_scripts += 1
            rel_script = script_path.relative_to(project_dir).as_posix()
            logging.info("▶ Running %s / %s", project_dir.name, rel_script)
            ok, error_type = run_script(project_dir, script_path, timeout_s)
            if ok:
                continue

            script_name = script_path.name
            failures.append(
                ScriptFailure(
                    project=project_dir.name,
                    script=script_name,
                    mode=detect_mode(script_name),
                    llm=detect_llm(script_name),
                    error_type=error_type,
                )
            )

    analysis_path = iteration_dir / ANALYSIS_FILENAME
    analysis_path.write_text(
        format_analysis(iteration_dir.name, failures),
        encoding="utf-8",
    )
    return total_scripts, len(failures)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    root_dir = (
        Path(args.root_dir).resolve()
        if args.root_dir
        else Path(__file__).resolve().parent.parent
    )

    iteration_dirs = find_iteration_dirs(root_dir, args.iteration)
    if not iteration_dirs:
        raise SystemExit(
            f"No folders found matching {ITERATION_PREFIX}* in {root_dir}"
        )

    grand_total_scripts = 0
    grand_total_failures = 0
    for iteration_dir in iteration_dirs:
        total_scripts, total_failures = analyze_iteration(iteration_dir, args.timeout_s)
        grand_total_scripts += total_scripts
        grand_total_failures += total_failures
        print(
            f"{iteration_dir.name}: analyzed {total_scripts} scripts, "
            f"errors={total_failures}, wrote {ANALYSIS_FILENAME}"
        )

    print(
        f"Completed. Iterations={len(iteration_dirs)}, "
        f"scripts={grand_total_scripts}, errors={grand_total_failures}"
    )


if __name__ == "__main__":
    main()
