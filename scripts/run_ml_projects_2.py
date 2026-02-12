#!/usr/bin/env python3
"""
Traverse a repository of ML projects and run generated scripts, logging metrics to CSV.

Applied improvements:
1) argparse + logging + configurable paths/params
2) close file handles properly (no leaked descriptors)
3) run scripts by full path (safer than basename)
4) per-script timeout with terminate/kill; status TIMEOUT
5) RAPL energy on Linux when available; macOS fallback approximation with note
6) improved memory sampling (lower overhead, safer) via psutil when available
7) improved accuracy parsing (last match, cap read size)
9) csv.DictWriter + flush per row
10) script_priority moved out of loop
11) script discovery via rglob with ignore dirs (cross-platform)

Notes:
- This script assumes each project has its own venv at <project>/venv.
- It will attempt to install requirements.txt if present unless --no-requirements is set.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------
# Defaults / constants
# ----------------------------

DEFAULT_RESULTS_DIR = "results"
DEFAULT_REPOS_DIR = "repos"
DEFAULT_MACOS_AVG_POWER_W = 20.0

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

ACCURACY_PATTERNS = [
    re.compile(r"^ACCURACY\s*=\s*(.+?)\s*$"),
    re.compile(r"^Accuracy\s*:\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^ACC\s*=\s*(.+?)\s*$", re.IGNORECASE),
]

# read at most 2MB from output for parsing accuracy to avoid huge logs causing slowdowns
MAX_PARSE_BYTES = 2 * 1024 * 1024


# ----------------------------
# CLI / logging
# ----------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ML project scripts and log metrics to CSV.")
    p.add_argument("--repos-dir", default=DEFAULT_REPOS_DIR, help="Directory containing project folders.")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory to write results CSV.")
    p.add_argument("--macos-power-w", type=float, default=DEFAULT_MACOS_AVG_POWER_W, help="Fallback macOS avg power (W).")
    p.add_argument("--timeout-s", type=float, default=900.0, help="Timeout per script in seconds.")
    p.add_argument("--sample-interval-s", type=float, default=0.25, help="Memory sampling interval (seconds).")
    p.add_argument("--project-regex", default=None, help="Only process projects whose directory name matches this regex.")
    p.add_argument("--max-projects", type=int, default=None, help="Stop after processing N projects.")
    p.add_argument("--no-requirements", action="store_true", help="Do not install requirements.txt per project.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# ----------------------------
# Helpers: traversal + script discovery
# ----------------------------

def should_skip_dir(dir_path: Path, ignore_names: set[str]) -> bool:
    # Skip if any component is an ignored dir name
    return any(part in ignore_names for part in dir_path.parts)


def iter_project_scripts(project_dir: Path, ignore_names: set[str]) -> List[Path]:
    """
    Find scripts named GENAIGREENML*.py anywhere under project_dir, excluding ignored dirs and hidden dirs.
    """
    scripts: List[Path] = []
    for root, dirs, files in os.walk(project_dir):
        root_p = Path(root)

        # prune hidden + ignored dirs in-place (faster)
        dirs[:] = [d for d in dirs if d not in ignore_names and not d.startswith(".")]

        if should_skip_dir(root_p, ignore_names):
            continue

        for name in files:
            if name.startswith("GENAIGREENML") and name.endswith(".py"):
                scripts.append(root_p / name)
    return scripts


def script_priority(path: Path) -> Tuple[int, str]:
    """
    Sort order: original first, then assisted, then autonomous, then others.
    Within same group sort by filename.
    """
    name = path.name.lower()
    if "original" in name:
        return (0, path.name)
    if "assisted" in name:
        return (1, path.name)
    if "autonomous" in name:
        return (2, path.name)
    return (3, path.name)


# ----------------------------
# Requirements (optional)
# ----------------------------

def ensure_requirements(project_dir: Path) -> None:
    req = project_dir / "requirements.txt"
    pip = project_dir / "venv" / "bin" / "pip"
    if req.is_file() and pip.is_file():
        # Keep quiet by default; enable --verbose to see install attempts
        logging.debug("[i] Installing requirements for %s", project_dir.name)
        subprocess.run([str(pip), "install", "-r", str(req)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


# ----------------------------
# Energy measurement
# ----------------------------

def read_energy_linux_uj() -> Optional[int]:
    """
    Sum Intel RAPL energy counters (microjoules).
    Returns None if not available.
    """
    base = Path("/sys/class/powercap")
    if not base.is_dir():
        return None

    total = 0
    found = False
    try:
        for entry in base.iterdir():
            if not entry.name.startswith("intel-rapl:"):
                continue
            path = entry / "energy_uj"
            try:
                val = int(path.read_text(encoding="utf-8").strip() or "0")
                total += val
                found = True
            except (OSError, ValueError):
                continue
    except OSError:
        return None

    return total if found else None


# ----------------------------
# Accuracy parsing
# ----------------------------

def read_accuracy(output_path: Path) -> str:
    """
    Parse accuracy from output logs. Returns last matched value.
    Caps read size to MAX_PARSE_BYTES.
    """
    try:
        data = output_path.read_bytes()
    except OSError:
        return ""

    if len(data) > MAX_PARSE_BYTES:
        data = data[-MAX_PARSE_BYTES:]  # keep the tail; metrics often printed at end

    text = data.decode("utf-8", errors="ignore").splitlines()

    last = ""
    for line in text:
        s = line.strip()
        for pat in ACCURACY_PATTERNS:
            m = pat.match(s)
            if m:
                last = m.group(1).strip()
    return last


# ----------------------------
# Memory tracking
# ----------------------------

def mem_peak_delta_mb_for_process(proc: subprocess.Popen, sample_interval_s: float) -> float:
    """
    Track peak RSS while process runs. Returns (peak - baseline) in MB.
    If psutil isn't available or errors occur, returns 0.0.

    Lower sampling overhead than 50ms polling.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return 0.0

    try:
        p = psutil.Process(proc.pid)
    except Exception:
        return 0.0

    try:
        baseline = p.memory_info().rss
    except Exception:
        baseline = 0

    peak = baseline

    # Sample until process exits
    while proc.poll() is None:
        try:
            rss = p.memory_info().rss
            if rss > peak:
                peak = rss
        except Exception:
            pass
        time.sleep(sample_interval_s)

    # A few extra samples right after exit to catch late RSS reporting
    for _ in range(3):
        try:
            rss = p.memory_info().rss
            if rss > peak:
                peak = rss
        except Exception:
            break
        time.sleep(sample_interval_s)

    delta = max(0, peak - baseline)
    return delta / (1024 ** 2)


# ----------------------------
# Runner
# ----------------------------

@dataclass
class RunResult:
    status: str
    exit_code: Optional[int]
    accuracy: str
    mem_delta_mb: float
    exec_time_s: float
    energy_j: str
    notes: str
    output_log: Path


def run_script(
    venv_python: Path,
    project_dir: Path,
    script_path: Path,
    timeout_s: float,
    sample_interval_s: float,
    macos_avg_power_w: float,
) -> RunResult:
    # Write combined stdout/stderr to a temp file (closed properly)
    fd, out_path_str = tempfile.mkstemp(prefix="runlog_", suffix=".txt")
    os.close(fd)
    output_path = Path(out_path_str)

    energy_before_uj = read_energy_linux_uj()
    start = time.time()

    exit_code: Optional[int] = None
    status = "FAILED"
    notes = ""
    energy_j = ""

    try:
        with output_path.open("w", encoding="utf-8") as out_f:
            proc = subprocess.Popen(
                [str(venv_python), str(script_path)],
                cwd=str(project_dir),
                stdout=out_f,
                stderr=subprocess.STDOUT,
            )

            # Track memory while process runs (poll-based)
            mem_delta_mb = mem_peak_delta_mb_for_process(proc, sample_interval_s)

            try:
                exit_code = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                notes = f"timeout after {timeout_s:.1f}s"
                # try graceful terminate then kill
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    exit_code = proc.wait(timeout=10)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        exit_code = proc.wait(timeout=5)
                    except Exception:
                        exit_code = None

            end = time.time()
            exec_time_s = end - start

    except Exception as e:
        # If Popen/open fails, capture error
        mem_delta_mb = 0.0
        exec_time_s = time.time() - start
        notes = f"runner error: {repr(e)}"
        status = "FAILED"

    # If we didn't compute mem_delta_mb inside the normal flow, ensure it's defined
    if "mem_delta_mb" not in locals():
        mem_delta_mb = 0.0
    if "exec_time_s" not in locals():
        exec_time_s = time.time() - start

    # Status based on exit code if not TIMEOUT
    if status != "TIMEOUT":
        status = "OK" if exit_code == 0 else "FAILED"

    accuracy = read_accuracy(output_path)

    # Energy calculation
    if energy_before_uj is not None:
        energy_after_uj = read_energy_linux_uj()
        if energy_after_uj is not None:
            delta_uj = energy_after_uj - energy_before_uj
            if delta_uj < 0:
                # possible counter wrap or reset
                notes = (notes + " | " if notes else "") + "RAPL counter wrap/reset suspected"
                delta_uj = 0
            energy_j = f"{delta_uj / 1_000_000:.6f}"
        else:
            energy_j = ""
            notes = (notes + " | " if notes else "") + "energy not available (RAPL read failed)"
    else:
        # Keep macOS approximation fallback with explicit note
        energy_j = f"{(exec_time_s * macos_avg_power_w):.3f}"
        notes = (notes + " | " if notes else "") + f"energy approx on macOS using {macos_avg_power_w}W"

    return RunResult(
        status=status,
        exit_code=exit_code,
        accuracy=accuracy,
        mem_delta_mb=mem_delta_mb,
        exec_time_s=exec_time_s,
        energy_j=energy_j,
        notes=notes,
        output_log=output_path,
    )


# ----------------------------
# Main
# ----------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    setup_logging(args.verbose)

    repos_dir = Path(args.repos_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if not repos_dir.is_dir():
        logging.error("[!] repos dir not found: %s", repos_dir)
        return 2

    project_re = re.compile(args.project_regex) if args.project_regex else None
    ignore_names = set(DEFAULT_IGNORE_DIR_NAMES)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"results_{timestamp}.csv"

    fieldnames = [
        "timestamp",
        "project",
        "script",
        "status",
        "exit_code",
        "accuracy",
        "mem_delta_mb",
        "exec_time_s",
        "energy_j",
        "notes",
    ]

    processed_projects = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for project_dir in sorted(repos_dir.iterdir(), key=lambda p: p.name.lower()):
            if not project_dir.is_dir():
                continue
            if project_re and not project_re.search(project_dir.name):
                continue

            scripts = iter_project_scripts(project_dir, ignore_names)
            scripts.sort(key=script_priority)

            if not scripts:
                writer.writerow(
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "project": project_dir.name,
                        "script": "GENAIGREENML*.py",
                        "status": "SKIPPED",
                        "exit_code": "",
                        "accuracy": "",
                        "mem_delta_mb": "",
                        "exec_time_s": "",
                        "energy_j": "",
                        "notes": "",
                    }
                )
                f.flush()
                continue

            if not args.no_requirements:
                ensure_requirements(project_dir)

            venv_python = project_dir / "venv" / "bin" / "python"
            if not (venv_python.is_file() and os.access(str(venv_python), os.X_OK)):
                for script_path in scripts:
                    writer.writerow(
                        {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "project": project_dir.name,
                            "script": script_path.relative_to(project_dir).as_posix(),
                            "status": "FAILED",
                            "exit_code": "",
                            "accuracy": "",
                            "mem_delta_mb": "",
                            "exec_time_s": "",
                            "energy_j": "",
                            "notes": "venv python missing",
                        }
                    )
                    f.flush()
                continue

            for script_path in scripts:
                rel_script = script_path.relative_to(project_dir).as_posix()
                logging.info("▶ Running %s / %s", project_dir.name, rel_script)

                result = run_script(
                    venv_python=venv_python,
                    project_dir=project_dir,
                    script_path=script_path,
                    timeout_s=args.timeout_s,
                    sample_interval_s=args.sample_interval_s,
                    macos_avg_power_w=args.macos_power_w,
                )

                writer.writerow(
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "project": project_dir.name,
                        "script": rel_script,
                        "status": result.status,
                        "exit_code": "" if result.exit_code is None else str(result.exit_code),
                        "accuracy": result.accuracy,
                        "mem_delta_mb": f"{result.mem_delta_mb:.3f}",
                        "exec_time_s": f"{result.exec_time_s:.9f}",
                        "energy_j": result.energy_j,
                        "notes": result.notes,
                    }
                )
                f.flush()

                # Clean up output log
                try:
                    result.output_log.unlink()
                except OSError:
                    pass

            processed_projects += 1
            if args.max_projects and processed_projects >= args.max_projects:
                break

    logging.info("✅ Finished. Results written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())