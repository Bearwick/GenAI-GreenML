#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
import shutil
import time
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

TRACEBACK_ERR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\.]*)(?::\s*(.*))?$")


def _venv_python_candidates(project_dir: Path) -> List[Path]:
    venv_dir = project_dir / "venv"
    return [
        venv_dir / "bin" / "python",
        venv_dir / "Scripts" / "python.exe",
    ]


def _is_pip_usable(venv_python: Path) -> tuple[bool, str]:
    try:
        version_result = subprocess.run(
            [str(venv_python), "-m", "pip", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if version_result.returncode != 0:
            msg = (version_result.stderr or version_result.stdout or "").strip()
            return False, msg or f"pip --version failed with exit code {version_result.returncode}"

        install_help_result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if install_help_result.returncode == 0:
            return True, (version_result.stdout or version_result.stderr or "").strip()

        msg = (install_help_result.stderr or install_help_result.stdout or "").strip()
        return False, msg or f"pip install --help failed with exit code {install_help_result.returncode}"
    except Exception as e:
        return False, repr(e)


@dataclass
class ScriptFailure:
    project: str
    script: str
    mode: str
    llm: str
    error_type: str
    error_cause: str


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


def extract_error_details(
    output: str, returncode: int | None, timed_out: bool
) -> Tuple[str, str]:
    if timed_out:
        return "TimeoutExpired", "Script timed out"

    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    for line in reversed(lines):
        m = TRACEBACK_ERR_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        # Avoid classifying file paths or plain words as exceptions.
        if "." in name or name.endswith("Error") or name.endswith("Exception"):
            cause = (m.group(2) or "").strip()
            if cause:
                return name, cause
            return name, name

    if returncode is None:
        return "UnknownError", "UnknownError"
    return f"NonZeroExit({returncode})", f"Exited with code {returncode}"


def iter_generated_scripts(project_dir: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES and not d.startswith(".")]
        for f in files:
            if is_generated_script(f):
                yield Path(root) / f


def select_python(project_dir: Path) -> Path:
    for candidate in _venv_python_candidates(project_dir):
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate
    return Path(sys.executable)


def ensure_project_venv(project_dir: Path) -> tuple[Path | None, str]:
    venv_dir = project_dir / "venv"

    for candidate in _venv_python_candidates(project_dir):
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            pip_ok, pip_msg = _is_pip_usable(candidate)
            if pip_ok:
                return candidate, ""

            # Try repairing pip in-place before recreating the environment.
            subprocess.run(
                [str(candidate), "-m", "ensurepip", "--upgrade"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            pip_ok, pip_msg = _is_pip_usable(candidate)
            if pip_ok:
                return candidate, "pip repaired via ensurepip"

            logging.warning(
                "existing venv python for %s is not usable (%s). Recreating ...",
                project_dir.name,
                pip_msg,
            )
            try:
                shutil.rmtree(venv_dir)
            except OSError as e:
                return None, f"existing venv is corrupted and could not be removed: {e}"

    candidates = [sys.executable, "python3", "python"]
    last_error = ""

    for py in candidates:
        if not py:
            continue
        try:
            logging.info("[i] Creating venv for %s using %s", project_dir.name, py)
            result = subprocess.run(
                [py, "-m", "venv", str(venv_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                msg = stderr or stdout or f"venv creation failed with exit code {result.returncode}"
                last_error = f"{py}: {msg}"
                continue

            for candidate in _venv_python_candidates(project_dir):
                if candidate.is_file() and os.access(str(candidate), os.X_OK):
                    pip_ok, pip_msg = _is_pip_usable(candidate)
                    if pip_ok:
                        return candidate, "venv auto-created"

                    return None, f"{py}: created venv but pip is not usable ({pip_msg})"

            last_error = f"{py}: venv created but python executable not found"
        except Exception as e:
            last_error = f"{py}: {repr(e)}"

    return None, f"venv creation failed ({last_error})"


def sanitize_requirements_text(text: str) -> str:
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            lines.append(raw)
            continue
        if "==" in line and not line.startswith("-e "):
            pkg = line.split("==", 1)[0].strip()
            lines.append(pkg)
        else:
            lines.append(raw)
    return "\n".join(lines) + "\n"


def ensure_requirements(project_dir: Path, venv_python: Path) -> tuple[bool, str]:
    req = project_dir / "requirements.txt"
    if not req.is_file():
        return True, ""

    logging.debug("[i] Installing requirements for %s", project_dir.name)

    subprocess.run(
        [str(venv_python), "-m", "ensurepip", "--upgrade"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    result = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-r", str(req)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        return True, "requirements installed"

    temp_req_path: Path | None = None
    try:
        original = req.read_text(encoding="utf-8")
        relaxed = sanitize_requirements_text(original)

        fd, temp_req = tempfile.mkstemp(prefix="relaxed_requirements_", suffix=".txt")
        os.close(fd)
        temp_req_path = Path(temp_req)
        temp_req_path.write_text(relaxed, encoding="utf-8")

        retry = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(temp_req_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if retry.returncode == 0:
            return True, "requirements installed (relaxed)"

        err = (retry.stderr or "").strip()
        out = (retry.stdout or "").strip()
        msg = err or out or f"relaxed pip install failed with exit code {retry.returncode}"
        msg = msg.replace("\n", " ")[:500]
        return False, f"requirements install failed after relaxing pins: {msg}"
    except Exception as e:
        err = (result.stderr or "").strip()
        out = (result.stdout or "").strip()
        msg = err or out or repr(e)
        msg = msg.replace("\n", " ")[:500]
        return False, f"requirements install failed: {msg}"
    finally:
        try:
            if temp_req_path is not None:
                temp_req_path.unlink()
        except Exception:
            pass


def run_script(
    venv_python: Path,
    project_dir: Path,
    script_path: Path,
    timeout_s: float,
) -> Tuple[bool, str, str]:
    py = venv_python
    timed_out = False
    rc: int | None = None
    out = ""
    output_path: Path | None = None

    try:
        fd, out_path_str = tempfile.mkstemp(prefix="runlog_", suffix=".txt")
        os.close(fd)
        output_path = Path(out_path_str)

        try:
            with output_path.open("w", encoding="utf-8") as out_f:
                proc = subprocess.Popen(
                    [str(py), str(script_path)],
                    cwd=str(project_dir),
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                )
                try:
                    rc = proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        rc = proc.wait(timeout=10)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        try:
                            rc = proc.wait(timeout=5)
                        except Exception:
                            rc = None
        except Exception as e:
            return False, type(e).__name__, str(e) or type(e).__name__

        try:
            out = output_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            out = ""
    except Exception as e:
        return False, type(e).__name__, str(e) or type(e).__name__
    finally:
        try:
            if output_path is not None and output_path.exists():
                output_path.unlink()
        except Exception:
            pass

    if timed_out:
        return False, "TimeoutExpired", "Script timed out"

    if rc == 0:
        return True, "", ""

    error_type, error_cause = extract_error_details(out, rc, timed_out)
    return False, error_type, error_cause


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
    by_mode_llm_count: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    by_mode_llm_types: DefaultDict[str, DefaultDict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    by_error_cause_projects: DefaultDict[str, DefaultDict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for f in failures:
        by_mode_count[f.mode] += 1
        by_mode_types[f.mode][f.error_type] += 1
        by_llm_count[f.llm] += 1
        by_llm_types[f.llm][f.error_type] += 1
        by_error_projects[f.error_type].append(f"{f.project}:{f.script}")
        by_mode_llm_count[f.mode][f.llm] += 1
        by_mode_llm_types[f.mode][f.llm][f.error_type] += 1
        by_error_cause_projects[f.error_type][f.error_cause].append(f"{f.project}:{f.script}")

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
    lines.append(
        "Total Errors per LLM by original: "
        f"{fmt_counter(by_mode_llm_count.get('original', Counter()))}"
    )
    lines.append(
        "Total Errors per LLM by assisted: "
        f"{fmt_counter(by_mode_llm_count.get('assisted', Counter()))}"
    )
    lines.append(
        "Total Errors per LLM by autonomous: "
        f"{fmt_counter(by_mode_llm_count.get('autonomous', Counter()))}"
    )

    llm_type_parts: List[str] = []
    for llm in sorted(by_llm_types.keys()):
        llm_type_parts.append(f"{llm} -> {fmt_counter(by_llm_types[llm])}")
    lines.append(
        "Type of Error per LLM: "
        + (" | ".join(llm_type_parts) if llm_type_parts else "none")
    )
    for mode in ("original", "assisted", "autonomous"):
        mode_parts: List[str] = []
        for llm in sorted(by_mode_llm_types.get(mode, {}).keys()):
            mode_parts.append(f"{llm} -> {fmt_counter(by_mode_llm_types[mode][llm])}")
        lines.append(
            f"Type of Error per LLM by {mode}: "
            + (" | ".join(mode_parts) if mode_parts else "none")
        )

    lines.append("Error type with list of projects:script")
    if not by_error_projects:
        lines.append("none")
    else:
        for err in sorted(by_error_projects.keys(), key=lambda k: (-len(by_error_projects[k]), k)):
            entries = sorted(by_error_projects[err])
            lines.append(f"{err}: {', '.join(entries)}")

    lines.append("Error type with exact causes and list of projects:script")
    if not by_error_cause_projects:
        lines.append("none")
    else:
        for err in sorted(by_error_cause_projects.keys(), key=lambda k: (-len(by_error_projects[k]), k)):
            lines.append(f"{err}:")
            cause_map = by_error_cause_projects[err]
            sorted_causes = sorted(cause_map.keys(), key=lambda c: (-len(cause_map[c]), c))
            for cause in sorted_causes:
                entries = sorted(cause_map[cause])
                lines.append(f"  {cause}: {', '.join(entries)}")

    lines.append("")
    return "\n".join(lines)


def analyze_iteration(iteration_dir: Path, timeout_s: float) -> Tuple[int, int]:
    failures: List[ScriptFailure] = []
    total_scripts = 0

    project_dirs = [p for p in iteration_dir.iterdir() if p.is_dir()]
    for project_dir in sorted(project_dirs, key=lambda p: p.name):
        venv_python, venv_note = ensure_project_venv(project_dir)
        if venv_python is None:
            req_fails = list(iter_generated_scripts(project_dir))
            if not req_fails:
                continue
            for script_path in req_fails:
                total_scripts += 1
                failures.append(
                    ScriptFailure(
                        project=project_dir.name,
                        script=script_path.name,
                        mode=detect_mode(script_path.name),
                        llm=detect_llm(script_path.name),
                        error_type="EnvironmentError",
                        error_cause=venv_note or "Failed to initialize project venv",
                    )
                )
            continue

        req_ok, req_note = ensure_requirements(project_dir, venv_python)
        if not req_ok:
            logging.warning("requirements installation failed for %s: %s", project_dir.name, req_note)

        scripts = sorted(iter_generated_scripts(project_dir), key=lambda p: p.name)
        for script_path in scripts:
            total_scripts += 1
            rel_script = script_path.relative_to(project_dir).as_posix()
            logging.info("▶ Running %s / %s", project_dir.name, rel_script)
            ok, error_type, error_cause = run_script(
                venv_python=venv_python,
                project_dir=project_dir,
                script_path=script_path,
                timeout_s=timeout_s,
            )
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
                    error_cause=error_cause,
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
