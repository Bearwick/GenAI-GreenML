#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPOS_DIR = REPO_ROOT / "repos"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "cyclomatic_code_complexity"
GENERATED_PREFIX = "GENAIGREENML"
GENERATED_SUFFIX = ".py"
ORIGINAL_FILE = "GENAIGREENML_original_telemetry_chatgpt.py"
METRIC_KEYS = ("complexity", "ncloc", "cognitive_complexity")
GENERATED_MODES = ("assisted", "autonomous")
ALL_MODES = ("original", "assisted", "autonomous")
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
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 16
PLOT_TITLE_FONTSIZE = 18
MODE_TITLE_FONTSIZE = 18
BOXPLOT_SPACING = 0.65
BOXPLOT_WIDTH = 0.35


@dataclass(frozen=True)
class LocalFileRecord:
    project: str
    relative_path: str
    relative_path_root: str
    file: str
    mode: str
    llm: str
    path: Path


@dataclass(frozen=True)
class ComplexityRecord:
    project: str
    relative_path: str
    file: str
    mode: str
    llm: str
    complexity: float
    ncloc: float
    cognitive_complexity: float
    complexity_per_100_ncloc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SonarQube Cloud analysis on GENAIGREENML files and produce "
            "cyclomatic complexity figures, summaries, and paired tests."
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
        help="Output directory for the analysis artefacts.",
    )
    parser.add_argument(
        "--sonar-host-url",
        default=os.environ.get("SONAR_HOST_URL", "https://sonarcloud.io"),
        help="SonarQube Cloud host URL (default: env SONAR_HOST_URL or https://sonarcloud.io).",
    )
    parser.add_argument(
        "--sonar-organization",
        default=os.environ.get("SONAR_ORGANIZATION"),
        help="SonarQube Cloud organization key (default: env SONAR_ORGANIZATION).",
    )
    parser.add_argument(
        "--sonar-project-key",
        default=os.environ.get("SONAR_PROJECT_KEY"),
        help="SonarQube Cloud project key (default: env SONAR_PROJECT_KEY).",
    )
    parser.add_argument(
        "--sonar-project-name",
        default="GENAIGREENML Cyclomatic Complexity",
        help="Project name used when running the scanner.",
    )
    parser.add_argument(
        "--scanner-bin",
        default=os.environ.get("SONAR_SCANNER_BIN", "sonar-scanner"),
        help="Path to the sonar-scanner executable (default: env SONAR_SCANNER_BIN or sonar-scanner).",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Skip scanner execution and fetch metrics from the existing Sonar project state.",
    )
    parser.add_argument(
        "--reuse-cached-api",
        action="store_true",
        help=(
            "Reuse component_tree_raw.json and project_measures.json from the output directory "
            "instead of running the scanner or calling the SonarQube API."
        ),
    )
    parser.add_argument(
        "--analysis-timeout-s",
        type=float,
        default=1800.0,
        help="Timeout when waiting for SonarQube to finish processing analysis.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=5.0,
        help="Polling interval while waiting for SonarQube analysis completion.",
    )
    return parser.parse_args()


def import_external_libs():
    try:
        import numpy as np  # type: ignore
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
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore
        from scipy import stats  # type: ignore
        from statsmodels.stats.multicomp import pairwise_tukeyhsd  # type: ignore
        from statsmodels.stats.multitest import multipletests  # type: ignore

        return np, plt, Line2D, stats, pairwise_tukeyhsd, multipletests
    except ModuleNotFoundError as exc:
        pkg = str(exc).split("'")[-2] if "'" in str(exc) else str(exc)
        raise SystemExit(
            "Missing dependency for cyclomatic complexity analysis: "
            f"{pkg}. Install required packages with: "
            "python3 -m pip install numpy scipy matplotlib statsmodels"
        )


def normalize_path_key(path_value: str) -> str:
    return path_value.replace("\\", "/").lstrip("./")


def should_skip_dir(dir_path: Path, ignore_names: set[str]) -> bool:
    return any(part in ignore_names for part in dir_path.parts)


def detect_mode(file_name: str) -> str:
    lower = file_name.lower()
    if "original" in lower:
        return "original"
    if "assisted" in lower:
        return "assisted"
    if "autonomous" in lower:
        return "autonomous"
    return "unknown"


def detect_llm(file_name: str) -> str:
    lower = file_name.lower()
    if "original" in lower:
        return "baseline"
    stem = Path(file_name).stem
    if "_" not in stem:
        return "unknown"
    return stem.rsplit("_", 1)[-1].lower()


def display_llm_name(name: str) -> str:
    mapping = {
        "baseline": "Original",
        "chatgpt": "ChatGPT",
        "gemini": "Gemini",
        "claude": "Claude",
        "codex": "Codex",
        "unknown": "Unknown",
    }
    return mapping.get(name, name[:1].upper() + name[1:])


def display_mode_name(name: str) -> str:
    mapping = {
        "original": "Original",
        "assisted": "Assisted",
        "autonomous": "Autonomous",
        "unknown": "Unknown",
    }
    return mapping.get(name, name[:1].upper() + name[1:])


def is_generated_file(name: str) -> bool:
    return name.startswith(GENERATED_PREFIX) and name.endswith(GENERATED_SUFFIX)


def iter_generated_files(repos_dir: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(repos_dir):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES and not d.startswith(".")]
        if should_skip_dir(root_path, IGNORE_DIR_NAMES):
            continue
        for file_name in files:
            if is_generated_file(file_name):
                yield root_path / file_name


def collect_local_metadata(repos_dir: Path) -> tuple[list[LocalFileRecord], dict[str, LocalFileRecord]]:
    if not repos_dir.is_dir():
        raise SystemExit(f"Repos directory does not exist: {repos_dir}")

    records: list[LocalFileRecord] = []
    path_lookup: dict[str, LocalFileRecord] = {}
    basename_counts: dict[str, int] = defaultdict(int)

    for file_path in iter_generated_files(repos_dir):
        rel_to_repos = file_path.relative_to(repos_dir)
        if len(rel_to_repos.parts) < 2:
            continue

        project = rel_to_repos.parts[0]
        record = LocalFileRecord(
            project=project,
            relative_path=normalize_path_key(str(rel_to_repos)),
            relative_path_root=normalize_path_key(str(Path("repos") / rel_to_repos)),
            file=file_path.name,
            mode=detect_mode(file_path.name),
            llm=detect_llm(file_path.name),
            path=file_path,
        )
        records.append(record)
        path_lookup[record.relative_path] = record
        path_lookup[record.relative_path_root] = record
        basename_counts[record.file] += 1

    for record in records:
        if basename_counts[record.file] == 1:
            path_lookup.setdefault(record.file, record)

    return sorted(records, key=lambda r: (r.project, r.relative_path)), path_lookup


def extract_metric(component: dict, metric_key: str) -> float:
    for measure in component.get("measures", []):
        if measure.get("metric") == metric_key:
            try:
                return float(measure.get("value"))
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def find_local_record_for_component(component: dict, path_lookup: dict[str, LocalFileRecord]) -> Optional[LocalFileRecord]:
    candidates: list[str] = []
    for raw in (component.get("path"), component.get("name")):
        if isinstance(raw, str) and raw:
            candidates.append(normalize_path_key(raw))

    component_key = component.get("key")
    if isinstance(component_key, str) and ":" in component_key:
        candidates.append(normalize_path_key(component_key.split(":", 1)[1]))
    elif isinstance(component_key, str) and component_key:
        candidates.append(normalize_path_key(component_key))

    for candidate in candidates:
        if candidate in path_lookup:
            return path_lookup[candidate]
        if candidate.startswith("repos/") and candidate[5:] in path_lookup:
            return path_lookup[candidate[5:]]
        if not candidate.startswith("repos/"):
            repos_candidate = f"repos/{candidate}"
            if repos_candidate in path_lookup:
                return path_lookup[repos_candidate]
        basename = Path(candidate).name
        if basename in path_lookup:
            return path_lookup[basename]
    return None


def require_sonar_settings(args: argparse.Namespace) -> tuple[str, str, str]:
    token = os.environ.get("SONAR_TOKEN")
    if not token:
        raise SystemExit("SONAR_TOKEN must be set in the environment to call the SonarQube Cloud API.")
    if not args.sonar_project_key:
        raise SystemExit("Set --sonar-project-key or SONAR_PROJECT_KEY before running the analysis.")
    if not args.sonar_organization and not args.skip_scan:
        raise SystemExit("Set --sonar-organization or SONAR_ORGANIZATION before running the scanner.")
    return token, args.sonar_project_key, args.sonar_organization or ""


def load_cached_json(path: Path, label: str):
    if not path.is_file():
        raise SystemExit(
            f"Cached {label} file was not found: {path}. "
            "Run the analysis once with SonarQube enabled before using --reuse-cached-api."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Cached {label} file is not valid JSON: {path} ({exc})")


def api_get_json(base_url: str, path: str, token: str, query: Optional[dict[str, object]] = None) -> dict:
    query_string = ""
    if query:
        query_string = "?" + urllib.parse.urlencode(query, doseq=True)
    url = f"{base_url.rstrip('/')}{path}{query_string}"
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"SonarQube API request failed ({exc.code}) for {url}: {body}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"SonarQube API request failed for {url}: {exc}")


def run_sonar_scanner(
    args: argparse.Namespace,
    token: str,
    output_dir: Path,
) -> dict[str, str]:
    scanner_bin = shutil.which(args.scanner_bin) or (Path(args.scanner_bin) if Path(args.scanner_bin).exists() else None)
    if not scanner_bin:
        raise SystemExit(
            "sonar-scanner was not found. Install SonarScanner CLI or set SONAR_SCANNER_BIN/--scanner-bin "
            "to the scanner executable before running this analysis."
        )

    scanner_bin_str = str(scanner_bin)
    working_dir = output_dir / ".scannerwork"
    metadata_file = output_dir / "report-task.txt"
    scanner_log = output_dir / "sonar_scanner.log"

    if working_dir.exists():
        shutil.rmtree(working_dir)
    if metadata_file.exists():
        metadata_file.unlink()

    cmd = [
        scanner_bin_str,
        f"-Dsonar.host.url={args.sonar_host_url}",
        f"-Dsonar.projectKey={args.sonar_project_key}",
        f"-Dsonar.projectName={args.sonar_project_name}",
        f"-Dsonar.organization={args.sonar_organization}",
        f"-Dsonar.token={token}",
        "-Dsonar.sources=repos",
        "-Dsonar.inclusions=**/GENAIGREENML*.py",
        "-Dsonar.exclusions=**/.git/**,**/.venv/**,**/venv/**,**/__pycache__/**,**/.mypy_cache/**,**/.pytest_cache/**,**/.tox/**,**/node_modules/**,**/dist/**,**/build/**,**/.idea/**,**/.vscode/**",
        "-Dsonar.scm.disabled=true",
        "-Dsonar.sourceEncoding=UTF-8",
        "-Dsonar.python.version=3.13",
        f"-Dsonar.working.directory={working_dir}",
        f"-Dsonar.scanner.metadataFilePath={metadata_file}",
    ]

    env = os.environ.copy()
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    scanner_log.write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0:
        raise SystemExit(
            "sonar-scanner failed. See "
            f"{scanner_log} for the full output."
        )
    if not metadata_file.is_file():
        raise SystemExit(
            "sonar-scanner completed but no report-task.txt was written. "
            f"See {scanner_log} for details."
        )
    return parse_report_task(metadata_file)


def parse_report_task(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def wait_for_analysis_completion(
    server_url: str,
    ce_task_id: str,
    token: str,
    timeout_s: float,
    poll_interval_s: float,
    output_dir: Path,
) -> dict:
    deadline = time.time() + timeout_s
    last_response: dict = {}
    while time.time() < deadline:
        last_response = api_get_json(
            server_url,
            "/api/ce/task",
            token,
            {"id": ce_task_id},
        )
        task = last_response.get("task", {})
        status = task.get("status")
        (output_dir / "ce_task_status.json").write_text(
            json.dumps(last_response, indent=2),
            encoding="utf-8",
        )
        if status == "SUCCESS":
            return last_response
        if status in {"FAILED", "CANCELED"}:
            raise SystemExit(f"SonarQube compute engine task ended with status {status}: {json.dumps(last_response)}")
        time.sleep(poll_interval_s)
    raise SystemExit(f"Timed out waiting for SonarQube analysis task {ce_task_id} to complete.")


def fetch_project_measures(server_url: str, project_key: str, token: str) -> dict:
    return api_get_json(
        server_url,
        "/api/measures/component",
        token,
        {"component": project_key, "metricKeys": ",".join(METRIC_KEYS)},
    )


def fetch_file_measure_components(server_url: str, project_key: str, token: str) -> list[dict]:
    all_components: list[dict] = []
    page = 1
    page_size = 500
    while True:
        payload = api_get_json(
            server_url,
            "/api/measures/component_tree",
            token,
            {
                "component": project_key,
                "metricKeys": ",".join(METRIC_KEYS),
                "qualifiers": "FIL",
                "ps": page_size,
                "p": page,
            },
        )
        components = payload.get("components", [])
        all_components.extend(components)
        paging = payload.get("paging", {})
        total = int(paging.get("total", len(all_components)))
        if len(all_components) >= total or not components:
            return all_components
        page += 1


def build_complexity_records(
    components: Sequence[dict],
    path_lookup: dict[str, LocalFileRecord],
) -> tuple[list[ComplexityRecord], list[dict]]:
    records: list[ComplexityRecord] = []
    unmatched_components: list[dict] = []
    for component in components:
        local_record = find_local_record_for_component(component, path_lookup)
        if local_record is None:
            unmatched_components.append(component)
            continue

        complexity = extract_metric(component, "complexity")
        ncloc = extract_metric(component, "ncloc")
        cognitive_complexity = extract_metric(component, "cognitive_complexity")
        complexity_density = float("nan")
        if ncloc and math.isfinite(ncloc) and ncloc > 0 and math.isfinite(complexity):
            complexity_density = (complexity / ncloc) * 100.0

        records.append(
            ComplexityRecord(
                project=local_record.project,
                relative_path=local_record.relative_path,
                file=local_record.file,
                mode=local_record.mode,
                llm=local_record.llm,
                complexity=complexity,
                ncloc=ncloc,
                cognitive_complexity=cognitive_complexity,
                complexity_per_100_ncloc=complexity_density,
            )
        )
    return sorted(records, key=lambda r: (r.project, r.relative_path)), unmatched_components


def find_missing_local_records(
    local_records: Sequence[LocalFileRecord],
    complexity_records: Sequence[ComplexityRecord],
) -> list[LocalFileRecord]:
    matched_paths = {record.relative_path for record in complexity_records}
    return [record for record in local_records if record.relative_path not in matched_paths]


def safe_stdev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])
    idx = (len(sorted_values) - 1) * (p / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[int(idx)])
    weight = idx - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def summarise(values: Sequence[float]) -> dict[str, float]:
    finite = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not finite:
        return {
            "count": 0.0,
            "mean": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
            "std_dev": math.nan,
            "p25": math.nan,
            "p75": math.nan,
        }
    mean_value = sum(finite) / len(finite)
    if len(finite) % 2 == 1:
        median_value = finite[len(finite) // 2]
    else:
        mid = len(finite) // 2
        median_value = (finite[mid - 1] + finite[mid]) / 2.0
    sd = safe_stdev(finite)
    return {
        "count": float(len(finite)),
        "mean": float(mean_value),
        "median": float(median_value),
        "min": float(finite[0]),
        "max": float(finite[-1]),
        "std_dev": float(sd) if sd is not None else math.nan,
        "p25": percentile(finite, 25),
        "p75": percentile(finite, 75),
    }


def format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return "n/a"


def paired_cohens_d(diff, np) -> float:
    arr = np.asarray(diff, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float("nan")
    denom = arr.std(ddof=1)
    if denom == 0:
        return float("nan")
    return float(arr.mean() / denom)


def paired_ci_mean(diff, stats, np, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(diff, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))
    mean_diff = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return (float(mean_diff), float(mean_diff))
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    half_width = t_crit * se
    return float(mean_diff - half_width), float(mean_diff + half_width)


def holm_adjust_pvalues(p_values, multipletests, np) -> list[float]:
    arr = np.asarray(p_values, dtype=float)
    if arr.size == 0:
        return []
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = ~np.isnan(arr)
    if np.any(valid):
        out[valid] = multipletests(arr[valid], alpha=0.05, method="holm")[1]
    return out.tolist()


def save_raw_csv(path: Path, records: Sequence[ComplexityRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "project",
                "relative_path",
                "file",
                "mode",
                "llm",
                "complexity",
                "ncloc",
                "cognitive_complexity",
                "complexity_per_100_ncloc",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.project,
                    record.relative_path,
                    record.file,
                    record.mode,
                    record.llm,
                    format_number(record.complexity),
                    format_number(record.ncloc),
                    format_number(record.cognitive_complexity),
                    format_number(record.complexity_per_100_ncloc),
                ]
            )


def save_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    header = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_paired_csv(path: Path, rows: list[dict[str, str]]) -> None:
    save_summary_csv(path, rows)


def group_values(
    records: Sequence[ComplexityRecord],
    value_getter: Callable[[ComplexityRecord], float],
    group_getter: Callable[[ComplexityRecord], str],
    predicate: Callable[[ComplexityRecord], bool],
) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if not predicate(record):
            continue
        value = value_getter(record)
        if not math.isfinite(value):
            continue
        groups[group_getter(record)].append(float(value))
    return {key: groups[key] for key in sorted(groups)}


def build_summary_rows(records: Sequence[ComplexityRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add_rows(group_type: str, groups: dict[str, list[float]], metric_label: str) -> None:
        for group_name, values in groups.items():
            stats_row = summarise(values)
            rows.append(
                {
                    "group_type": group_type,
                    "metric": metric_label,
                    "group": group_name,
                    "count": format_number(stats_row["count"]),
                    "mean": format_number(stats_row["mean"]),
                    "median": format_number(stats_row["median"]),
                    "min": format_number(stats_row["min"]),
                    "max": format_number(stats_row["max"]),
                    "std_dev": format_number(stats_row["std_dev"]),
                    "p25": format_number(stats_row["p25"]),
                    "p75": format_number(stats_row["p75"]),
                }
            )

    add_rows(
        "mode",
        group_values(
            records,
            lambda r: r.complexity,
            lambda r: r.mode,
            lambda r: r.mode in ALL_MODES,
        ),
        "complexity",
    )
    add_rows(
        "mode",
        group_values(
            records,
            lambda r: r.complexity_per_100_ncloc,
            lambda r: r.mode,
            lambda r: r.mode in ALL_MODES,
        ),
        "complexity_per_100_ncloc",
    )
    add_rows(
        "llm_assisted",
        group_values(
            records,
            lambda r: r.complexity,
            lambda r: r.llm,
            lambda r: r.mode == "assisted",
        ),
        "complexity",
    )
    add_rows(
        "llm_autonomous",
        group_values(
            records,
            lambda r: r.complexity,
            lambda r: r.llm,
            lambda r: r.mode == "autonomous",
        ),
        "complexity",
    )
    add_rows(
        "llm_generated",
        group_values(
            records,
            lambda r: r.complexity,
            lambda r: r.llm,
            lambda r: r.mode in GENERATED_MODES,
        ),
        "complexity",
    )
    add_rows(
        "llm_assisted",
        group_values(
            records,
            lambda r: r.complexity_per_100_ncloc,
            lambda r: r.llm,
            lambda r: r.mode == "assisted",
        ),
        "complexity_per_100_ncloc",
    )
    add_rows(
        "llm_autonomous",
        group_values(
            records,
            lambda r: r.complexity_per_100_ncloc,
            lambda r: r.llm,
            lambda r: r.mode == "autonomous",
        ),
        "complexity_per_100_ncloc",
    )
    add_rows(
        "llm_generated",
        group_values(
            records,
            lambda r: r.complexity_per_100_ncloc,
            lambda r: r.llm,
            lambda r: r.mode in GENERATED_MODES,
        ),
        "complexity_per_100_ncloc",
    )
    return rows


def anova_and_tukey_summary(
    section_label: str,
    groups: dict[str, list[float]],
    stats,
    np,
    pairwise_tukeyhsd,
    lines: list[str],
) -> None:
    lines.append(f"\n{section_label}")
    nonempty = {name: values for name, values in groups.items() if values}
    if len(nonempty) < 2:
        lines.append("Insufficient groups for comparison.")
        return

    ordered_names = sorted(nonempty)
    data = [nonempty[name] for name in ordered_names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_stat, p_value = stats.f_oneway(*data)

    if np.isnan(f_stat) or np.isnan(p_value):
        lines.append(f"ANOVA undefined (constant/insufficient variance) | groups={', '.join(ordered_names)}")
    else:
        lines.append(f"ANOVA: F={f_stat:.5f} | p_anova={p_value:.6f} | groups={', '.join(ordered_names)}")

    values: list[float] = []
    labels: list[str] = []
    for name in ordered_names:
        for value in nonempty[name]:
            if math.isfinite(value):
                values.append(float(value))
                labels.append(name)

    if len(set(labels)) < 2:
        lines.append("Tukey HSD: insufficient grouped data")
        return

    tukey = pairwise_tukeyhsd(endog=np.array(values, dtype=float), groups=np.array(labels), alpha=0.05)
    lines.append("Tukey HSD:")
    for row in tukey.summary().as_text().splitlines():
        lines.append(row)


def create_original_lookup(records: Sequence[ComplexityRecord]) -> dict[str, ComplexityRecord]:
    out: dict[str, ComplexityRecord] = {}
    for record in records:
        if record.mode == "original":
            out[record.project] = record
    return out


def create_mode_llm_lookup(records: Sequence[ComplexityRecord]) -> dict[tuple[str, str, str], ComplexityRecord]:
    return {
        (record.mode, record.llm, record.project): record
        for record in records
        if record.mode in GENERATED_MODES
    }


def build_paired_rows_from_records(records: Sequence[ComplexityRecord]) -> list[dict[str, str]]:
    original_lookup = create_original_lookup(records)
    rows: list[dict[str, str]] = []
    for record in records:
        if record.mode not in GENERATED_MODES:
            continue
        original = original_lookup.get(record.project)
        if original is None:
            continue
        rows.append(
            {
                "comparison": f"original_vs_{record.mode}",
                "project": record.project,
                "llm": record.llm,
                "generated_relative_path": record.relative_path,
                "original_relative_path": original.relative_path,
                "generated_complexity": format_number(record.complexity),
                "original_complexity": format_number(original.complexity),
                "delta_complexity": format_number(record.complexity - original.complexity),
                "generated_complexity_per_100_ncloc": format_number(record.complexity_per_100_ncloc),
                "original_complexity_per_100_ncloc": format_number(original.complexity_per_100_ncloc),
                "delta_complexity_per_100_ncloc": format_number(
                    record.complexity_per_100_ncloc - original.complexity_per_100_ncloc
                ),
            }
        )

    lookup = create_mode_llm_lookup(records)
    projects = sorted({record.project for record in records})
    llms = sorted({record.llm for record in records if record.mode in GENERATED_MODES})
    for llm in llms:
        for project in projects:
            assisted = lookup.get(("assisted", llm, project))
            autonomous = lookup.get(("autonomous", llm, project))
            if assisted is None or autonomous is None:
                continue
            rows.append(
                {
                    "comparison": "assisted_vs_autonomous",
                    "project": project,
                    "llm": llm,
                    "generated_relative_path": assisted.relative_path,
                    "original_relative_path": autonomous.relative_path,
                    "generated_complexity": format_number(assisted.complexity),
                    "original_complexity": format_number(autonomous.complexity),
                    "delta_complexity": format_number(assisted.complexity - autonomous.complexity),
                    "generated_complexity_per_100_ncloc": format_number(assisted.complexity_per_100_ncloc),
                    "original_complexity_per_100_ncloc": format_number(autonomous.complexity_per_100_ncloc),
                    "delta_complexity_per_100_ncloc": format_number(
                        assisted.complexity_per_100_ncloc - autonomous.complexity_per_100_ncloc
                    ),
                }
            )
    return rows


def save_boxplot(
    data: dict[str, list[float]],
    title: str,
    ylabel: str,
    output_path: Path,
    plt,
    np,
    Line2D,
) -> Optional[Path]:
    groups = [(label, values) for label, values in data.items() if values]
    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(groups)), 6))
    positions = np.arange(len(groups), dtype=float) * BOXPLOT_SPACING + 1.0
    ax.boxplot(
        [values for _, values in groups],
        positions=positions,
        widths=BOXPLOT_WIDTH,
        showfliers=False,
        showmeans=False,
        medianprops={"color": "green", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        boxprops={"linewidth": 1.5},
    )
    labels = [label for label, values in groups]
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{label}\n(n={len(values)})" for label, values in groups],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel(ylabel)
    legend_handles = [
        Line2D([0], [0], color="green", linewidth=2, label="Median"),
    ]
    original_group = next((values for label, values in groups if label == "Original"), None)
    if original_group:
        original_median = float(np.median(original_group))
        ax.axhline(original_median, color="blue", linestyle="--", linewidth=1)
        legend_handles.append(
            Line2D([0], [0], color="blue", linestyle="--", linewidth=1, label="Original")
        )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_mode_boxplot(records: Sequence[ComplexityRecord], output_dir: Path, plt, np, Line2D) -> Optional[Path]:
    groups = {
        display_mode_name(mode): [r.complexity for r in records if r.mode == mode and math.isfinite(r.complexity)]
        for mode in ALL_MODES
    }
    return save_boxplot(
        groups,
        "Cyclomatic Complexity by Mode",
        "Cyclomatic Complexity",
        output_dir / "boxplot_complexity_by_mode.png",
        plt,
        np,
        Line2D,
    )


def save_llm_boxplot(
    records: Sequence[ComplexityRecord],
    output_dir: Path,
    mode_filter: Optional[str],
    title: str,
    file_name: str,
    plt,
    np,
    Line2D,
) -> Optional[Path]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.mode == "original":
            continue
        if mode_filter and record.mode != mode_filter:
            continue
        if not math.isfinite(record.complexity):
            continue
        groups[display_llm_name(record.llm)].append(record.complexity)
    return save_boxplot(
        {key: groups[key] for key in sorted(groups)},
        title,
        "Cyclomatic Complexity",
        output_dir / file_name,
        plt,
        np,
        Line2D,
    )


def save_density_boxplot(records: Sequence[ComplexityRecord], output_dir: Path, plt, np, Line2D) -> Optional[Path]:
    groups = {
        display_mode_name(mode): [
            r.complexity_per_100_ncloc
            for r in records
            if r.mode == mode and math.isfinite(r.complexity_per_100_ncloc)
        ]
        for mode in ALL_MODES
    }
    return save_boxplot(
        groups,
        "Cyclomatic Complexity per 100 NCLOC by Mode",
        "Cyclomatic Complexity per 100 NCLOC",
        output_dir / "boxplot_complexity_per_100_ncloc_by_mode.png",
        plt,
        np,
        Line2D,
    )


def save_scatter(records: Sequence[ComplexityRecord], output_dir: Path, plt, Line2D) -> Optional[Path]:
    points = [r for r in records if math.isfinite(r.complexity) and math.isfinite(r.ncloc)]
    if not points:
        return None

    colors = {"original": "#2ca02c", "assisted": "#ff7f0e", "autonomous": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in ALL_MODES:
        subset = [r for r in points if r.mode == mode]
        if not subset:
            continue
        ax.scatter(
            [r.ncloc for r in subset],
            [r.complexity for r in subset],
            alpha=0.8,
            color=colors[mode],
            label=display_mode_name(mode),
            edgecolors="black",
            linewidths=0.4,
        )
    ax.set_xlabel("NCLOC")
    ax.set_ylabel("Cyclomatic Complexity")
    ax.grid(alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label=display_mode_name(mode), markerfacecolor=colors[mode], markeredgecolor="black", markersize=8)
            for mode in ALL_MODES
            if any(r.mode == mode for r in points)
        ]
    )
    fig.tight_layout()
    output_path = output_dir / "scatter_complexity_vs_ncloc_by_mode.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def build_delta_groups(
    records: Sequence[ComplexityRecord],
    value_getter: Callable[[ComplexityRecord], float],
) -> dict[str, dict[str, list[float]]]:
    original_lookup = create_original_lookup(records)
    groups: dict[str, dict[str, list[float]]] = {
        "assisted": defaultdict(list),
        "autonomous": defaultdict(list),
    }
    for record in records:
        if record.mode not in GENERATED_MODES:
            continue
        original = original_lookup.get(record.project)
        if original is None:
            continue
        record_value = value_getter(record)
        original_value = value_getter(original)
        if not math.isfinite(record_value) or not math.isfinite(original_value):
            continue
        groups[record.mode][record.llm].append(record_value - original_value)
    return groups


def save_delta_boxplots(
    delta_groups: dict[str, dict[str, list[float]]],
    title: str,
    ylabel: str,
    output_path: Path,
    plt,
    np,
    Line2D,
) -> Optional[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
    has_data = False
    for ax, mode in zip(axes, GENERATED_MODES):
        llm_groups = [
            (display_llm_name(llm), delta_groups[mode][llm])
            for llm in sorted(delta_groups[mode])
            if delta_groups[mode][llm]
        ]
        if not llm_groups:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            continue
        has_data = True
        positions = np.arange(len(llm_groups), dtype=float) * BOXPLOT_SPACING + 1.0
        ax.boxplot(
            [values for _, values in llm_groups],
            positions=positions,
            widths=BOXPLOT_WIDTH,
            showfliers=False,
            showmeans=False,
            medianprops={"color": "green", "linewidth": 2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            boxprops={"linewidth": 1.5},
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{label}\n(n={len(values)})" for label, values in llm_groups],
            rotation=20,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(
            handles=[
                Line2D([0], [0], color="green", linewidth=2, label="Median"),
                Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="No change (0)"),
            ],
            loc="upper right",
            frameon=True,
        )
    if not has_data:
        plt.close(fig)
        return None
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def paired_ttest_summary(
    label: str,
    llm_values: dict[str, tuple[list[float], list[float]]],
    stats,
    np,
    multipletests,
    lines: list[str],
) -> None:
    lines.append(f"\n{label}")
    rows: list[dict[str, float | str]] = []
    for llm in sorted(llm_values):
        a_values, b_values = llm_values[llm]
        if len(a_values) < 2 or len(b_values) < 2:
            lines.append(f"{display_llm_name(llm)}: insufficient pairs for t-test")
            continue
        a = np.asarray(a_values, dtype=float)
        b = np.asarray(b_values, dtype=float)
        n_pairs = min(a.size, b.size)
        if n_pairs < 2:
            lines.append(f"{display_llm_name(llm)}: insufficient pairs for t-test")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_value = stats.ttest_rel(a, b, nan_policy="omit")

        if np.isnan(t_stat) or np.isnan(p_value):
            lines.append(f"{display_llm_name(llm)}: insufficient variance for t-test")
            continue
        diff = a - b
        diff = diff[~np.isnan(diff)]
        if diff.size < 2:
            lines.append(f"{display_llm_name(llm)}: insufficient pairs for t-test")
            continue
        mean_diff = float(np.mean(diff))
        median_diff = float(np.median(diff))
        cohen_d = paired_cohens_d(diff, np)
        ci_low, ci_high = paired_ci_mean(diff, stats, np)
        rows.append(
            {
                "llm": llm,
                "n": int(diff.size),
                "t": float(t_stat),
                "p": float(p_value),
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "cohen_d": cohen_d,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    adjusted = holm_adjust_pvalues([row["p"] for row in rows], multipletests, np)
    for row, p_adj in zip(rows, adjusted):
        lines.append(
            f"{display_llm_name(str(row['llm']))}: n={row['n']} | t={row['t']:.5f} | "
            f"p_ttest={row['p']:.6f} | p_ttest_holm={p_adj:.6f} | mean_diff={row['mean_diff']:.4f} | "
            f"median_diff={row['median_diff']:.4f} | cohen_d={row['cohen_d']:.4f} | "
            f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )


def build_paired_value_maps(
    records: Sequence[ComplexityRecord],
    value_getter: Callable[[ComplexityRecord], float],
) -> tuple[dict[str, tuple[list[float], list[float]]], dict[str, tuple[list[float], list[float]]], dict[str, tuple[list[float], list[float]]]]:
    original_lookup = create_original_lookup(records)
    mode_lookup = create_mode_llm_lookup(records)
    llms = sorted({record.llm for record in records if record.mode in GENERATED_MODES})

    original_vs_assisted: dict[str, tuple[list[float], list[float]]] = {}
    original_vs_autonomous: dict[str, tuple[list[float], list[float]]] = {}
    assisted_vs_autonomous: dict[str, tuple[list[float], list[float]]] = {}

    for llm in llms:
        generated_assisted: list[float] = []
        original_assisted: list[float] = []
        generated_autonomous: list[float] = []
        original_autonomous: list[float] = []
        assisted_values: list[float] = []
        autonomous_values: list[float] = []

        for project, original_record in sorted(original_lookup.items()):
            assisted = mode_lookup.get(("assisted", llm, project))
            autonomous = mode_lookup.get(("autonomous", llm, project))
            original_value = value_getter(original_record)

            if assisted is not None:
                assisted_value = value_getter(assisted)
                if math.isfinite(assisted_value) and math.isfinite(original_value):
                    generated_assisted.append(assisted_value)
                    original_assisted.append(original_value)

            if autonomous is not None:
                autonomous_value = value_getter(autonomous)
                if math.isfinite(autonomous_value) and math.isfinite(original_value):
                    generated_autonomous.append(autonomous_value)
                    original_autonomous.append(original_value)

            if assisted is not None and autonomous is not None:
                assisted_value = value_getter(assisted)
                autonomous_value = value_getter(autonomous)
                if math.isfinite(assisted_value) and math.isfinite(autonomous_value):
                    assisted_values.append(assisted_value)
                    autonomous_values.append(autonomous_value)

        original_vs_assisted[llm] = (generated_assisted, original_assisted)
        original_vs_autonomous[llm] = (generated_autonomous, original_autonomous)
        assisted_vs_autonomous[llm] = (assisted_values, autonomous_values)

    return original_vs_assisted, original_vs_autonomous, assisted_vs_autonomous


def build_analysis_text(
    records: Sequence[ComplexityRecord],
    local_records: Sequence[LocalFileRecord],
    unmatched_components: Sequence[dict],
    missing_local_records: Sequence[LocalFileRecord],
    plot_paths: Sequence[Path],
    summary_rows: Sequence[dict[str, str]],
    scan_metadata: dict[str, str],
    project_measures: dict,
    np,
    stats,
    pairwise_tukeyhsd,
    multipletests,
) -> str:
    lines: list[str] = []
    lines.append("Cyclomatic Code Complexity Analysis")
    lines.append(f"Local generated files discovered: {len(local_records)}")
    lines.append(f"Files returned by SonarQube API: {len(records)}")
    lines.append(f"Unmatched API components: {len(unmatched_components)}")
    lines.append(f"Local files missing from SonarQube results: {len(missing_local_records)}")
    if scan_metadata:
        lines.append("\nScan Metadata")
        for key in sorted(scan_metadata):
            lines.append(f"{key}: {scan_metadata[key]}")
    if project_measures:
        lines.append("\nProject Measures")
        component = project_measures.get("component", {})
        for measure in component.get("measures", []):
            lines.append(f"{measure.get('metric')}: {measure.get('value')}")

    lines.append("\nSummary Statistics")
    for row in summary_rows:
        lines.append(
            f"{row['group_type']} | {row['metric']} | {row['group']}: "
            f"n={row['count']} mean={row['mean']} median={row['median']} "
            f"min={row['min']} max={row['max']} std={row['std_dev']} "
            f"p25={row['p25']} p75={row['p75']}"
        )

    lines.append("\nANOVA by mode [raw complexity]")
    lines.append("(one-way ANOVA + Tukey HSD post-hoc)")
    anova_and_tukey_summary(
        "Metric: complexity | factor: mode",
        group_values(records, lambda r: r.complexity, lambda r: display_mode_name(r.mode), lambda r: r.mode in ALL_MODES),
        stats,
        np,
        pairwise_tukeyhsd,
        lines,
    )

    lines.append("\nANOVA by LLM [raw complexity]")
    lines.append("(one-way ANOVA + Tukey HSD post-hoc)")
    anova_and_tukey_summary(
        "Mode: assisted | Metric: complexity | factor: model",
        group_values(records, lambda r: r.complexity, lambda r: display_llm_name(r.llm), lambda r: r.mode == "assisted"),
        stats,
        np,
        pairwise_tukeyhsd,
        lines,
    )
    anova_and_tukey_summary(
        "Mode: autonomous | Metric: complexity | factor: model",
        group_values(records, lambda r: r.complexity, lambda r: display_llm_name(r.llm), lambda r: r.mode == "autonomous"),
        stats,
        np,
        pairwise_tukeyhsd,
        lines,
    )
    anova_and_tukey_summary(
        "Mode: assisted + autonomous | Metric: complexity | factor: model",
        group_values(records, lambda r: r.complexity, lambda r: display_llm_name(r.llm), lambda r: r.mode in GENERATED_MODES),
        stats,
        np,
        pairwise_tukeyhsd,
        lines,
    )

    lines.append("\nANOVA by mode [normalized complexity]")
    lines.append("(one-way ANOVA + Tukey HSD post-hoc)")
    anova_and_tukey_summary(
        "Metric: complexity_per_100_ncloc | factor: mode",
        group_values(
            records,
            lambda r: r.complexity_per_100_ncloc,
            lambda r: display_mode_name(r.mode),
            lambda r: r.mode in ALL_MODES,
        ),
        stats,
        np,
        pairwise_tukeyhsd,
        lines,
    )

    lines.append("\nANOVA by LLM [normalized complexity]")
    lines.append("(one-way ANOVA + Tukey HSD post-hoc)")
    anova_and_tukey_summary(
        "Mode: assisted | Metric: complexity_per_100_ncloc | factor: model",
        group_values(
            records,
            lambda r: r.complexity_per_100_ncloc,
            lambda r: display_llm_name(r.llm),
            lambda r: r.mode == "assisted",
        ),
        stats,
        np,
        pairwise_tukeyhsd,
        lines,
    )
    anova_and_tukey_summary(
        "Mode: autonomous | Metric: complexity_per_100_ncloc | factor: model",
        group_values(
            records,
            lambda r: r.complexity_per_100_ncloc,
            lambda r: display_llm_name(r.llm),
            lambda r: r.mode == "autonomous",
        ),
        stats,
        np,
        pairwise_tukeyhsd,
        lines,
    )
    anova_and_tukey_summary(
        "Mode: assisted + autonomous | Metric: complexity_per_100_ncloc | factor: model",
        group_values(
            records,
            lambda r: r.complexity_per_100_ncloc,
            lambda r: display_llm_name(r.llm),
            lambda r: r.mode in GENERATED_MODES,
        ),
        stats,
        np,
        pairwise_tukeyhsd,
        lines,
    )

    orig_assist_raw, orig_auto_raw, assist_auto_raw = build_paired_value_maps(records, lambda r: r.complexity)
    lines.append("\nPaired t-tests [raw complexity]: Original vs Assisted by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: across LLMs in this section")
    paired_ttest_summary(
        "Metric: complexity",
        orig_assist_raw,
        stats,
        np,
        multipletests,
        lines,
    )
    lines.append("\nPaired t-tests [raw complexity]: Original vs Autonomous by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: across LLMs in this section")
    paired_ttest_summary(
        "Metric: complexity",
        orig_auto_raw,
        stats,
        np,
        multipletests,
        lines,
    )
    lines.append("\nPaired t-tests [raw complexity]: Assisted vs Autonomous by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: across LLMs in this section")
    paired_ttest_summary(
        "Metric: complexity",
        assist_auto_raw,
        stats,
        np,
        multipletests,
        lines,
    )

    orig_assist_norm, orig_auto_norm, assist_auto_norm = build_paired_value_maps(
        records,
        lambda r: r.complexity_per_100_ncloc,
    )
    lines.append("\nPaired t-tests [normalized complexity]: Original vs Assisted by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: across LLMs in this section")
    paired_ttest_summary(
        "Metric: complexity_per_100_ncloc",
        orig_assist_norm,
        stats,
        np,
        multipletests,
        lines,
    )
    lines.append("\nPaired t-tests [normalized complexity]: Original vs Autonomous by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: across LLMs in this section")
    paired_ttest_summary(
        "Metric: complexity_per_100_ncloc",
        orig_auto_norm,
        stats,
        np,
        multipletests,
        lines,
    )
    lines.append("\nPaired t-tests [normalized complexity]: Assisted vs Autonomous by model")
    lines.append("(paired by project and model)")
    lines.append("Holm correction: across LLMs in this section")
    paired_ttest_summary(
        "Metric: complexity_per_100_ncloc",
        assist_auto_norm,
        stats,
        np,
        multipletests,
        lines,
    )

    lines.append("\nFigures")
    for plot_path in plot_paths:
        lines.append(str(plot_path))

    if unmatched_components:
        lines.append("\nUnmatched API Components")
        for component in unmatched_components:
            lines.append(
                json.dumps(
                    {
                        "key": component.get("key"),
                        "name": component.get("name"),
                        "path": component.get("path"),
                    }
                )
            )
    if missing_local_records:
        lines.append("\nLocal Files Missing from SonarQube Results")
        for record in missing_local_records:
            lines.append(record.relative_path)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    repos_dir = Path(args.repos_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_records, path_lookup = collect_local_metadata(repos_dir)

    np, plt, Line2D, stats, pairwise_tukeyhsd, multipletests = import_external_libs()

    scan_metadata: dict[str, str] = {}
    project_measures: dict
    components: list[dict]

    if args.reuse_cached_api:
        scan_metadata = {
            "serverUrl": args.sonar_host_url,
            "projectKey": args.sonar_project_key or "cached",
            "mode": "reuse_cached_api",
        }
        project_measures = load_cached_json(output_dir / "project_measures.json", "project measures")
        components = load_cached_json(output_dir / "component_tree_raw.json", "component tree")
    else:
        token, project_key, _organization = require_sonar_settings(args)
        if args.skip_scan:
            scan_metadata = {
                "serverUrl": args.sonar_host_url,
                "projectKey": project_key,
                "mode": "skip_scan",
            }
        else:
            scan_metadata = run_sonar_scanner(args, token, output_dir)
            ce_task_id = scan_metadata.get("ceTaskId")
            server_url = scan_metadata.get("serverUrl", args.sonar_host_url)
            if not ce_task_id:
                raise SystemExit("sonar-scanner did not return a ceTaskId in report-task.txt.")
            wait_for_analysis_completion(
                server_url,
                ce_task_id,
                token,
                args.analysis_timeout_s,
                args.poll_interval_s,
                output_dir,
            )

        server_url = scan_metadata.get("serverUrl", args.sonar_host_url)
        project_measures = fetch_project_measures(server_url, project_key, token)
        components = fetch_file_measure_components(server_url, project_key, token)

        (output_dir / "project_measures.json").write_text(
            json.dumps(project_measures, indent=2),
            encoding="utf-8",
        )
        (output_dir / "component_tree_raw.json").write_text(
            json.dumps(components, indent=2),
            encoding="utf-8",
        )
    (output_dir / "local_generated_files.json").write_text(
        json.dumps(
            [
                {
                    "project": record.project,
                    "relative_path": record.relative_path,
                    "file": record.file,
                    "mode": record.mode,
                    "llm": record.llm,
                }
                for record in local_records
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    records, unmatched_components = build_complexity_records(components, path_lookup)
    missing_local_records = find_missing_local_records(local_records, records)
    (output_dir / "missing_local_files.json").write_text(
        json.dumps([record.relative_path for record in missing_local_records], indent=2),
        encoding="utf-8",
    )
    if not records:
        raise SystemExit(
            "No file-level SonarQube complexity records matched the local GENAIGREENML files. "
            "Check the Sonar project key, scanner configuration, and API credentials."
        )

    raw_csv_path = output_dir / "cyclomatic_code_complexity_raw.csv"
    save_raw_csv(raw_csv_path, records)

    summary_rows = build_summary_rows(records)
    save_summary_csv(output_dir / "cyclomatic_code_complexity_summary.csv", summary_rows)

    paired_rows = build_paired_rows_from_records(records)
    save_paired_csv(output_dir / "cyclomatic_code_complexity_paired_deltas.csv", paired_rows)

    plot_paths: list[Path] = []
    for path in (
        save_mode_boxplot(records, output_dir, plt, np, Line2D),
        save_llm_boxplot(
            records,
            output_dir,
            "assisted",
            "Cyclomatic Complexity by LLM (Assisted)",
            "boxplot_complexity_by_llm_assisted.png",
            plt,
            np,
            Line2D,
        ),
        save_llm_boxplot(
            records,
            output_dir,
            "autonomous",
            "Cyclomatic Complexity by LLM (Autonomous)",
            "boxplot_complexity_by_llm_autonomous.png",
            plt,
            np,
            Line2D,
        ),
        save_llm_boxplot(
            records,
            output_dir,
            None,
            "Cyclomatic Complexity by LLM (Assisted + Autonomous)",
            "boxplot_complexity_by_llm_generated.png",
            plt,
            np,
            Line2D,
        ),
        save_scatter(records, output_dir, plt, Line2D),
        save_density_boxplot(records, output_dir, plt, np, Line2D),
        save_delta_boxplots(
            build_delta_groups(records, lambda r: r.complexity),
            "Delta Cyclomatic Complexity vs Original",
            "Delta Cyclomatic Complexity",
            output_dir / "boxplot_delta_complexity_vs_original.png",
            plt,
            np,
            Line2D,
        ),
        save_delta_boxplots(
            build_delta_groups(records, lambda r: r.complexity_per_100_ncloc),
            "Delta Cyclomatic Complexity per 100 NCLOC vs Original",
            "Delta Cyclomatic Complexity per 100 NCLOC",
            output_dir / "boxplot_delta_complexity_per_100_ncloc_vs_original.png",
            plt,
            np,
            Line2D,
        ),
    ):
        if path is not None:
            plot_paths.append(path)

    analysis_text = build_analysis_text(
        records,
        local_records,
        unmatched_components,
        missing_local_records,
        plot_paths,
        summary_rows,
        scan_metadata,
        project_measures,
        np,
        stats,
        pairwise_tukeyhsd,
        multipletests,
    )
    analysis_path = output_dir / "cyclomatic_code_complexity_analysis.txt"
    analysis_path.write_text(analysis_text, encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print(f"Matched SonarQube records: {len(records)}")
    print(f"Local generated files discovered: {len(local_records)}")
    print(f"Unmatched API components: {len(unmatched_components)}")
    print(f"Local files missing from SonarQube results: {len(missing_local_records)}")
    print(f"Raw CSV: {raw_csv_path}")
    print(f"Summary CSV: {output_dir / 'cyclomatic_code_complexity_summary.csv'}")
    print(f"Paired deltas CSV: {output_dir / 'cyclomatic_code_complexity_paired_deltas.csv'}")
    print(f"Analysis text: {analysis_path}")
    print("Figures:")
    for plot_path in plot_paths:
        print(plot_path)


if __name__ == "__main__":
    main()
