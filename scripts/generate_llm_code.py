#!/usr/bin/env python3
"""
Traverse repos/, detect a primary ML source file + dataset headers, then ask selected LLMs
to generate refactored variants.

Changes vs. your original:
- (1) Single LLM registry + single generate() path (no repetitive if/elif ladders)
- (2) Config + clients (no global API modules)
- (3) pathlib traversal with explicit ignore dirs; cross-platform
- (4) Better source-file heuristics (supports multi-file repos)
- (5) Safer dataset detection (quoted paths + ML loader context + data/ dirs)
- (6) Exceptions logged with repr(e) (no silent swallowing)
- (7) Atomic writes + skip-if-exists; override with --force
- (8) argparse CLI + validation + filters
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from threading import Lock
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------
# Config
# ----------------------------

DATASET_EXTS = [
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".jsonl",
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".arff",
    ".sav",
    ".mat",
]

TEXT_HEADER_EXTS = {".csv", ".tsv", ".txt", ".jsonl"}  # what "first line headers" makes sense for

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

DEFAULT_ALL_LLMS = ["chatgpt", "gemini", "codex", "claude"] # "groq"


@dataclass(frozen=True)
class Config:
    base_dir: Path
    repos_dir: Path
    base_name: str
    scripts_venv_py: Path
    apis_dir: Path


# ----------------------------
# Utilities
# ----------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def should_skip_dir(dir_path: Path, ignore_names: set[str]) -> bool:
    parts = dir_path.parts
    # Skip if any path component is an ignored directory name
    return any(p in ignore_names for p in parts)


def iter_files(project_dir: Path, ignore_names: set[str]) -> Iterable[Path]:
    # Walk manually so we can prune dirs (faster than rglob for big repos)
    for root, dirs, files in os.walk(project_dir):
        root_p = Path(root)
        # prune dirs in-place
        dirs[:] = [d for d in dirs if d not in ignore_names and not d.startswith(".")]
        if should_skip_dir(root_p, ignore_names):
            continue
        for name in files:
            yield root_p / name


def list_project_files(project_dir: Path, ignore_names: set[str]) -> List[Path]:
    return [p for p in iter_files(project_dir, ignore_names)]


# ----------------------------
# Source picking (heuristic)
# ----------------------------

ML_IMPORT_HINTS = (
    "sklearn",
    "scikit-learn",
    "torch",
    "tensorflow",
    "keras",
    "xgboost",
    "lightgbm",
    "catboost",
    "pandas",
    "numpy",
)

PREFERRED_ENTRY_NAMES = (
    "original.py",
    "main.py",
    "train.py",
    "run.py",
    "pipeline.py",
    "model.py",
    "app.py",
    "script.py",
)


def score_source_file(py_file: Path, content: str) -> int:
    """
    Higher score = more likely the "main" ML script.
    Cheap heuristics only.
    """
    score = 0
    name = py_file.name.lower()

    # filename preference
    for i, pref in enumerate(PREFERRED_ENTRY_NAMES):
        if name == pref:
            score += 500 - i * 20

    # entry-ish markers
    if "if __name__ == '__main__'" in content or 'if __name__ == "__main__"' in content:
        score += 120
    if re.search(r"\bargparse\b", content):
        score += 40

    # ML hints
    for hint in ML_IMPORT_HINTS:
        if hint in content:
            score += 25

    # data loading hints
    if re.search(r"\bread_csv\b|\bread_parquet\b|\bload\b|\bdataset\b", content):
        score += 30

    # penalize tiny util files
    loc = content.count("\n") + 1 if content else 0
    score += min(loc, 800) // 5  # up to +160

    # slight penalty for obvious non-ML / packaging scripts
    if name in {"setup.py", "conftest.py"}:
        score -= 200

    return score


def pick_source_file(
    project_dir: Path,
    files: List[Path],
    base_name: str,
) -> Optional[Path]:
    # 1) explicit original.py wins
    original = project_dir / "original.py"
    if original.is_file():
        return original

    # 2) gather candidate .py files excluding generated ones and venv
    candidates: List[Path] = []
    for p in files:
        if p.suffix.lower() != ".py":
            continue
        if p.name == "requirements.txt":
            continue
        if p.name.startswith(base_name):
            continue
        candidates.append(p)

    if not candidates:
        return None

    # 3) If exactly one candidate, keep your old behavior
    if len(candidates) == 1:
        return candidates[0]

    # 4) Heuristic scoring across candidates
    scored: List[Tuple[int, Path]] = []
    for p in candidates:
        content = read_text(p)
        s = score_source_file(p, content)
        scored.append((s, p))

    scored.sort(key=lambda t: t[0], reverse=True)

    # Require at least some signal to avoid picking random tiny utils.py in non-ML repos
    best_score, best_path = scored[0]
    if best_score < 80:
        return None

    return best_path


# ----------------------------
# Dataset detection + headers
# ----------------------------

def find_preferred_dataset(files: List[Path]) -> Optional[Path]:
    # Prefer explicit GENAIGREENMLDATASET.*
    preferred = [p for p in files if p.is_file() and p.name.startswith("GENAIGREENMLDATASET.")]
    return preferred[0] if preferred else None


def find_dataset_candidates(files: List[Path]) -> List[Path]:
    exts = {e.lower() for e in DATASET_EXTS}
    cands = [p for p in files if p.is_file() and p.suffix.lower() in exts]
    # Prefer common data directories
    def rank(p: Path) -> int:
        parts = [x.lower() for x in p.parts]
        r = 0
        if any(x in {"data", "dataset", "datasets", "input"} for x in parts):
            r += 50
        # smaller files often better than giant binary blobs, but we don’t stat heavily; light rank by extension
        if p.suffix.lower() in {".csv", ".tsv", ".jsonl"}:
            r += 20
        return r

    cands.sort(key=rank, reverse=True)
    return cands


def find_dataset_in_code(project_dir: Path, src_file: Path, files: List[Path]) -> Optional[Path]:
    """
    Look for quoted dataset filenames/paths in common loader contexts.
    Example matches: pd.read_csv("data/train.csv"), open('x.jsonl'), np.load("arr.npy")
    """
    code = read_text(src_file)
    if not code:
        return None

    # Only match quoted strings ending in allowed extensions (reduces false positives a lot)
    exts_alt = "|".join(re.escape(e.lstrip(".")) for e in DATASET_EXTS)
    quoted_path_re = re.compile(
        rf"""["']([^"']+\.({exts_alt}))["']""",
        re.IGNORECASE,
    )

    # Prefer those near common loader calls (simple window check)
    loader_hint_re = re.compile(
        r"\b(read_csv|read_table|read_parquet|read_json|loadtxt|genfromtxt|np\.load|pickle\.load|joblib\.load|open)\b",
        re.IGNORECASE,
    )

    matches = list(quoted_path_re.finditer(code))
    if not matches:
        return None

    # Create lookup by basename for fast match
    by_name: Dict[str, List[Path]] = {}
    for p in files:
        by_name.setdefault(p.name, []).append(p)

    best: Optional[Path] = None
    best_rank = -10**9

    for m in matches:
        raw = m.group(1)
        base = Path(raw).name
        # Determine context rank: loader hints near the match?
        start = max(0, m.start() - 80)
        end = min(len(code), m.end() + 80)
        window = code[start:end]
        context_rank = 0
        if loader_hint_re.search(window):
            context_rank += 100

        # Prefer paths that resolve within project, else match by basename within repo
        candidate: Optional[Path] = None
        resolved = (project_dir / raw).resolve()
        if resolved.exists() and project_dir.resolve() in resolved.parents:
            candidate = resolved
            context_rank += 60
        else:
            # fallback: any file with same basename
            options = by_name.get(base, [])
            if options:
                candidate = options[0]
                context_rank += 30

        if candidate is None:
            continue

        # Prefer files in data-ish directories
        parts = [x.lower() for x in candidate.parts]
        if any(x in {"data", "dataset", "datasets", "input"} for x in parts):
            context_rank += 20

        if context_rank > best_rank:
            best_rank = context_rank
            best = candidate

    return best


def dataset_headers(dataset_path: Optional[Path]) -> str:
    if not dataset_path:
        return ""
    if dataset_path.suffix.lower() not in TEXT_HEADER_EXTS:
        return ""
    try:
        with dataset_path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.readline().strip()
    except OSError:
        return ""


def find_dataset_file(project_dir: Path, src_file: Path, files: List[Path]) -> Optional[Path]:
    # 1) Prefer explicit GENAIGREENMLDATASET.*
    preferred = find_preferred_dataset(files)
    if preferred:
        return preferred

    # 2) Find referenced dataset in code (safer)
    referenced = find_dataset_in_code(project_dir, src_file, files)
    if referenced:
        return referenced

    # 3) Fallback to best-ranked dataset candidate
    candidates = find_dataset_candidates(files)
    return candidates[0] if candidates else None



# ----------------------------
# Pause Logic
# ----------------------------

@dataclass
class EveryNRequestsPause:
    every: int
    sleep_seconds: int
    _count: int = 0
    _lock: Lock = Lock()

    def hit(self, llm_name: str) -> None:
        with self._lock:
            self._count += 1
            if self.every > 0 and self._count % self.every == 0:
                logging.info("[i] %s rate limit pause (%ss) after %d requests", llm_name, self.sleep_seconds, self._count)
                time.sleep(self.sleep_seconds)

# ----------------------------
# LLM client registry
# ----------------------------

@dataclass
class LLMClient:
    name: str
    generate_code: Callable[[str, str, str], str]  # (mode, source_code, headers) -> str


def load_llm_clients(apis_dir: Path, enable_gemini_pause: bool = False) -> Dict[str, Optional[LLMClient]]:
    """
    Imports your modules from scripts/APIs and builds a normalized registry.

    Expected module functions:
      module.generate_code(mode, source_code, headers) -> str
    """
    sys.path.insert(0, str(apis_dir))

    registry: Dict[str, Optional[LLMClient]] = {}

    gemini_pause = EveryNRequestsPause(every=10, sleep_seconds=60)

    def _try_load(name: str, import_name: str) -> Optional[LLMClient]:
        try:
            mod = __import__(import_name)
            gen = getattr(mod, "generate_code", None)
            if not callable(gen):
                logging.warning("[!] %s module loaded but missing generate_code()", name)
                return None

            if name == "gemini" and enable_gemini_pause:
                def wrapped_generate(mode: str, source_code: str, headers: str) -> str:
                    gemini_pause.hit("Gemini")
                    return gen(mode, source_code, headers)

                return LLMClient(name=name, generate_code=wrapped_generate)

            return LLMClient(name=name, generate_code=gen)

        except Exception as e:
            logging.debug("[i] Could not import %s (%s): %r", name, import_name, e)
            return None

    registry["gemini"] = _try_load("gemini", "gemini_api")
    registry["chatgpt"] = _try_load("chatgpt", "chatgpt_api")
    registry["codex"] = _try_load("codex", "openai_codex_api")
    registry["claude"] = _try_load("claude", "claude_api")
    # registry["groq"] = _try_load("groq", "groq_api")

    return registry


def generate(
    client: LLMClient,
    mode: str,
    src_file: Path,
    headers: str,
    project_name: str,
) -> str:
    try:
        source_code = None
        #if mode != "autonomous":
        source_code = read_text(src_file)

        if not source_code.strip():
            return ""
            
        out = client.generate_code(mode, source_code, headers)
        return (out or "").strip()
    except Exception as e:
        logging.error("[!] %s | %s | %s generation failed: %r", project_name, client.name, mode, e)
        return ""


# ----------------------------
# Output
# ----------------------------

def output_path(project_dir: Path, base_name: str, mode: str, llm_name: Optional[str]) -> Path:
    if mode == "original_telemetry":
        return project_dir / f"{base_name}_original_telemetry_{llm_name}.py"
    assert llm_name is not None
    return project_dir / f"{base_name}_{mode}_{llm_name}.py"


def write_output_file(
    out_file: Path,
    script_name: str,
    llm_name: str,
    mode: str,
    code: str,
    force: bool,
) -> bool:
    """
    Returns True if written, False if skipped.
    """
    if out_file.exists() and not force:
        # skip if non-empty
        try:
            if out_file.stat().st_size > 0:
                logging.info("[i] Skip existing: %s", out_file.name)
                return False
        except OSError:
            pass

    header = (
        f"# Generated by {script_name}\n"
        f"# LLM: {llm_name}\n"
        f"# Mode: {mode}\n\n"
    )
    atomic_write_text(out_file, header + code)
    logging.info("[+] Wrote: %s", out_file.name)
    return True


# ----------------------------
# Requirements update (kept as-is, but called optionally)
# ----------------------------

def update_requirements(project_dir: Path) -> None:
    venv_py = project_dir / "venv" / "bin" / "python"
    if not venv_py.is_file():
        logging.warning("[!] venv python not found; skipping requirements update for %s", project_dir.name)
        return

    subprocess.run(
        [str(venv_py), "-m", "pip", "install", "-q", "pipreqs"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        [
            str(venv_py),
            "-m",
            "pipreqs",
            str(project_dir),
            "--force",
            "--encoding",
            "utf-8",
            "--ignore",
            str(project_dir / "venv"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


# ----------------------------
# CLI / main
# ----------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LLM refactors for repos under repos/")
    p.add_argument(
        "--llms",
        nargs="*",
        default=None,
        help=f"LLMs to use (default: {DEFAULT_ALL_LLMS}). First one is primary for original_telemetry.",
    )
    p.add_argument(
        "--modes",
        nargs="*",
        default=["assisted", "autonomous"],
        choices=["assisted", "autonomous"],
        help="Modes to generate for each LLM (original_telemetry is always primary-only).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--dry-run", action="store_true", help="Scan and report, but do not call LLMs or write outputs.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    p.add_argument("--project-regex", default=None, help="Only process projects whose directory name matches this regex.")
    p.add_argument("--max-projects", type=int, default=None, help="Stop after processing N projects.")
    p.add_argument(
        "--no-requirements",
        action="store_true",
        help="Do not run pipreqs requirements update per project.",
    )
    p.add_argument(
        "--gemini-pause",
        action="store_true",
        help="Enable Gemini rate-limit pause (60s every 10 requests). Disabled by default.",
    )
    return p.parse_args(argv)


def ensure_scripts_venv(config: Config) -> None:
    if config.scripts_venv_py.is_file() and Path(sys.executable).resolve() != config.scripts_venv_py.resolve():
        os.execv(str(config.scripts_venv_py), [str(config.scripts_venv_py), *sys.argv])

    if not config.scripts_venv_py.is_file():
        logging.warning("[!] scripts venv not found. Create it with: python3 -m venv scripts/venv")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    setup_logging(args.verbose)

    base_dir = Path(__file__).resolve().parent.parent
    config = Config(
        base_dir=base_dir,
        repos_dir=base_dir / "repos",
        base_name="GENAIGREENML",
        scripts_venv_py=base_dir / "scripts" / "venv" / "bin" / "python",
        apis_dir=base_dir / "scripts" / "APIs",
    )

    ensure_scripts_venv(config)

    if not config.repos_dir.is_dir():
        raise SystemExit(f"Repos directory not found: {config.repos_dir}")

    llm_names = args.llms if args.llms is not None and len(args.llms) > 0 else list(DEFAULT_ALL_LLMS)

    # validate llms
    unknown = [x for x in llm_names if x not in DEFAULT_ALL_LLMS]
    if unknown:
        logging.error("[!] Unknown LLM(s): %s. Allowed: %s", unknown, DEFAULT_ALL_LLMS)
        return 2

    primary_llm_name = llm_names[0]
    modes: List[str] = list(args.modes)

    # load clients
    clients = load_llm_clients(config.apis_dir, enable_gemini_pause=args.gemini_pause)

    # warn about missing clients
    for name in llm_names:
        if clients.get(name) is None:
            logging.warning("[!] %s client missing/unavailable (import failed or no generate_code())", name)

    project_re: Optional[re.Pattern[str]] = re.compile(args.project_regex) if args.project_regex else None
    ignore_names = set(DEFAULT_IGNORE_DIR_NAMES)

    processed = 0

    for project_dir in sorted(config.repos_dir.iterdir(), key=lambda p: p.name.lower()):
        if not project_dir.is_dir():
            continue

        if project_re and not project_re.search(project_dir.name):
            continue

        files = list_project_files(project_dir, ignore_names)

        # Keep your original "skip if GENAIGREENMLDATASET present"
        if any(p.name.startswith("GENAIGREENMLDATASET.") for p in files):
            logging.info("[i] Skipping %s (GENAIGREENMLDATASET present)", project_dir.name)
            continue

        logging.info("[*] Project: %s", project_dir.name)

        src_file = pick_source_file(project_dir, files, config.base_name)
        if not src_file:
            logging.info("[i] Skipping %s (no suitable source file found)", project_dir.name)
            continue

        dataset_path = find_dataset_file(project_dir, src_file, files)
        headers = dataset_headers(dataset_path)

        if args.dry_run:
            logging.info("    src: %s", src_file.relative_to(project_dir))
            if dataset_path:
                logging.info("    dataset: %s", dataset_path.relative_to(project_dir))
            else:
                logging.info("    dataset: (none)")
            processed += 1
            if args.max_projects and processed >= args.max_projects:
                break
            continue

        # (Primary-only) original_telemetry
        primary_client = clients.get(primary_llm_name)
        skip_primary_existing = False
        out_primary = output_path(project_dir, config.base_name, "original_telemetry", primary_llm_name)
        if out_primary.exists() and not args.force:
            try:
                if out_primary.stat().st_size > 0:
                    logging.info("[i] Skip existing: %s", out_primary.name)
                    skip_primary_existing = True
            except OSError:
                pass

        if skip_primary_existing:
            pass
        elif primary_client is not None:
            code = generate(primary_client, "original_telemetry", src_file, None, project_dir.name)
            if code:
                write_output_file(
                    out_file=out_primary,
                    script_name=Path(__file__).name,
                    llm_name=primary_llm_name,
                    mode="original_telemetry",
                    code=code,
                    force=args.force,
                )
            else:
                logging.info("[i] Skipping %s original_telemetry (no response)", primary_llm_name)
        else:
            logging.info("[i] Skipping %s original_telemetry (client unavailable)", primary_llm_name)

        # Per-LLM assisted/autonomous
        for llm_name in llm_names:
            client = clients.get(llm_name)
            if client is None:
                logging.info("[i] Skipping %s (client unavailable)", llm_name)
                continue

            for mode in modes:
                out = output_path(project_dir, config.base_name, mode, llm_name)
                if out.exists() and not args.force:
                    try:
                        if out.stat().st_size > 0:
                            logging.info("[i] Skip existing: %s", out.name)
                            continue
                    except OSError:
                        pass

                if mode == "autonomous":
                    src_for_mode = src_file #None
                else:
                    src_for_mode = src_file

                code = generate(client, mode, src_for_mode, headers, project_dir.name)
                if not code:
                    logging.info("[i] Skipping %s %s (no response)", llm_name, mode)
                    continue

                write_output_file(
                    out_file=out,
                    script_name=Path(__file__).name,
                    llm_name=llm_name,
                    mode=mode,
                    code=code,
                    force=args.force,
                )

        if not args.no_requirements:
            update_requirements(project_dir)

        processed += 1
        if args.max_projects and processed >= args.max_projects:
            break

    logging.info("[+] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
