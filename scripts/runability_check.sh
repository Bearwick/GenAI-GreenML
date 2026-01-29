#!/usr/bin/env bash
# Run ML projects in ./repos with venv activation and simple entrypoint selection.

set -u

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
REPOS_DIR="$REPO_ROOT/repos"
LOG_FILE="$REPOS_DIR/multiple_py_in_project"
RUN_ERROR_LOG="$REPOS_DIR/running_errors"
RUN_COMPLETED_LOG="$REPOS_DIR/ml_projects_run_completed"

if [[ ! -d "$REPOS_DIR" ]]; then
  echo "Repos directory not found: $REPOS_DIR" >&2
  exit 1
fi

> "$LOG_FILE"
> "$RUN_ERROR_LOG"
> "$RUN_COMPLETED_LOG"

# Iterate over first-level directories in repos
while IFS= read -r -d '' project_dir; do
  project_name="$(basename "$project_dir")"
  echo "[*] Project: $project_name"

  cd "$project_dir" || continue

  if [[ ! -d "venv" ]]; then
    echo "[!] venv not found in $project_name; skipping"
    cd "$REPOS_DIR" || exit 1
    continue
  fi

  # shellcheck disable=SC1091
  source "venv/bin/activate"

  if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt >/dev/null 2>&1 || echo "[!] pip install failed in $project_name"
  else
    echo "[!] requirements.txt missing in $project_name"
  fi

  if [[ -f "original.py" ]]; then
    if ! python "original.py"; then
      echo "[!] Run failed: $project_name/original.py"
      echo "$project_name|original.py" >> "$RUN_ERROR_LOG"
    else
      echo "$project_name|original.py" >> "$RUN_COMPLETED_LOG"
    fi
  else
    py_files=()
    while IFS= read -r -d '' py_file; do
      py_files+=("$py_file")
    done < <(find . -type f -name "*.py" \
      -not -name "GENAIGREENML*.py" \
      -not -path "./venv/*" \
      -not -path "./.venv/*" \
      -not -path "./.git/*" -print0)

    if [[ ${#py_files[@]} -eq 1 ]]; then
      if ! python "${py_files[0]}"; then
        echo "[!] Run failed: ${py_files[0]}"
        echo "$project_name|${py_files[0]}" >> "$RUN_ERROR_LOG"
      else
        echo "$project_name|${py_files[0]}" >> "$RUN_COMPLETED_LOG"
      fi
    elif [[ ${#py_files[@]} -gt 1 ]]; then
      echo "$project_name" >> "$LOG_FILE"
      echo "[i] Multiple .py files; logged to $LOG_FILE"
    else
      echo "[i] No .py files found in $project_name"
    fi
  fi

  deactivate >/dev/null 2>&1 || true
  cd "$REPOS_DIR" || exit 1

  echo ""
done < <(find "$REPOS_DIR" -mindepth 1 -maxdepth 1 -type d -print0)
