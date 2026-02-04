#!/usr/bin/env bash
set -euo pipefail

REPOS_DIR="repos"
LOG_FILE="$REPOS_DIR/multiple_datasets_in_project"

> "$LOG_FILE"

dataset_find() {
  local project="$1"
  find "$project" -type f \( \
    -name "*.csv" -o -name "*.tsv" -o -name "*.txt" -o -name "*.json" -o -name "*.jsonl" -o \
    -name "*.xlsx" -o -name "*.xls" -o -name "*.parquet" -o -name "*.feather" -o \
    -name "*.npy" -o -name "*.npz" -o -name "*.pkl" -o -name "*.pickle" -o \
    -name "*.h5" -o -name "*.hdf5" -o -name "*.arff" -o -name "*.sav" -o -name "*.mat" \
  \) \
  -not -name "requirements.txt" \
  -not -path "$project/venv/*" \
  -not -path "$project/.venv/*" \
  -not -path "$project/.git/*" \
  -not -path "$project/.*/**"
}

normalize_header() {
  local header="$1"
  # Keep printable chars, normalize whitespace, trim, lowercase.
  header="$(printf "%s" "$header" | LC_ALL=C tr -cd '[:print:]\t' | LC_ALL=C tr -s '[:space:]' ' ')"
  header="${header#"${header%%[![:space:]]*}"}"
  header="${header%"${header##*[![:space:]]}"}"
  printf "%s" "$header" | LC_ALL=C tr '[:upper:]' '[:lower:]'
}

for project in "$REPOS_DIR"/*; do
  [[ ! -d "$project" ]] && continue
  project_name=$(basename "$project")

  dataset_files=()
  while IFS= read -r f; do
    [[ -n "$f" ]] && dataset_files+=("$f")
  done < <(dataset_find "$project")
  if [[ ${#dataset_files[@]} -le 1 ]]; then
    continue
  fi

  base_header=""
  mismatch=false
  comparable_count=0
  for f in "${dataset_files[@]}"; do
    header="$(LC_ALL=C head -n 1 "$f" 2>/dev/null | LC_ALL=C tr -d '\r' 2>/dev/null || true)"
    header="$(normalize_header "$header")"
    [[ -z "$header" ]] && continue
    comparable_count=$((comparable_count + 1))
    if [[ -z "$base_header" ]]; then
      base_header="$header"
    elif [[ "$header" != "$base_header" ]]; then
      mismatch=true
      break
    fi
  done

  if [[ "$comparable_count" -le 1 ]]; then
    continue
  fi

  if [[ "$mismatch" == true ]]; then
    echo "$project_name" >> "$LOG_FILE"
  fi
done
