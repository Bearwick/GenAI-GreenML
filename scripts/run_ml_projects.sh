#!/usr/bin/env bash
set -euo pipefail

REPOS_DIR="repos"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$RESULTS_DIR/results_$TIMESTAMP.csv"

echo "timestamp,project,script,status,accuracy,mem_delta_mb,exec_time_s,energy_j,notes" > "$LOG_FILE"

OS_NAME="$(uname -s)"
MACOS_AVG_POWER_W=20

# ---------------- PYTHON PICKER ----------------
pick_python() {
  local project_dir="$1"
  local venv_py="$project_dir/venv/bin/python"
  if [[ -x "$venv_py" ]]; then
    if "$venv_py" -c "import sys" >/dev/null 2>&1; then
      echo "$venv_py"
      return
    fi
  fi
  echo "python3"
}

# ---------------- ENERGY (Linux only, Intel RAPL) ----------------
read_energy_linux() {
  local total=0
  shopt -s nullglob
  for f in /sys/class/powercap/intel-rapl:*/energy_uj; do
    val=$(cat "$f" 2>/dev/null || echo 0)
    total=$((total + val))
  done
  [[ "$total" -gt 0 ]] && echo "$total" || echo ""
}

# ---------------- MEMORY POLLING ----------------
poll_memory() {
  local pid="$1"
  local peak=0
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{print $1}')
    [[ -n "$rss" && "$rss" -gt "$peak" ]] && peak="$rss"
    sleep 0.005
  done
  echo "$peak"
}

# ---------------- MAIN LOOP ----------------
for project in "$REPOS_DIR"/*; do
  [[ ! -d "$project" ]] && continue

  project_name=$(basename "$project")

  script_files=()
  while IFS= read -r -d '' script_path; do
    script_files+=("$script_path")
  done < <(find "$project" -maxdepth 1 -type f -name "GENAIGREENML*.py" -print0)

  if [[ ${#script_files[@]} -eq 0 ]]; then
    echo "$(date -Iseconds),$project_name,GENAIGREENML*.py,SKIPPED,,,$$,," >> "$LOG_FILE"
    continue
  fi

  for script_path in "${script_files[@]}"; do
    script="$(basename "$script_path")"

    echo "▶ Running $project_name / $script"

    if [[ -f "$project/requirements.txt" && -x "$project/venv/bin/pip" ]]; then
      "$project/venv/bin/pip" install -r "$project/requirements.txt" >/dev/null 2>&1 || \
        echo "[!] pip install failed for $project_name"
    fi

    # Energy before
    ENERGY_BEFORE=""
    [[ "$OS_NAME" == "Linux" ]] && ENERGY_BEFORE=$(read_energy_linux)

    START_TIME=$(date +%s.%N)
    OUTPUT_FILE="$(mktemp)"

    (
      cd "$project"
      if [[ -x "venv/bin/python" ]]; then
        "venv/bin/python" "$script" >"$OUTPUT_FILE" 2>&1
      else
        echo "[!] venv python missing for $project_name" >"$OUTPUT_FILE"
        exit 1
      fi
    ) &
    PID=$!

    BASELINE_RSS=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{print $1}' || echo 0)

    PEAK_RSS=$(poll_memory "$PID")

    wait "$PID"
    EXIT_CODE=$?

    END_TIME=$(date +%s.%N)

    EXEC_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    MEM_DELTA_MB=$(echo "scale=3; ($PEAK_RSS - $BASELINE_RSS)/1024" | bc)
    [[ "$MEM_DELTA_MB" == "-"* ]] && MEM_DELTA_MB=0

    ENERGY_J=""
    ACCURACY=""
    NOTES=""

    if [[ "$OS_NAME" == "Linux" && -n "$ENERGY_BEFORE" ]]; then
      ENERGY_AFTER=$(read_energy_linux)
      if [[ -n "$ENERGY_AFTER" ]]; then
        ENERGY_J=$(echo "scale=6; ($ENERGY_AFTER - $ENERGY_BEFORE)/1000000" | bc)
      else
        NOTES="energy not available"
      fi
    else
      # Approximate energy on macOS using a constant average power draw.
      ENERGY_J=$(echo "scale=3; $EXEC_TIME * $MACOS_AVG_POWER_W" | bc)
      NOTES="energy approx on macOS using ${MACOS_AVG_POWER_W}W"
    fi

    STATUS="OK"
    [[ "$EXIT_CODE" -ne 0 ]] && STATUS="FAILED"

    if [[ -s "$OUTPUT_FILE" ]]; then
      ACCURACY=$(awk -F= '/^ACCURACY=/{val=$2} END{print val}' "$OUTPUT_FILE")
    fi

    echo "$(date -Iseconds),$project_name,$script,$STATUS,$ACCURACY,$MEM_DELTA_MB,$EXEC_TIME,$ENERGY_J,$NOTES" >> "$LOG_FILE"
    rm -f "$OUTPUT_FILE"
  done
done

echo "✅ Finished. Results written to $LOG_FILE"
