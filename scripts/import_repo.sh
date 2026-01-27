#!/usr/bin/env bash
# scripts/import_repo.sh
#
# Import (snapshot-copy) a GitHub ML repo into ./repos/<name>
# - Clones to temp dir
# - Copies files (no .git)
# - Writes SOURCE.md
# - Creates requirements.txt + venv if missing
# - Optionally commits
#
# Usage:
# ./scripts/import_repo.sh <repo_url> [project_name] [--commit] [--branch <branch>]

set -euo pipefail

die() { echo "Error: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

need_cmd git
need_cmd rsync
need_cmd python3

REPO_URL="${1:-}"
[[ -n "$REPO_URL" ]] || die "Missing <repo_url>"

PROJECT_NAME="${2:-}"
shift || true
shift || true

COMMIT=false
BRANCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) COMMIT=true; shift ;;
    --branch)
      shift
      [[ $# -gt 0 ]] || die "--branch requires a value"
      BRANCH="$1"
      shift
      ;;
    *) die "Unknown argument: $1" ;;
  esac
done

git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "Run from inside a git repository"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p repos

# Derive name from repo if not given
if [[ -z "$PROJECT_NAME" ]]; then
  base="${REPO_URL##*/}"
  base="${base%.git}"
  PROJECT_NAME="$base"
fi

TARGET_DIR="repos/$PROJECT_NAME"
[[ ! -e "$TARGET_DIR" ]] || die "Target exists: $TARGET_DIR"

TMP_DIR="$(mktemp -d -t import_repo_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "[*] Importing $REPO_URL → $TARGET_DIR"

# Clone upstream
cd "$TMP_DIR"
if [[ -n "$BRANCH" ]]; then
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" repo
else
  git clone --depth 1 "$REPO_URL" repo
fi

cd repo
UPSTREAM_COMMIT="$(git rev-parse HEAD)"
UPSTREAM_BRANCH="$(git rev-parse --abbrev-ref HEAD || true)"
IMPORT_DATE="$(date +%Y-%m-%d)"

# Copy snapshot
cd "$REPO_ROOT"
mkdir -p "$TARGET_DIR"

rsync -a \
  --exclude='.git' \
  --exclude='.github' \
  --exclude='.idea' \
  --exclude='.vscode' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  "$TMP_DIR/repo/" \
  "$TARGET_DIR/"

# ----------------------------
# Python environment bootstrap (venv + pipreqs)
# ----------------------------

REQ_CREATED=false
VENV_CREATED=false

VENV_DIR="$TARGET_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_PIPREQS="$VENV_DIR/bin/pipreqs"

# 1) Create virtual environment if missing
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[*] Creating virtual environment: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  VENV_CREATED=true
fi

# 2) Install pipreqs inside the venv (ensure pip exists + is reasonably fresh)
echo "[*] Installing/Updating pip + pipreqs inside venv"
"$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VENV_PIP" install --upgrade pip setuptools wheel >/dev/null
"$VENV_PIP" install --upgrade pipreqs >/dev/null

# 3) Generate requirements.txt using pipreqs (only if missing)
if [[ ! -f "$TARGET_DIR/requirements.txt" ]]; then
  echo "[*] Generating requirements.txt using pipreqs"
  # Note: ignore the venv folder so it doesn't pollute dependency detection
  "$VENV_PIPREQS" "$TARGET_DIR" \
    --force \
    --encoding utf-8 \
    --ignore "$VENV_DIR"

  REQ_CREATED=true
fi

# Create virtual environment if missing
if [[ ! -d "$TARGET_DIR/venv" ]]; then
  echo "[*] Creating virtual environment"
  python3 -m venv "$TARGET_DIR/venv"
  VENV_CREATED=true
fi

# ----------------------------
# SOURCE.md
# ----------------------------

cat > "$TARGET_DIR/SOURCE.md" <<EOF
# Source Information

Original repository:
$REPO_URL

Imported (snapshot) on:
$IMPORT_DATE

Upstream commit:
$UPSTREAM_COMMIT

Upstream branch:
$UPSTREAM_BRANCH

Environment bootstrap:
- requirements.txt: $( [[ "$REQ_CREATED" == true ]] && echo "created" || echo "already present" )
- virtualenv (venv): $( [[ "$VENV_CREATED" == true ]] && echo "created" || echo "already present" )

Notes:
- Imported as a one-time snapshot (no submodules, no upstream sync)
- Virtual environment is not auto-activated

Modifications:
- None (initial import)
EOF

# Safety check
find "$TARGET_DIR" -name ".git" -type d | grep -q . \
  && die "Nested .git directory detected"

echo "[+] Import complete"

# Optional commit
if [[ "$COMMIT" == true ]]; then
  git add "$TARGET_DIR"
  git commit -m "Import ML project: $PROJECT_NAME (snapshot + env)"
  echo "[+] Committed"
else
  echo "[i] Remember to:"
  echo "    git add $TARGET_DIR && git commit"
fi