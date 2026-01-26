#!/usr/bin/env bash
# import_repo.sh
#
# Import (snapshot-copy) a GitHub ML repo into your dataset repo under ./projects/<name>
# - Clones to a temp dir
# - Copies files (excluding .git and common junk)
# - Writes SOURCE.md with provenance + optional license hint
# - Optionally commits to your dataset repo
#
# Usage:
#   ./import_repo.sh <repo_url> [project_name] [--commit] [--branch <branch>] [--keep-tmp]
#
# Examples:
#   ./import_repo.sh https://github.com/akshaybahadur21/BreastCancer_Classification.git breast_cancer_classification --commit
#   ./import_repo.sh https://github.com/user/repo.git --branch main
#
set -euo pipefail

die() { echo "Error: $*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

need_cmd git
need_cmd rsync

REPO_URL="${1:-}"
[[ -n "$REPO_URL" ]] || die "Missing <repo_url>. Usage: ./import_repo.sh <repo_url> [project_name] [--commit] [--branch <branch>] [--keep-tmp]"

PROJECT_NAME="${2:-}"
shift || true
shift || true

COMMIT=false
KEEP_TMP=false
BRANCH=""

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) COMMIT=true; shift ;;
    --keep-tmp) KEEP_TMP=true; shift ;;
    --branch)
      shift
      [[ $# -gt 0 ]] || die "--branch requires a value"
      BRANCH="$1"
      shift
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

# Must be run from the dataset repo root (where you want ./projects/)
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Run this from inside your dataset git repository."

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p repos

# Derive project name if not provided
if [[ -z "$PROJECT_NAME" ]]; then
  # Strip trailing .git and take last path segment
  base="${REPO_URL##*/}"
  base="${base%.git}"
  # Basic slugify: lowercase, spaces->-, remove odd chars
  PROJECT_NAME="$(echo "$base" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9._-')"
fi

TARGET_DIR="repos/$PROJECT_NAME"
[[ ! -e "$TARGET_DIR" ]] || die "Target already exists: $TARGET_DIR (choose a different project_name or remove the folder)"

TMP_DIR="$(mktemp -d -t import_repo_XXXXXX)"
cleanup() {
  if [[ "$KEEP_TMP" == false ]]; then
    rm -rf "$TMP_DIR"
  else
    echo "Keeping temp dir: $TMP_DIR"
  fi
}
trap cleanup EXIT

echo "[*] Repo root:   $REPO_ROOT"
echo "[*] Importing:   $REPO_URL"
echo "[*] Project dir: $TARGET_DIR"
echo "[*] Temp dir:    $TMP_DIR"

# Clone
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

# Copy snapshot (exclude .git and typical junk)
cd "$REPO_ROOT"
mkdir -p "$TARGET_DIR"

rsync -a \
  --exclude='.git' \
  --exclude='.github' \
  --exclude='.idea' \
  --exclude='.vscode' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.pyd' \
  --exclude='.DS_Store' \
  --exclude='*.ipynb_checkpoints' \
  --exclude='node_modules' \
  "$TMP_DIR/repo/" \
  "$TARGET_DIR/"

# Try to detect license file name (simple hint)
LICENSE_HINT="Unknown (check upstream)"
if [[ -f "$TARGET_DIR/LICENSE" ]]; then
  LICENSE_HINT="LICENSE file present (verify type)"
elif [[ -f "$TARGET_DIR/LICENCE" ]]; then
  LICENSE_HINT="LICENCE file present (verify type)"
elif [[ -f "$TARGET_DIR/LICENSE.md" ]]; then
  LICENSE_HINT="LICENSE.md present (verify type)"
elif [[ -f "$TARGET_DIR/COPYING" ]]; then
  LICENSE_HINT="COPYING present (verify type)"
fi

# Write provenance
cat > "$TARGET_DIR/SOURCE.md" <<EOF
# Source Information

Original repository:
$REPO_URL

Imported (snapshot) on:
$IMPORT_DATE

Upstream commit (at import):
$UPSTREAM_COMMIT

Upstream branch (at import):
$UPSTREAM_BRANCH

License:
$LICENSE_HINT

Notes:
- This project was imported as a one-time snapshot (no submodules; no upstream syncing).
- Record any modifications below.

Modifications:
- None (initial import)
EOF

# Safety check: ensure no nested git metadata
if find "$TARGET_DIR" -name ".git" -type d | grep -q .; then
  die "Nested .git directory found inside $TARGET_DIR. Import aborted."
fi

echo "[+] Imported snapshot into $TARGET_DIR"
echo "[+] Wrote $TARGET_DIR/SOURCE.md"

# Optional: git add/commit
if [[ "$COMMIT" == true ]]; then
  cd "$REPO_ROOT"
  git add "$TARGET_DIR"
  git commit -m "Import ML project: $PROJECT_NAME (snapshot)" || {
    echo "[!] Nothing to commit (or commit failed)."
  }
  echo "[+] Committed. Now run: git push"
else
  echo "[i] Next steps:"
  echo "    git add $TARGET_DIR"
  echo "    git commit -m \"Import ML project: $PROJECT_NAME (snapshot)\""
  echo "    git push"
fi
