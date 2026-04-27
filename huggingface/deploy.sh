#!/usr/bin/env bash
# Deploy this repo na Hugging Face Space.
# Wywołanie:
#   ./huggingface/deploy.sh <hf_username>/<space_name>
# np.
#   ./huggingface/deploy.sh jjbiernacki/pdf-map-parcels
#
# Wymagania:
#   - Space (Docker SDK, blank) musi już istnieć — utwórz na
#     https://huggingface.co/new-space
#   - Zalogowany hf CLI:  .venv/bin/hf auth login
#     (token z https://huggingface.co/settings/tokens, scope "write")
set -euo pipefail

if [[ $# -ne 1 || "$1" != */* ]]; then
  echo "Usage: $0 <user>/<space>" >&2
  exit 1
fi
SPACE="$1"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo ">> Klonuję pusty Space → $WORK/space"
git clone "https://huggingface.co/spaces/${SPACE}" "$WORK/space"

cd "$WORK/space"

echo ">> Kopiuję kod z $SRC_DIR"
rsync -a \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.claude/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='rendered.png' \
  --exclude='crop*.png' \
  --exclude='debug_*.png' \
  --exclude='tmp/' \
  "$SRC_DIR/" ./

echo ">> Wystawiam Dockerfile + README do roota Space'a"
cp huggingface/Dockerfile ./Dockerfile
cp huggingface/README.md  ./README.md
cp huggingface/.gitattributes ./.gitattributes 2>/dev/null || true

git add -A
if git diff --cached --quiet; then
  echo ">> Brak zmian — nic do pushowania."
  exit 0
fi
git commit -m "Deploy from pdf-map-parcels"
echo ">> Push do HF Space"
git push

echo
echo "✓ Deploy poszedł. URL:"
echo "  https://huggingface.co/spaces/${SPACE}"
echo
echo "Build trwa 5–8 min (instalacja torcha + pre-warm EasyOCR)."
echo "Postęp: zakładka 'Logs' w UI Space'a."
