#!/usr/bin/env bash
#
# Render every deck to self-contained reveal.js HTML.
#
#     scripts/render.sh          # all eighteen decks
#     scripts/render.sh 04       # just session 04
#
# Requires Quarto: https://quarto.org/docs/download/
#
# The lecture decks execute their own Python at render time, so the numbers on
# the slides are the numbers a student's machine produces. That is the point of
# doing it this way: a slide cannot drift away from the code that made it.

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v quarto >/dev/null 2>&1; then
    echo "Quarto not found. Install it from https://quarto.org/docs/download/" >&2
    exit 1
fi

if [ -z "${QUARTO_PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
    export QUARTO_PYTHON="$PWD/.venv/bin/python"
fi
if [ -n "${QUARTO_PYTHON:-}" ]; then
    echo "Executing deck code with: $QUARTO_PYTHON"
    if ! "$QUARTO_PYTHON" -c "import pandas, sklearn, statsmodels, matplotlib" 2>/dev/null; then
        echo "  WARNING: that interpreter is missing packages the figures need." >&2
        echo "           pip install -r requirements.txt" >&2
    fi
fi

filter="${1:-}"
found=0
for deck in [0-9][0-9]-*/0[0-2]-*/*.qmd; do
    [ -e "$deck" ] || continue
    if [ -n "$filter" ] && [[ "$deck" != *"$filter"* ]]; then continue; fi
    found=1
    echo "--> $deck"
    quarto render "$deck" --to revealjs
done

if [ "$found" -eq 0 ]; then
    echo "No decks matched '${filter}'." >&2
    exit 1
fi
echo "Done."
