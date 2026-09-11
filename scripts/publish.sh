#!/usr/bin/env bash
#
# Build the GitHub Pages site into docs/.
#
#     scripts/publish.sh
#
# Why this is not just `cp *.html docs/`
# -------------------------------------
# The decks are authored with `embed-resources: true`, so each one inlines
# reveal.js, its fonts and its figures and comes out around 3.3 MB. That is a
# feature for a deck you download and open on a train with no wifi, and it is
# the wrong thing to commit eighteen times: 61 MB of near-identical binary
# payload, added again on every re-render, in a repository students clone.
#
# So this script stages the sources into a scratch directory, turns embedding
# off there, and renders them as one Quarto *website* project — which shares a
# single copy of reveal.js across all eighteen decks. The result is a few
# megabytes instead of sixty, and the committed sources are untouched.
#
# Requires Quarto: https://quarto.org/docs/download/

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

if ! command -v quarto >/dev/null 2>&1; then
    echo "Quarto not found. Install it from https://quarto.org/docs/download/" >&2
    exit 1
fi

if [ -z "${QUARTO_PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
    export QUARTO_PYTHON="$ROOT/.venv/bin/python"
fi

echo "Staging sources in $STAGE"
cp -R theme "$STAGE/"
cp essentials.py "$STAGE/"
for deck in [0-9][0-9]-*/0[0-2]-*/*.qmd; do
    mkdir -p "$STAGE/$(dirname "$deck")"
    # Turn embedding off so the project shares one copy of reveal.js.
    sed 's/^\( *\)embed-resources: true$/\1embed-resources: false/' \
        "$deck" > "$STAGE/$deck"
done

# The lecture decks import essentials and the plot style by relative path, and
# they resolve those from their own directory — so the staging tree has to have
# the same shape as the repository. It does, because we copied it that way.
cat > "$STAGE/_quarto.yml" <<'YML'
project:
  type: website
  output-dir: _site
  render:
    - "*/0*/*.qmd"

website:
  title: "Essentials of Data Science in Python"

format:
  revealjs:
    embed-resources: false
YML

echo "Rendering the site"
( cd "$STAGE" && quarto render )

echo "Copying into docs/"
rm -rf "$ROOT/docs"
mkdir -p "$ROOT/docs"
cp -R "$STAGE/_site/." "$ROOT/docs/"

# GitHub Pages runs Jekyll unless told not to, and Jekyll ignores directories
# whose names begin with an underscore — which is where Quarto puts its
# libraries. Without this file the site loads with no CSS and no reveal.js.
touch "$ROOT/docs/.nojekyll"

# The per-session Python cheatsheets, which the landing page links to by name.
# Built separately by scripts/build_cheatsheets.py — rendering them needs LaTeX,
# which is a heavier dependency than the decks have, so this only copies what is
# already there rather than building it.
sheets=0
for src in [0-9][0-9]-*/PYTHON-CHEATSHEET.pdf; do
    [ -e "$src" ] || continue
    cp "$src" "$ROOT/docs/python-cheatsheet-${src:0:2}.pdf"
    sheets=$((sheets + 1))
done
if [ "$sheets" -eq 0 ]; then
    echo "No cheatsheets found — run: python3 scripts/build_cheatsheets.py" >&2
else
    echo "Copied $sheets python cheatsheets into docs/"
fi

python3 "$ROOT/scripts/build_index.py"

echo
echo "Site built:  docs/  ($(du -sh "$ROOT/docs" | cut -f1))"
echo "Preview it:  python3 -m http.server -d docs 8000"
echo
echo "GitHub Pages must be pointed at it once, by hand:"
echo "  Settings -> Pages -> Source: 'Deploy from a branch'"
echo "                      Branch: main,  folder: /docs"
