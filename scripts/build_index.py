#!/usr/bin/env python3
"""Write docs/index.html — the landing page for the Pages site.

Generated rather than hand-maintained, because the session list, the deck paths
and the puzzle summaries all already exist in the repository. A hand-written
index would be a fourth place to update when a session changes, and the one
most likely to be forgotten.
"""

from __future__ import annotations

import html
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BOOK = "https://warint.github.io/quantitative-methods/book"

# The one-line puzzle summary and the trap, per session. Kept here rather than
# scraped from the READMEs: the READMEs phrase these over several sentences, and
# a landing page needs one line.
SESSIONS = [
    ("01-regression", "Regression",
     "The Roller Coaster Problem",
     "A law of physics hides in two columns. Find its exponent.",
     "A straight line fits at <em>R²</em> = 0.96 and reports a meaningless slope.",
     [("2", "session-02"), ("3", "session-03")]),
    ("02-advanced-regression", "Advanced Regression",
     "The Sourdough Optimum",
     "240 loaves, eight bakers. Hand in the recipe that maximises rise.",
     "Omit the baker effects and the answer is four points out, at <em>R²</em> = 0.90.",
     [("6", "session-06")]),
    ("03-logistic-regression", "Logistic Regression",
     "The Mushroom Forager",
     "Decide which ones you would eat.",
     "The accuracy-maximising threshold poisons you one meal in four.",
     [("4", "session-04")]),
    ("04-elastic-net", "Ridge, Lasso, Elastic Net",
     "The Secret Recipe",
     "Sixty bakes, two hundred ingredients, seven that matter.",
     "The lasso silently zeroes one of two identical ingredients.",
     [("5", "session-05"), ("8", "session-08")]),
    ("05-classification", "Classification",
     "The Bestiary",
     "Four species, and a decision boundary you can check against the truth.",
     "Blindness to a 2% species costs 1.7 points of accuracy.",
     [("8", "session-08")]),
    ("06-causal-inference-did", "Causal Inference",
     "The Cat Café Effect",
     "186 cafés adopt a resident cat on staggered dates. Did it work?",
     "Two-way fixed effects returns <strong>−2.09, <em>p</em> = 0.00001</strong> "
     "when every true effect is positive.",
     [("10", "session-10"), ("11", "session-11")]),
]

KINDS = [("00-pre-session", "Pre-session"),
         ("01-lecture", "Lecture"),
         ("02-practice", "Practice")]


def deck_href(session: str, kind: str) -> str | None:
    """The rendered deck for one session and kind, as a docs-relative path."""
    d = DOCS / session / kind
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*.html"))
    return f"{session}/{kind}/{hits[0].name}" if hits else None


def main() -> None:
    if not DOCS.is_dir():
        raise SystemExit("docs/ does not exist — run scripts/publish.sh first")

    cards = []
    for i, (slug, title, puzzle, brief, trap, chapters) in enumerate(SESSIONS, 1):
        links = []
        for kind, label in KINDS:
            href = deck_href(slug, kind)
            links.append(
                f'<a class="deck" href="{href}">{label}</a>' if href
                else f'<span class="deck missing">{label}</span>')
        ch = " · ".join(
            f'<a href="{BOOK}/{f}.html">ch.&nbsp;{n}</a>' for n, f in chapters)
        cards.append(f"""      <article class="card">
        <div class="num">{i}</div>
        <div class="body">
          <h2>{html.escape(title)}</h2>
          <p class="puzzle">{html.escape(puzzle)}</p>
          <p class="brief">{html.escape(brief)}</p>
          <p class="trap"><span class="tl">The trap</span> {trap}</p>
          <p class="links">{"".join(links)}</p>
          <p class="reading">Reading: {ch}</p>
        </div>
      </article>""")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Essentials of Data Science in Python</title>
<style>
  :root {{
    --paper: #F7F4ED; --ink: #222; --muted: #6B6B67;
    --red: #E3120B; --line: #D8D3C8; --blue: #356B8C;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--paper); color: var(--ink);
    font: 16px/1.55 'Helvetica Neue', Inter, system-ui, -apple-system, sans-serif;
  }}
  .wrap {{ max-width: 62rem; margin: 0 auto; padding: 3rem 1.5rem 5rem; }}
  header {{ border-bottom: 3px solid var(--red); padding-bottom: 1.4rem; margin-bottom: 2rem; }}
  h1 {{ font-size: clamp(1.7rem, 4vw, 2.5rem); margin: 0 0 .4rem; letter-spacing: -.02em; }}
  .sub {{ color: var(--muted); font-size: 1.05rem; margin: 0; }}
  .byline {{ margin-top: .9rem; font-size: .92rem; color: var(--muted); }}
  .lede {{ font-size: 1.05rem; max-width: 46rem; }}
  .lede strong {{ color: var(--red); }}
  .card {{
    display: flex; gap: 1.2rem; padding: 1.5rem 0;
    border-bottom: 1px solid var(--line);
  }}
  .num {{
    flex: 0 0 2.6rem; font-size: 2rem; font-weight: 700;
    color: var(--line); line-height: 1; text-align: right;
  }}
  .body {{ flex: 1 1 auto; min-width: 0; }}
  .card h2 {{ font-size: 1.22rem; margin: 0 0 .25rem; }}
  .puzzle {{ margin: 0; font-weight: 600; color: var(--red); font-size: .98rem; }}
  .brief {{ margin: .35rem 0 .5rem; color: var(--ink); }}
  .trap {{ margin: .1rem 0 .9rem; color: var(--muted); font-size: .93rem; }}
  .tl {{
    display: inline-block; font-size: .7rem; letter-spacing: .07em;
    text-transform: uppercase; font-weight: 700; color: var(--red);
    border: 1px solid var(--red); border-radius: 2px;
    padding: 0 .3rem; margin-right: .35rem; vertical-align: 1px;
  }}
  .links {{ margin: 0 0 .5rem; display: flex; flex-wrap: wrap; gap: .5rem; }}
  a.deck {{
    display: inline-block; text-decoration: none; color: var(--ink);
    border: 1px solid var(--ink); border-radius: 2px;
    padding: .3rem .8rem; font-size: .87rem; font-weight: 600;
    transition: background .12s, color .12s;
  }}
  a.deck:hover {{ background: var(--ink); color: var(--paper); }}
  .deck.missing {{
    display: inline-block; border: 1px dashed var(--line); border-radius: 2px;
    padding: .3rem .8rem; font-size: .87rem; color: var(--line);
  }}
  .reading {{ margin: 0; font-size: .85rem; color: var(--muted); }}
  .reading a {{ color: var(--blue); }}
  footer {{ margin-top: 2.5rem; font-size: .88rem; color: var(--muted); }}
  footer a {{ color: var(--blue); }}
  code {{
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    font-size: .88em; background: rgba(0,0,0,.045);
    padding: .1em .35em; border-radius: 2px;
  }}
  @media (max-width: 34rem) {{
    .card {{ flex-direction: column; gap: .3rem; }}
    .num {{ text-align: left; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Essentials of Data Science in Python</h1>
      <p class="sub">Six sessions, from a straight line to modern
         difference-in-differences.</p>
      <p class="byline">Thierry Warin, PhD · HEC Montréal</p>
    </header>

    <p class="lede">
      Every dataset in this course is generated locally from a fixed seed, and
      every one contains a <strong>planted answer</strong> you can check yourself
      against — plus a <strong>trap</strong>: a method most people reach for
      first, that reports a healthy fit statistic and gets the wrong answer.
      Finding the trap is the exercise.
    </p>

    <main>
{chr(10).join(cards)}
    </main>

    <footer>
      <p>
        Source, data generators and lecture notes:
        <a href="https://github.com/warint/essentials-data-science-python">github.com/warint/essentials-data-science-python</a><br>
        Companion volume:
        <a href="{BOOK}/">Quantitative Methods in International Business with Python</a>
      </p>
      <p>
        Decks are reveal.js — <code>→</code> and <code>←</code> to navigate,
        <code>Esc</code> for the overview, <code>?</code> for all shortcuts.
      </p>
      <p>
        Teaching materials CC BY-NC 4.0 · code MIT · © 2026 Thierry Warin
      </p>
    </footer>
  </div>
</body>
</html>
"""
    (DOCS / "index.html").write_text(page)
    n = sum(1 for s, *_ in SESSIONS for k, _ in KINDS if deck_href(s, k))
    print(f"docs/index.html written — {len(SESSIONS)} sessions, {n} decks linked")


if __name__ == "__main__":
    main()
