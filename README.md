# Essentials of Data Science in Python

Six sessions on the methods that carry most applied quantitative work: linear
regression and its diagnostics, interactions and fixed effects, logistic
regression, regularisation, classification, and causal inference with modern
difference-in-differences.

**Thierry Warin, PhD** · HEC Montréal

---

## What makes this course different

Every dataset here is **generated from a fixed seed by
[`essentials.py`](essentials.py)**, and every one of them contains a **planted
answer** — a physical constant, an optimal recipe, a list of seven ingredients
that matter, a set of treatment effects. You are not asked to produce a
defensible-sounding number. You are asked to recover a specific one, and you can
check whether you did.

Each dataset also contains a **trap**: a method that most people reach for
first, that reports a healthy fit statistic, and that gets the wrong answer.
Finding the trap *is* the exercise. In session 6 the trap is an estimator that
returns the wrong sign with a *p*-value of 0.00001.

Three consequences follow from generating rather than downloading:

- **It runs with the wifi off.** A ninety-minute workshop that opens with two
  hundred people hitting the same server opens badly.
- **Everyone has the same bytes.** Your Cook's distance and your neighbour's are
  the same number, so a disagreement is about the analysis — the only kind worth
  having in class.
- **The answers are testable.** `scripts/verify_puzzles.py` asserts all 39
  planted properties. Change a coefficient in the generator and the suite tells
  you which lesson you just destroyed.

## The six sessions

| | Session | The method | The puzzle | The trap |
|---|---|---|---|---|
| 1 | [Regression](01-regression/) | OLS, residuals, leverage, influence | **The Roller Coaster Problem** — a law of physics is hiding in two columns. Find its exponent. | A straight line fits at *R²* = 0.96 and reports a meaningless slope |
| 2 | [Advanced regression](02-advanced-regression/) | Non-linearity, interactions, fixed effects | **The Sourdough Optimum** — find the recipe that maximises rise | Omitting baker effects moves the answer four points of hydration, at *R²* = 0.90 |
| 3 | [Logistic regression](03-logistic-regression/) | Binary and ordinal logit, odds ratios, ROC | **The Mushroom Forager** — decide which ones you would eat | The accuracy-maximising threshold poisons you once every four meals |
| 4 | [Elastic net](04-elastic-net/) | Ridge, lasso, cross-validation, bias–variance | **The Secret Recipe** — 60 bakes, 200 ingredients, 7 that matter | The lasso silently drops one of two identical ingredients |
| 5 | [Classification](05-classification/) | *k*-NN, decision boundaries, confusion matrices | **The Bestiary** — four species, and a boundary you can check against the truth | Accuracy hides a 2% class entirely; three noise columns cost real performance |
| 6 | [Causal inference](06-causal-inference-did/) | DiD, Goodman-Bacon, Callaway–Sant'Anna, honest DiD | **The Cat Café Effect** — 186 cafés, staggered adoption. Did the cat work? | Two-way fixed effects returns **−2.09, *p* = 0.00001**, when every true effect is positive |

Each session ships three things:

- `00-pre-session/` — what to do before class, and roughly how long it takes.
- `01-lecture/` — the deck, plus `chapter.md`: prose notes that read on their own.
- `02-practice/` — a ninety-minute workshop, a starter script, and the answer key.

## Getting started

```bash
git clone https://github.com/warint/essentials-data-science-python.git
cd essentials-data-science-python

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/verify_puzzles.py   # 39 checks; all should pass
```

Then, in Python:

```python
import essentials as ess

ess.catalog()                      # every dataset, and what it asks of you
coasters = ess.load("coasters")    # built locally in under a second, then cached
```

Nothing is downloaded. `load()` builds the frame from its seed on first call and
caches it as parquet in `data/`, which is git-ignored because it is reproducible
rather than precious.

## Prerequisites

Session 1's [pre-session](01-regression/00-pre-session/) carries the setup:
installing Python, making a virtual environment, enough pandas to load and shape
a frame, and enough inference to read a standard error. Budget two hours for it
once. Everything after that assumes it.

You do not need prior statistics. You do need to be willing to be wrong in
public, because five of the six sessions are built around a method that fails.

## Rendering the decks

```bash
scripts/render.sh        # all decks
scripts/render.sh 06     # just session 6
```

Requires [Quarto](https://quarto.org/docs/download/). The lecture decks execute
their own Python at render time, so the numbers on the slides are the numbers
your machine produces — a slide cannot drift away from the code that made it.

## Repository layout

```
essentials.py              every dataset, and the planted answer inside it
scripts/verify_puzzles.py  39 assertions that the answers are still there
scripts/render.sh          build the decks
theme/                     slide theme and the shared matplotlib palette
NN-session-name/
  README.md                what this session is for
  chapter.md               the lecture notes, as prose
  00-pre-session/          before class
  01-lecture/              the deck
  02-practice/             the workshop, a starter script, and SOLUTION.md
```

## A standing rule

Every deliverable must document **one instance where you identified a model
output as wrong, unverifiable or misleading, and how you established that**. It
applies to language models and to your own regressions equally. You are never
penalised for using a tool. You are penalised for using it uncritically — which
is, more or less, what this whole course is about.

## Licence

Teaching materials under [CC BY-NC 4.0](LICENSE); code under the
[MIT licence](LICENSE-CODE). Both © 2026 Thierry Warin.
