# Session 2 · Advanced Regression

**Curves, interactions, and the groups you did not measure.**

Three extensions to the linear model, and the arithmetic of what each one does to
the interpretation of everything else. The session is built so that all three are
needed at once: leave any one out and the answer is confidently wrong.

## The puzzle

**The Sourdough Optimum.** 240 loaves from eight bakers. Hand in a recipe — one
hydration percentage and one oven temperature — with an honest interval.

The planted optimum is **86.6% hydration at 236.9 °C**, an interior point. The
traps stack:

- A **linear** model recommends 95% and 260 °C, the corner of the design, because
  a plane has no peak.
- Adding **quadratics** gets you to 81%, better and still wrong.
- Adding the **interaction** reveals that wetter dough wants a hotter oven —
  228.8 °C at 65% hydration against 238.1 °C at 90%.
- The last piece is the **bakers**. Those who prefer wet dough are the less
  skilled ones, so hydration and skill are confounded. The model without baker
  fixed effects reports $R^2 = 0.904$, significance everywhere, and an answer
  4.2 percentage points wrong.

That last number is the point of the session: a fit statistic is not a
diagnostic.

## What you should be able to do afterwards

- Fit a quadratic and locate its optimum from the coefficients
- Say what a main effect means once a squared term or interaction is present
- Report conditional effects rather than coefficients when an interaction exists
- Add fixed effects, explain what changed, and state what they cost
- Compare nested models with an F-test and non-nested ones with AIC
- Report an optimum as a region

## The written treatment

[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/) (Warin),
chapter [6 · Regression: Advanced Considerations](https://warint.github.io/quantitative-methods/book/session-06.html).
Assigned in the pre-session; [`chapter.md`](chapter.md) follows its argument on
this session's data.

## Contents

| | |
|---|---|
| [`00-pre-session/`](00-pre-session/) | Load the data, three shapes, self-check. **45 min.** |
| [`01-lecture/`](01-lecture/) | The deck. Executes its own Python. |
| [`chapter.md`](chapter.md) | The lecture notes, as prose. |
| [`02-practice/`](02-practice/) | The workshop, a starter script, and `SOLUTION.md`. |
| [`PYTHON-CHEATSHEET.pdf`](PYTHON-CHEATSHEET.pdf) | Sessions 1 to 2, on a page: the environment, and every name so far. |

## Data

```python
import essentials as ess
d = ess.load("sourdough")            # 240 rows
ess.sourdough_optimum()              # the planted answer, solved from the coefficients
```
