# Session 5 · Classification

**A boundary you can check against the truth.**

The only session where the right answer is visible. The data is generated from
known distributions, so the best achievable accuracy can be computed in advance
and the true decision boundary can be drawn underneath your own.

## The puzzle

**The Bestiary.** 1,600 creatures, four species, five measurements — two that
carry information and three that are noise.

- The **Bayes floor is 0.8625**. No classifier can beat it, because the species
  overlap and 7% of field identifications are wrong. A test accuracy of 0.98
  would be a bug report.
- The **bias–variance trade becomes a picture**: at $k=1$ the boundary shatters
  into islands, each a single mislabelled creature (train 0.999, test 0.762); at
  $k=400$ it is smooth and in the wrong place; at $k \approx 45$ it tracks the
  true boundary and lands at the floor.
- One species, `ember_drake`, is **2.1% of the data**. A classifier that never
  predicts it at all loses **1.7 percentage points** of accuracy. Anything
  optimising accuracy has almost no reason to find it.
- The three **noise columns** cost nearly three points at the optimal $k$,
  because distance in an irrelevant dimension is still distance — and $k$-NN has
  no coefficients, so it cannot ignore a feature you give it.

## What you should be able to do afterwards

- Compute or estimate a ceiling, and use it to detect leakage rather than to celebrate
- Fit $k$-NN, choose $k$ by cross-validation on training data only
- Draw a decision boundary, and read what it is fitting
- Read a multi-class confusion matrix and say what accuracy concealed
- Choose between precision and recall from the problem
- Explain why irrelevant features punish distance methods

## The written treatment

[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/) (Warin),
chapter [8 · K-Nearest Neighbours and the Bias–Variance Trade-off](https://warint.github.io/quantitative-methods/book/session-08.html).
Assigned in the pre-session; [`chapter.md`](chapter.md) follows its argument on
this session's data.

## Contents

| | |
|---|---|
| [`00-pre-session/`](00-pre-session/) | The Bayes classifier, $k$-NN by hand. **60 min.** |
| [`01-lecture/`](01-lecture/) | The deck. Executes its own Python. |
| [`chapter.md`](chapter.md) | The lecture notes, as prose. |
| [`02-practice/`](02-practice/) | The workshop, a starter script, and `SOLUTION.md`. |

## Data

```python
import essentials as ess
d = ess.load("bestiary")                                  # 1,600 rows
ess.bestiary_posterior(wing_span, tail_length)             # the true posterior
ess.bestiary_species()                                     # column order
```
