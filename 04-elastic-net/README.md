# Session 4 · Ridge, Lasso, Elastic Net

**Getting it wrong on purpose, and getting a better answer.**

The first session where least squares does not merely perform badly but has no
unique solution at all. Regularisation, cross-validation, and the bias–variance
trade that makes a deliberately biased estimator the right choice.

## The puzzle

**The Secret Recipe.** 60 bakes, 200 candidate ingredients measured in grams, one
taste score. Name the ingredients that matter.

$p/n = 3.3$, so OLS interpolates: **training MSE is exactly zero** and test MSE
(36.23) exceeds the variance of the outcome (35.45). The model predicts worse
than the mean.

Then three estimators, and three separate lessons:

- **Ridge** keeps all 200 coefficients — 135 of them above 0.1 — and
  cross-validates to MSE 28.50. It structurally cannot produce a zero, so it
  gives you a prediction and no list.
- **Lasso** reaches MSE 9.90 and hands you 25 ingredients, six of the seven real
  ones plus nineteen false positives. The false positives are not a bug:
  cross-validation optimises *prediction*, and that penalty is smaller than the
  one that selects best.
- Two ingredients, `baking_soda` and `baking_powder`, are correlated at 0.963
  with **identical** true effects. The lasso zeroes one outright. The **elastic
  net** keeps both — but re-run it across five CV seeds and the *split* swings
  while the *sum* holds steady. Two collinear columns carry one identifiable
  quantity, and no penalty conjures a second.

And one planted signal — `dutch_cocoa`, true effect +1.1 — is not recoverable by
anything at this sample size. Noticing its absence is worth more than selecting
variables correctly.

## What you should be able to do afterwards

- Say what "no unique solution" means when $p > n$, and why zero training error
  is a symptom
- Decompose error into bias, variance and noise, and accept bias deliberately
- Fit ridge, lasso and elastic net, choosing the penalty by cross-validation
- Explain why the lasso produces exact zeros and ridge cannot
- Recognise when two collinear variables carry only one identifiable quantity
- Refuse to report a selected set as a list of true effects, or $p$-values on one

## The written treatment

[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/) (Warin),
chapter [5 · Regularisation: Ridge, Lasso, and the Elastic Net](https://warint.github.io/quantitative-methods/book/session-05.html),
with the bias–variance sections of chapter [8](https://warint.github.io/quantitative-methods/book/session-08.html).
Assigned in the pre-session; [`chapter.md`](chapter.md) follows their argument on
this session's data.

## Contents

| | |
|---|---|
| [`00-pre-session/`](00-pre-session/) | $p > n$, bias–variance, cross-validation. **60 min.** |
| [`01-lecture/`](01-lecture/) | The deck. Executes its own Python. |
| [`chapter.md`](chapter.md) | The lecture notes, as prose. |
| [`02-practice/`](02-practice/) | The workshop, a starter script, and `SOLUTION.md`. |

## Data

```python
import essentials as ess
r = ess.load("secret_recipe")     # 60 x 202
```
