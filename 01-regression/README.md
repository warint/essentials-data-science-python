# Session 1 · Regression

**Fitting a line, and finding out whether to believe it.**

The session that everything else rests on. It fits a single-variable linear
model, reads every column of its output, and then spends most of its time on the
question that matters more: is this model the right *shape*?

## The puzzle

**The Roller Coaster Problem.** 84 coasters, with the height of the first drop
and the top speed. Two of those columns are related by a law of physics of the
form $v = c\,h^{k}$, and the task is to report $k$.

The trap is $R^2$. A straight line fits the gravity-driven coasters at
$R^2 = 0.96$ and reports a slope that corresponds to nothing. The residuals give
it away immediately — they arch, changing sign twice across the range — and
taking logs turns the power law into a line whose slope is the exponent. The
answer is 0.5, and the recovered interval is [0.484, 0.511].

Then six rows turn out to be running the regression. They are launched coasters,
driven by motors rather than gravity, and keeping them moves the exponent to
0.37. Cook's distance finds them; the `propulsion` column explains them.

## What you should be able to do afterwards

- Fit from a formula and read a coefficient in its units, with its uncertainty
- Say what a residuals-versus-fitted plot shows, and what you looked for
- Compute leverage, and show that it sums to $p$
- Separate an outlier from an influential point with Cook's distance
- Recognise a curved relationship, and fix it with a transformation
- Distrust $R^2$ as evidence of correctness

## The written treatment

[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/) (Warin),
chapters [2 · Exploratory Data Analysis, and the First Model](https://warint.github.io/quantitative-methods/book/session-02.html)
and [3 · Regression: Adequacy, Validity, and Robustness](https://warint.github.io/quantitative-methods/book/session-03.html).
Assigned in the pre-session; [`chapter.md`](chapter.md) follows their argument on
this session's data.

## Contents

| | |
|---|---|
| [`00-pre-session/`](00-pre-session/) | Setup, enough pandas, enough inference. **Two hours** — this is the long one. |
| [`01-lecture/`](01-lecture/) | The deck. Executes its own Python, so the slide numbers are your numbers. |
| [`chapter.md`](chapter.md) | The lecture notes, as prose. Reads on its own. |
| [`02-practice/`](02-practice/) | The 90-minute workshop, a starter script, and `SOLUTION.md`. |
| [`PYTHON-CHEATSHEET.pdf`](PYTHON-CHEATSHEET.pdf) | Everything in this session, on a page. Print it. |

## Data

```python
import essentials as ess
d = ess.load("coasters")     # 84 rows, 9 columns, built locally
```
