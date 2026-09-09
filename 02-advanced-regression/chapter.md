# 2 · Advanced Regression

> A model that explains ninety per cent of the variance, with every coefficient
> significant, recommending a recipe four percentage points wrong. Nothing in its
> own output says so.

## 2.1 The shapes a line cannot make

Session 1 fixed a curve with a logarithm. That works when the relationship is a
power law, and it is the wrong tool for three situations that come up constantly.

**A hill.** Rise increases with hydration and then decreases. There is a *best*
value in the middle. A linear model has no interior maximum, so its
recommendation is always at an edge of the data.

**A slope that depends on something else.** The best oven temperature may differ
for wet dough and dry dough. That is one effect *conditional on* another, not two
independent effects.

**A level that differs by group.** One baker's loaves rise 9 mm more than
another's for reasons no column records. If group membership also correlates with
your variable of interest, the estimate of that variable absorbs the difference.

This chapter is the three fixes — quadratics, interactions, fixed effects — and
the arithmetic of what each one does to the interpretation of everything else.

## 2.2 The naive recommendation

```python
import essentials as ess
import statsmodels.formula.api as smf

d = ess.load("sourdough")
flat = smf.ols("rise_mm ~ hydration_pct + oven_c", data=d).fit()
```

Both coefficients are positive, so this model's advice is "as wet and as hot as
possible": 95% hydration, 260 °C — the top-right corner of the design.

> **A recommendation at the boundary of your data is a fact about your model,
> not about the world.** It is the answer a plane always gives, because a plane
> has no peak.

## 2.3 Curves

Add a squared term:

$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon .$$

Differentiating and setting to zero puts the turning point at

$$ x^\star = -\frac{\beta_1}{2\beta_2},$$

a maximum when $\beta_2 < 0$.

```python
quad = smf.ols("rise_mm ~ hydration_pct + I(hydration_pct**2)", data=d).fit()
b1 = quad.params["hydration_pct"]
b2 = quad.params["I(hydration_pct ** 2)"]
-b1 / (2 * b2)      # 81.1 %
```

`I(...)` tells the formula parser to treat its contents as arithmetic; without
it, `**` is read as formula syntax.

### What the coefficients now mean

This is where interpretation goes wrong most often. Once $x^2$ is in the model,
the slope is not a constant:

$$ \frac{\partial y}{\partial x} = \beta_1 + 2\beta_2 x .$$

| at hydration | one more point of water is worth |
|---|---|
| 65% | +2.2 mm |
| 80% | −0.4 mm |
| 92% | −1.7 mm |

$\beta_1$ on its own is the slope **at $x = 0$** — a dough made entirely of
flour. It is a real number in the output and it answers no question anyone asked.

> There is no such thing as "the effect of hydration" in a quadratic model. There
> is only its effect *at a value*, and you have to choose which value to report.

## 2.4 Interactions

$$ y = \beta_0 + \beta_1 x + \beta_2 z + \beta_3 xz + \varepsilon
\qquad\Longrightarrow\qquad
\frac{\partial y}{\partial z} = \beta_2 + \beta_3 x $$

$\beta_3$ is the amount by which the effect of $z$ changes per unit of $x$. If it
is zero, the two variables act independently and you can give one recommendation
for each. **If it is not zero, you cannot.**

In a formula, `x:z` is the interaction alone and `x*z` expands to `x + z + x:z`.
Prefer `x*z`: an interaction without its main effects is almost never what you
meant, and it forces the surface through a constraint you did not intend.

On the sourdough data the interaction is +0.015 with $t = 17$. Solving for the
best temperature at a given hydration:

| hydration | best oven |
|---|---|
| 65% | 228.8 °C |
| 90% | 238.1 °C |

Nine degrees apart. Wetter dough wants a hotter oven, and an additive model
returns the same number for both.

### The interpretation trap

With an interaction present, $\beta_1$ is the effect of $x$ **when $z = 0$**. For
an oven temperature of zero degrees, that is not a quantity about baking. Two
standard responses:

- **Centre the variables.** Replace $z$ by $z - \bar z$, and $\beta_1$ becomes
  the effect of $x$ at the average $z$ — a value that exists.
- **Report conditional effects instead of coefficients.** Give the effect at two
  or three meaningful values of $z$, as the table above does. This is nearly
  always the clearer choice.

## 2.5 Fixed effects

Look at the bakers before fitting anything:

```python
d.groupby("baker")[["hydration_pct", "rise_mm"]].mean()
```

| baker | mean hydration | mean rise |
|---|---|---|
| A | 62.5 | 16.7 |
| D | 75.5 | **33.4** |
| G | 86.5 | 29.5 |
| H | **88.9** | 27.2 |

Baker H works closest to the true optimum and gets a worse rise than baker D. In
this data the bakers who prefer wet dough are the less skilled ones, so hydration
and skill move in opposite directions. A regression on hydration alone cannot
separate the two, and it will report the sum of the effect you want and the bias
you did not ask for.

`C(baker)` adds one dummy per baker (minus a reference level) so each gets their
own intercept. The remaining coefficients are then estimated **within** bakers:
from variation in hydration among that baker's own loaves, where skill is
constant because it is the same person.

| model | $R^2$ | AIC | hydration\* | error |
|---|---|---|---|---|
| linear + baker FE | 0.490 | 1819.4 | 95.00 | +8.38 pp |
| quadratics, no interaction | 0.919 | 1380.6 | 84.86 | −1.75 pp |
| interaction, no baker FE | 0.904 | 1411.1 | 82.41 | **−4.20 pp** |
| everything | 0.965 | 1185.5 | 86.69 | +0.07 pp |

The third row is the lesson of the chapter. **$R^2 = 0.904$ and the answer is
4.2 points wrong.** This is omitted variable bias, and the omitted variable was
not unobservable — it was in a column called `baker`.

### What fixed effects cost

They are not free, and they are not always right:

- **Parameters.** Seven for eight bakers. Affordable on 240 rows; not on 24.
- **Between-group information.** You can no longer ask whether bakers who work
  wetter do better, because that comparison is precisely what you removed.
- **Anything constant within a group.** If a baker always used the same flour,
  its effect and theirs are indistinguishable.

Fixed effects are the right tool when you want the within-group effect and
suspect the between-group comparison is contaminated. That is a judgement about
the data-generating process, not a default.

## 2.6 Report a region

A peak estimated from data is flat near the top:

```python
grid = pd.DataFrame({"hydration_pct": np.linspace(70, 95, 60),
                     "oven_c": 236.9, "baker": "D"})
full.get_prediction(grid).summary_frame()
```

Anything from roughly 84% to 89% is indistinguishable from the maximum. Quoting
86.69% implies a precision the data does not support. **The range is the finding.**

## 2.7 Comparing models

$$ \mathrm{AIC} = -2\log L + 2k $$

Fit, penalised for parameters. Lower is better, and only ever *relative* — an AIC
of 1,185 means nothing on its own.

For **nested** models — one is the other plus terms — use an F-test:

```python
f_stat, p_val, _ = restricted.compare_f_test(full)
```

For **non-nested** models, AIC or BIC is what you have. BIC penalises more
heavily ($k\log n$ rather than $2k$) and prefers simpler models as $n$ grows.

Two rules that are broken constantly:

1. **Compare only on identical rows.** Drop a missing value in one model and the
   likelihoods are computed on different data. The comparison is void, and
   nothing in the output will tell you.
2. **A model chosen by AIC was chosen by you.** Reporting its $p$-values as
   though you had not searched is the quiet misuse. The search used up some of
   the evidence, and the standard errors do not know about it.

## 2.8 Summary

1. A linear model cannot report an interior optimum. Edge answers are model
   artefacts.
2. With $x^2$ present, $\beta_1$ is the slope at zero. With an interaction
   present, $\beta_1$ is the effect when the other variable is zero. Both are
   usually meaningless. Report conditional effects instead.
3. An interaction means you cannot give one recommendation.
4. Fixed effects buy the within-group effect and cost parameters, between-group
   information, and anything constant within a group.
5. A model can explain 90% of the variance and be four points wrong, with nothing
   in its own output to warn you. Fit statistics are not diagnostics.

## Further reading

- Wooldridge (2019), *Introductory Econometrics*, 7th ed., ch. 6, 7, 14.
- Angrist & Pischke (2009), *Mostly Harmless Econometrics*, ch. 3.
- Brambor, Clark & Golder (2006), "Understanding interaction models",
  *Political Analysis* 14(1), 63–82.
- Kutner et al. (2005), *Applied Linear Statistical Models*, 5th ed., ch. 8.
