# 2 · Advanced Regression

> A model that explains ninety per cent of the variance, with every coefficient
> significant, recommending a recipe four percentage points wrong. Nothing in its
> own output says so.

## 2.1 One relationship, or eight?

Chapter 1 fitted a single line and reported one slope. That slope was an answer
to a question nobody had asked precisely: *whose* relationship is it?

The sourdough data is 240 loaves from eight bakers. Fit one line through all of
them and you get a slope of **+0.601** millimetres of rise per point of
hydration. Now fit the same line separately within each baker:

| baker | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| mean hydration | 62.5 | 64.3 | 71.4 | 75.5 | 74.7 | 81.3 | 86.5 | 88.9 |
| within-baker slope | **+2.62** | +1.98 | +1.23 | +1.51 | +1.20 | +0.40 | **−0.52** | **−0.48** |

```python
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("sourdough")

pooled = smf.ols("rise_mm ~ hydration_pct", data=d).fit()
print(round(pooled.params["hydration_pct"], 3))          # 0.601

slopes = {b: smf.ols("rise_mm ~ hydration_pct", data=g).fit().params["hydration_pct"]
          for b, g in d.groupby("baker")}
print({b: round(float(s), 2) for b, s in slopes.items()})   # A 2.62 ... H -0.48
print(d.groupby("baker").hydration_pct.mean().round(1).to_dict())
print(round(sum(slopes.values()) / len(slopes), 2))      # 0.99
```

Eight bakers, eight different slopes, **two of them negative**. The pooled slope
is not the average of the eight (that average is +0.99). It is a different
quantity, dominated by a comparison *between* bakers rather than a description of
what happens *within* one.

> Whether the pooled slope is the quantity you want depends entirely on the claim
> you intend to make. "Bakers who work wetter get less rise" is a between-baker
> claim, and the pooled estimate is the right object for it. "Adding water to
> *this* dough increases rise" is a within-baker claim, and the pooled estimate
> contains almost none of the variation that would answer it.

### And the eight slopes are one curve

Before reaching for panel machinery, notice what those eight numbers actually
are. The slope falls monotonically with the baker's mean hydration — +2.62 for
the driest, −0.48 for the wettest — and it crosses zero somewhere between bakers
F and G, whose mean hydrations are 81.3 and 86.5.

That is not eight relationships. It is **one quadratic, observed through eight
narrow windows**. Each baker's "slope" is the tangent to a single response
surface at the hydration where that baker happens to work, and the sign flips
because bakers G and H work past the peak. The planted optimum, as chapter 2's
puzzle establishes, is 86.6% hydration — right where the slopes change sign.

> Heterogeneous slopes are sometimes genuine heterogeneity and sometimes a
> misspecified functional form seen from several angles. **Look at the shape
> before you conclude the units differ.** Section 2.8 returns to this, because it
> determines the order in which you should do things.

## 2.2 Where the variation lives

The pooled/within gap is arithmetic, and it can be inspected before any model is
fitted. Total variation in a grouped variable splits into variation *between*
groups and variation *within* them, and a regression uses whichever is left after
the controls have absorbed the rest.

```python
import essentials as ess

d = ess.load("sourdough")

for col in ["hydration_pct", "oven_c", "rise_mm"]:
    total = d[col].var()
    within = d.groupby("baker")[col].transform(lambda s: s - s.mean()).var()
    between = total - within
    print(f"{col:14s} {total:7.2f}  between {between / total:6.1%}  within {within / total:6.1%}")
```

| variable | total variance | between bakers | within bakers |
|---|---|---|---|
| `hydration_pct` | 112.58 | **71.8%** | 28.2% |
| `oven_c` | 386.33 | 1.4% | **98.6%** |
| `rise_mm` | 207.71 | 17.2% | **82.8%** |

Two facts from that table govern everything below.

**Most of the variation in hydration is between bakers.** So a pooled regression
on hydration is overwhelmingly a between-baker comparison wearing panel clothing,
and adding baker fixed effects will throw away 72% of the predictor's variation.

**Oven temperature is almost entirely within.** It was randomised across loaves
rather than chosen by house style, so fixed effects cost it almost nothing. Two
predictors in the same model can be affected completely differently by the same
control, and the decomposition tells you which is which in advance.

## 2.3 Fixed effects: the within transformation

Give each baker their own intercept:

$$ y_{ig} = \alpha_g + \beta x_{ig} + \varepsilon_{ig} .$$

The $\alpha_g$ absorb everything about a baker that does not vary across their
loaves — their oven, their hands, their kitchen, their skill — **whether or not
you can measure it or even name it**. That is the appeal, and it is a large one:
an unmeasured confounder constant within a baker cannot bias $\hat\beta$, because
it has been differenced away.

There are three equivalent ways to compute it, and their equivalence is the
Frisch–Waugh–Lovell theorem in three costumes:

```python
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("sourdough")

# (a) a dummy for every baker
a = smf.ols("rise_mm ~ hydration_pct + C(baker)", data=d).fit()

# (b) subtract each baker's mean from both variables
dm = pd.DataFrame()
for c in ["rise_mm", "hydration_pct"]:
    dm[c] = d.groupby("baker")[c].transform(lambda s: s - s.mean())
b = smf.ols("rise_mm ~ hydration_pct - 1", data=dm).fit()

# (c) partial the dummies out of both sides, regress residual on residual
ry = smf.ols("rise_mm ~ C(baker)", data=d).fit().resid
rx = smf.ols("hydration_pct ~ C(baker)", data=d).fit().resid
c = sm.OLS(ry, rx).fit()

print(a.params["hydration_pct"], b.params["hydration_pct"], c.params.iloc[0])   # 0.999816 x 3
```

| method | slope |
|---|---|
| (a) dummies | +0.999816 |
| (b) demeaning | +0.999816 |
| (c) FWL, residual on residual | +0.999816 |

They agree to $4.4\times10^{-16}$. And the pooled slope was +0.601 — **sixty per
cent of the within slope.** That gap is the quantitative version of the table in
§2.2: two fifths of the pooled estimate was between-baker comparison, and it
disappears the moment each baker is allowed their own level.

Form (c) is the one worth understanding, because it says what "controlling for"
*means*. To control for baker is to remove the part of hydration that baker
explains, remove the part of rise that baker explains, and ask whether what is
left over still moves together. A coefficient "controlling for X" is a
relationship between two sets of residuals.

## 2.4 What fixed effects throw away, and what they cannot fix

Fixed effects are not free, and the costs are systematically undersold.

**Anything constant within a unit becomes inestimable.** If a baker always used
the same flour, its effect and theirs are indistinguishable and neither is
identified. **If the effect you care about is a time-invariant characteristic,
fixed effects delete your research question rather than answering it.**

**Precision falls, because only within variation is used.** Here that is 72% of
the variation in hydration discarded. If the within variation is small or badly
measured, fixed effects trade bias for a great deal of variance — and measurement
error is *amplified* by demeaning, because differencing removes signal while
leaving noise.

**Parameters cost something.** Seven for eight bakers. Affordable on 240 loaves;
not on 24.

**Time-varying confounders survive untouched.** The reassurance that "fixed
effects control for unobserved heterogeneity" holds only for heterogeneity that
does not move. Anything that changes within a unit over the sample — a new oven
mid-experiment, a heatwave, a change of flour supplier — passes straight through.

> Fixed effects are the right tool when you want the within-unit effect and
> suspect the between-unit comparison is contaminated. That is a judgement about
> the data-generating process, not a default.

## 2.5 Random effects, and why the choice is testable

The alternative treats the unit effects as draws from a distribution rather than
as parameters:

$$ y_{ig} = \mu + \beta x_{ig} + u_g + \varepsilon_{ig}, \qquad u_g \sim (0, \sigma_u^2) .$$

This is more efficient when it is true, because it uses between as well as within
variation, and it keeps time-invariant regressors estimable. It buys that with a
strong assumption: **$u_g$ must be uncorrelated with the regressors.**

In this data the assumption fails by construction. The generator gives the bakers
who prefer wet dough lower skill, so the unit effect and the predictor move
together. The comparison is testable, because fixed effects are consistent
whether or not the assumption holds while random effects are consistent only if
it does — so a systematic difference between the two indicts the assumption
(Hausman, 1978).

Mundlak's reformulation is the more illuminating one. Include each unit's *mean*
of the regressor alongside the regressor itself, and the coefficient on the mean
tests exactly the correlation at issue — turning a specification test into a
coefficient you can look at (Mundlak, 1978):

```python
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("sourdough")

d["hyd_mean"] = d.groupby("baker")["hydration_pct"].transform("mean")
mundlak = smf.ols("rise_mm ~ hydration_pct + hyd_mean", data=d).fit()
print(mundlak.summary().tables[1])   # hyd_mean -0.5553, se 0.1721, t -3.23, p 0.0014
```

| term | estimate | se | $t$ | $p$ |
|---|---|---|---|---|
| `hydration_pct` (within) | +0.9998 | | | |
| **`hyd_mean`** (between) | **−0.5553** | 0.1721 | **−3.23** | **0.0014** |

The coefficient on the within variation is exactly the fixed-effects estimate.
The coefficient on the mean is *significantly negative*: bakers who work wetter
get less rise, over and above what water does to any individual loaf. **That is
the confounding, measured, with a $p$-value on it.** Random effects here would be
biased, and Mundlak tells you so in a single line of output.

> Mundlak's formulation is easier to teach and harder to misapply than the
> Hausman test, because it turns "which estimator?" into "look at this
> coefficient".

## 2.6 Clustered standard errors, properly

The classical formula assumes each of the 240 rows contributes independent
information. It does not: loaves from the same baker share an oven and a pair of
hands, so their residuals are correlated. The cluster-robust estimator replaces
independence between *observations* with independence between *clusters*, allowing
arbitrary correlation within:

$$
\widehat{\operatorname{Var}}(\hat\beta)
= (X^{\top}X)^{-1}\Big(\sum_{g=1}^{G} X_g^{\top} e_g e_g^{\top} X_g\Big)(X^{\top}X)^{-1}
$$

The sandwich is the same shape as chapter 1's White estimator; only the middle
changes, from a sum over observations to a sum over groups.

| standard error on `hydration_pct` | se | $t$ |
|---|---|---|
| classical | 0.1383 | +7.23 |
| HC3 | 0.1514 | +6.60 |
| **clustered by baker** | **0.3559** | **+2.81** |

```python
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("sourdough")

fe = "rise_mm ~ hydration_pct + C(baker)"
for label, kw in [("classical", {}),
                  ("HC3", {"cov_type": "HC3"}),
                  ("clustered", {"cov_type": "cluster", "cov_kwds": {"groups": d.baker}})]:
    m = smf.ols(fe, data=d).fit(**kw)
    print(f"{label:10s} se {m.bse['hydration_pct']:.4f}  t {m.tvalues['hydration_pct']:+.2f}")
# classical 0.1383 +7.23 · HC3 0.1514 +6.60 · clustered 0.3559 +2.81
```

Clustering more than doubles the standard error and cuts the $t$-statistic from
7.2 to 2.8. The coefficient does not move at all — as promised in chapter 1,
dependence does not bias it.

Three points of discipline.

**Cluster at the level at which shocks arrive**, not at the level that gives the
answer you prefer. If you are unsure, the question is about the sampling process
rather than about the data (Abadie et al., 2023).

**The number of clusters is what matters, not the number of observations.** There
are **eight** bakers here. The cluster-robust asymptotics want something like
forty, and below that the estimator *understates* uncertainty — so the 0.3559
above is itself optimistic, and a cluster bootstrap would be the better tool.
Reporting a clustered standard error from eight clusters without saying there
were eight is the kind of omission that survives peer review and should not.

**Clustering is not a fix for a misspecified model.** It widens the interval
around whatever the specification estimates, correct or not.

## 2.7 Curves

Add a squared term:

$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon .$$

Differentiating and setting to zero puts the turning point at
$x^\star = -\beta_1/2\beta_2$, a maximum when $\beta_2 < 0$.

```python
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("sourdough")

quad = smf.ols("rise_mm ~ hydration_pct + I(hydration_pct**2)", data=d).fit()
print(-quad.params["hydration_pct"] / (2 * quad.params["I(hydration_pct ** 2)"]))   # 81.1 %
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
| 80% | +0.6 mm |
| 92% | −0.8 mm |

Those are read off the full surface of §2.11 — both squares, the interaction and
baker effects — at the average oven temperature of 225 °C, where the interaction
term adds $\beta_{xz}\,z$ to the slope:

```python
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("sourdough")

full = smf.ols("rise_mm ~ hydration_pct + I(hydration_pct**2) + oven_c"
               " + I(oven_c**2) + hydration_pct:oven_c + C(baker)", data=d).fit()
p, oven = full.params, d.oven_c.mean()
for x in [65, 80, 92]:
    slope = (p["hydration_pct"] + 2 * p["I(hydration_pct ** 2)"] * x
             + p["hydration_pct:oven_c"] * oven)
    print(x, round(slope, 2))   # 65 +2.24 · 80 +0.56 · 92 -0.77
```

$\beta_1$ alone is the slope **at $x = 0$** — a dough made entirely of flour. It
is a real number in the output and it answers no question anyone asked.

> There is no such thing as "the effect of hydration" in a quadratic model. There
> is only its effect *at a value*, and you must choose which value to report.

Note also what §2.1 now looks like in retrospect: the eight baker-specific slopes
were $\beta_1 + 2\beta_2 \bar x_g$ evaluated at eight different $\bar x_g$. The
"heterogeneity" was the curve.

## 2.8 Interactions

$$
y = \beta_0 + \beta_1 x + \beta_2 z + \beta_3 xz + \varepsilon
\qquad\Longrightarrow\qquad
\frac{\partial y}{\partial z} = \beta_2 + \beta_3 x
$$

$\beta_3$ is the amount by which the effect of $z$ changes per unit of $x$. If it
is zero, the two variables act independently and you can give one recommendation
for each. **If it is not zero, you cannot.**

In a formula, `x:z` is the interaction alone and `x*z` expands to `x + z + x:z`.
Prefer `x*z`: an interaction without its main effects forces the surface through
a constraint you almost certainly did not intend.

On the sourdough data the interaction is +0.015 with $t = 17$. Solving for the
best temperature at a given hydration:

| hydration | best oven |
|---|---|
| 65% | 229.0 °C |
| 90% | 238.2 °C |

```python
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("sourdough")

full = smf.ols("rise_mm ~ hydration_pct + I(hydration_pct**2) + oven_c"
               " + I(oven_c**2) + hydration_pct:oven_c + C(baker)", data=d).fit()
p = full.params
print(p["hydration_pct:oven_c"], full.tvalues["hydration_pct:oven_c"])   # +0.0151, t = 17.0

# d rise / d oven = b_t + 2 b_tt t + b_ht h = 0
for h in [65, 90]:
    best = -(p["oven_c"] + p["hydration_pct:oven_c"] * h) / (2 * p["I(oven_c ** 2)"])
    print(h, round(best, 1))   # 65 -> 229.0 · 90 -> 238.2
```

Nine degrees apart. Wetter dough wants a hotter oven, and an additive model
returns the same number for both.

### The interpretation trap

With an interaction present, $\beta_1$ is the effect of $x$ **when $z = 0$**. For
an oven temperature of zero degrees, that is not a quantity about baking. Two
standard responses:

- **Centre the variables.** Replace $z$ by $z - \bar z$, and $\beta_1$ becomes the
  effect of $x$ at the average $z$ — a value that exists.
- **Report conditional effects instead of coefficients.** Give the effect at two
  or three meaningful values of $z$, as the table above does. This is nearly
  always clearer.

### Testing whether a slope differs by group

Heterogeneity can be modelled rather than absorbed. An interaction with a group
indicator lets the slope itself vary, and **the coefficient *is* the difference in
slopes, so its standard error is the test**:

```python
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("sourdough")

d["wet_baker"] = d.baker.isin(list("EFGH")).astype(int)
m = smf.ols("rise_mm ~ hydration_pct * wet_baker + C(baker)", data=d).fit()
term = "hydration_pct:wet_baker"
print(m.params[term], m.bse[term], m.tvalues[term], m.pvalues[term])   # -1.571 0.258 -6.09 <0.001
```

| term | estimate | se | $t$ | $p$ |
|---|---|---|---|---|
| `hydration_pct:wet_baker` | −1.571 | 0.258 | −6.09 | < 0.001 |

That is the right way to compare two groups. **Fitting the groups separately and
eyeballing whether their intervals overlap is not a test** — it is two tests, each
about zero, neither about the difference, and their overlap has no defined error
rate.

## 2.9 Yule's warning, which predates all of this

Long before panel data existed, George Udny Yule showed that an association
present in every subgroup of a table can vanish or reverse when the subgroups are
combined, purely as an artefact of how the groups differ in size and composition
(Yule, 1903). Simpson revisited it half a century later, and the phenomenon
carries his name in most textbooks and Yule's in the careful ones (Simpson, 1951).

The eight baker slopes are the same phenomenon in continuous form: a relationship
that is strongly positive within most units and only weakly positive across them,
with two units running the other way.

> The lesson is not that the aggregate is wrong. It is that **the direction and
> magnitude of an association are properties of the level of aggregation, not of
> the world**, and that reporting one level without naming it is how the same data
> come to support opposite recommendations.

## 2.10 Get the functional form right first

One result belongs here because it is the loose end from chapter 1 and a warning
against reaching for panel machinery too early:

| model | $R^2$ | AIC | within slope |
|---|---|---|---|
| `hydration_pct + C(baker)` | 0.325 | 1884.4 | +0.9998 |
| full quadratic surface `+ C(baker)` | **0.965** | **1185.5** | — |

The fixed-effects estimate of +0.9998 is a **precise answer on the wrong scale**.
It is the average tangent of a curved surface, reported as though the surface were
a plane. Adding baker effects to a misspecified functional form produces a
well-identified estimate of a quantity nobody wants.

**The order matters: shape first, then units.** No amount of fixed effects or
clustering repairs a model fitted on the wrong scale — and the diagnostics of
chapter 1 are what tell you which you have.

## 2.11 Putting it together

| model | $R^2$ | AIC | hydration\* | error |
|---|---|---|---|---|
| linear + baker FE | 0.490 | 1819.4 | 95.00 | +8.38 pp |
| quadratics, no interaction | 0.919 | 1380.6 | 84.86 | −1.75 pp |
| interaction, **no baker FE** | 0.904 | 1411.1 | 82.41 | **−4.20 pp** |
| everything | **0.965** | **1185.5** | **86.69** | **+0.07 pp** |

*(The planted optimum is 86.62% hydration at 236.86 °C.)*

```python
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
import essentials as ess

d = ess.load("sourdough")
h_true, t_true = ess.sourdough_optimum()          # 86.62, 236.86

FULL = ("rise_mm ~ hydration_pct + I(hydration_pct**2) + oven_c"
        " + I(oven_c**2) + hydration_pct:oven_c + C(baker)")

def peak(fit):
    """Maximise the fitted surface over the design, for baker D."""
    def neg(z):
        row = pd.DataFrame([{"hydration_pct": z[0], "oven_c": z[1], "baker": "D"}])
        return -fit.predict(row).iloc[0]
    return minimize(neg, [75, 225], bounds=[(55, 95), (190, 260)]).x

for label, formula in [
    ("linear + baker FE", "rise_mm ~ hydration_pct + oven_c + C(baker)"),
    ("quadratics, no interaction", FULL.replace(" + hydration_pct:oven_c", "")),
    ("interaction, no baker FE", FULL.replace(" + C(baker)", "")),
    ("everything", FULL),
]:
    m = smf.ols(formula, data=d).fit()
    h, t = peak(m)
    print(f"{label:28s} R2 {m.rsquared:.3f}  AIC {m.aic:7.1f}  "
          f"hydration* {h:5.2f}  error {h - h_true:+.2f} pp")
# 0.490 1819.4 95.00 +8.38 · 0.919 1380.6 84.86 -1.75
# 0.904 1411.1 82.41 -4.20 · 0.965 1185.5 86.69 +0.07
```

The third row is the lesson of the chapter. **$R^2 = 0.904$, every coefficient
significant, and the recommendation is 4.2 percentage points wrong.** This is
omitted variable bias where the omitted variable was not unobservable — it was in
a column called `baker`.

The first row is the other lesson. A plane has no interior maximum, so its advice
is always "more of everything", and it lands on the corner of the design at 95%
and 260 °C. **A recommendation at the boundary of your data is a fact about your
model, not about the world.**

### Report a region

A peak estimated from data is flat near the top:

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import essentials as ess

d = ess.load("sourdough")
full = smf.ols("rise_mm ~ hydration_pct + I(hydration_pct**2) + oven_c"
               " + I(oven_c**2) + hydration_pct:oven_c + C(baker)", data=d).fit()

grid = pd.DataFrame({"hydration_pct": np.linspace(70, 95, 60),
                     "oven_c": 236.9, "baker": "D"})
pred = full.get_prediction(grid).summary_frame()

# hydrations whose predicted rise lies inside the 95% interval at the peak
top = pred["mean"].idxmax()
flat = grid.hydration_pct[pred["mean"] >= pred.loc[top, "mean_ci_lower"]]
print(round(flat.min(), 1), round(flat.max(), 1))   # 82.3 91.2

plt.plot(grid.hydration_pct, pred["mean"])
plt.fill_between(grid.hydration_pct, pred.mean_ci_lower, pred.mean_ci_upper, alpha=0.2)
plt.xlabel("hydration (%)"); plt.ylabel("predicted rise (mm)")
plt.show()
```

Anything from roughly 82% to 91% is indistinguishable from the maximum. Quoting
86.69% implies a precision the data does not support. **The range is the finding.**

## 2.12 Comparing models

For **nested** models — one is the other plus terms — use an F-test:

```python
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("sourdough")

FULL = ("rise_mm ~ hydration_pct + I(hydration_pct**2) + oven_c"
        " + I(oven_c**2) + hydration_pct:oven_c + C(baker)")
full = smf.ols(FULL, data=d).fit()
restricted = smf.ols(FULL.replace(" + hydration_pct:oven_c", ""), data=d).fit()

f_stat, p_val, df_diff = full.compare_f_test(restricted)   # call it on the larger model
print(f_stat, p_val, df_diff)                              # 289.1, 2.3e-42, 1
```

For **non-nested** models, AIC or BIC is what you have. Chapter 1 §1.12 has the
cautions; two are broken constantly and bear repeating here, because this is the
chapter where you will fit six specifications:

1. **Compare only on identical rows.** Drop a missing value in one model and the
   likelihoods are computed on different data. The comparison is void and nothing
   in the output will tell you.
2. **A model chosen by AIC was chosen by you.** Reporting its $p$-values as though
   you had not searched is the quiet misuse. The search consumed evidence the
   standard errors know nothing about.

## 2.13 A note on lagged outcomes

If your data has a time dimension and you are tempted to add a lagged outcome to
a fixed-effects model, read Nickell (1981) first. The demeaning correlates the lag
with the error, biasing the estimate by an amount that shrinks only as the number
of periods grows. With a handful of periods that bias is not negligible, and the
dynamic-panel literature exists to address it.

## 2.14 What to report

1. **Which variation identifies the estimate.** Between, within, or both. State
   it, because the estimates differ by large factors and a reader cannot infer it
   from the coefficient.
2. **More than one specification, side by side.** If they disagree, that
   disagreement is a finding and hiding it is a choice.
3. **What the fixed effects absorbed**, and therefore what you can no longer
   estimate.
4. **The clustering level, with the reason** — and how many clusters there are.
   Below about forty, say what you did about it.
5. **Heterogeneity tested, not assumed.** An interaction coefficient with its
   standard error, not two subgroup regressions compared by eye.
6. **Conditional effects rather than main effects**, wherever a square or an
   interaction is in the model.
7. **The functional form, checked before the panel machinery.** Fixed effects on
   the wrong scale is a precise answer to the wrong question.
8. **An optimum as a region.**

## 2.15 Summary

1. A pooled slope and a within slope are different quantities. Say which you are
   reporting, and which claim it supports.
2. The between/within decomposition tells you in advance what a control will
   cost. Here, baker fixed effects discard 72% of the variation in hydration and
   1.4% of the variation in oven temperature.
3. Fixed effects three ways — dummies, demeaning, FWL — are one estimator.
   "Controlling for X" means regressing residual on residual.
4. Fixed effects remove time-invariant confounders **and nothing else**.
5. Mundlak turns the fixed-versus-random question into a coefficient: here
   **−0.5553 with $p = 0.0014$**, the confounding measured directly.
6. Clustering does not move the estimate and here doubles its standard error.
   **Count your clusters** — eight is not enough for the asymptotics.
7. With a square or an interaction present, a main effect is the slope at zero.
   Report conditional effects.
8. Heterogeneous slopes may be a misspecified curve seen from several angles.
   **Shape first.**
9. A model can explain 90% of the variance and be four points wrong, with nothing
   in its own output to warn you.

## Further reading

The written treatment is
[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/),
chapter [6 · Regression: Advanced Considerations](https://warint.github.io/quantitative-methods/book/session-06.html).

The pooling problem is old. **Yule (1903)** demonstrated that associations reverse
under aggregation, and **Simpson (1951)** gave the phenomenon its modern textbook
treatment and, unfairly, its name.

The fixed-versus-random choice was settled as a testable proposition by
**Hausman (1978)**, whose test compares the two estimators on the logic that only
one is consistent under the null. **Mundlak (1978)** reframed it more usefully:
the random-effects estimator is a fixed-effects estimator with a restriction, and
adding the unit means of the regressors makes the restriction a coefficient you
can test directly.

Two literatures complicate the modern picture. On standard errors,
**Abadie, Athey, Imbens & Wooldridge (2023)** argue that clustering is a question
about the sampling and assignment process rather than a default, and that
clustering when it is not needed costs precision for nothing. **Cameron & Miller
(2015)** is the practical reference, including the small-cluster problem.

On the estimator itself, **de Chaisemartin & D'Haultfœuille (2020)** show that
two-way fixed effects with heterogeneous effects is a weighted average of
comparisons whose weights can be negative. That is a substantive problem rather
than a technicality, and it is the whole subject of chapter 6.

**Nickell (1981)** is the classic result on lagged outcomes in fixed-effects
models. For interactions specifically, **Brambor, Clark & Golder (2006)** is the
standard corrective to the way they are usually reported.

Textbook treatments: **Wooldridge (2019)**, *Introductory Econometrics*, 7th ed.,
ch. 6, 7 and 14; **Angrist & Pischke (2009)**, *Mostly Harmless Econometrics*,
ch. 3; **Baltagi (2021)**, *Econometric Analysis of Panel Data*, 6th ed.

> Each specification identifies $\beta$ from a different slice of the variation.
> If the slices carry different signal, the estimates will differ — **not because
> one is wrong, but because they are answering different questions.** Reporting a
> specification without saying which variation identifies it is reporting half a
> result.
