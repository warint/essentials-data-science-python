# 3 · Logistic Regression

> A rule that is 87.7% accurate, on data where 71.3% accuracy requires no model
> at all, and which puts a poisonous mushroom on your plate once every four
> meals.

## 3.1 When the outcome is a category

Everything so far has predicted a number. Most decisions are not numbers. Is
this mushroom safe. Does this borrower default. Does this patient need the scan.
The outcome is a category, and the machinery of chapters 1 and 2 does not
transfer intact.

The obvious move is to code the outcome 0/1 and fit least squares anyway. That is
the **linear probability model**, and it is not absurd: the fitted values estimate
$\operatorname{E}[y \mid x]$, which for a binary variable *is*
$\Pr(y = 1 \mid x)$, and the coefficients read directly as changes in
probability — which is what people actually want. It is used in serious applied
work.

It breaks in three specific ways, and being precise about which matters.

**First, it returns probabilities that are not probabilities.** On the mushroom
data, 254 fitted values fall below zero and 21 above one — **275 of 1,400, or one
in five** — with a range of $[-0.407, 1.358]$. The failure is worst when the base
rate is near 0 or 1, or when a predictor takes values outside the range it was
fitted on, which is to say exactly when you extrapolate.

```python
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("mushrooms")
F = ("poisonous ~ cap_diameter_cm + stem_height_cm + odor_index + ring_present"
     " + volva_present + C(spore_print, Treatment('brown'))"
     " + C(gill_spacing, Treatment('close')) + C(habitat, Treatment('grassland'))")

lpm = smf.ols(F, data=d).fit(cov_type="HC3")   # robust SEs: see the second failure
f = lpm.fittedvalues
print((f < 0).sum(), (f > 1).sum())            # 254 21
print(f.min().round(3), f.max().round(3))      # -0.407 1.358
```

**Second, the error variance cannot be constant.** If $y_i$ is Bernoulli with
probability $p_i$ then $\operatorname{Var}(y_i) = p_i(1-p_i)$ **by
construction**: the variance is a function of the mean, largest at $p = 0.5$ and
vanishing at either end. Heteroscedasticity here is not a defect of the data — it
is a property of binary outcomes, and no amount of clean data removes it. Chapter
1's table applies: the coefficients stay unbiased and the classical standard
errors are wrong, always. **Robust standard errors are mandatory rather than
optional.**

**Third, and most importantly, the functional form is wrong at the ends.** A
straight line says a fixed change in a predictor moves the probability by a fixed
amount whether you start at 0.02 or at 0.50. That is not how probabilities
behave. Moving from near-certain-not to slightly-possible is a large change in
the underlying propensity; moving from 0.50 to 0.55 is a small one. The line has
no way to express it.

## 3.2 The logistic function, and where it came from

Malthus had argued that population grows geometrically while subsistence grows
arithmetically, so catastrophe is arithmetic. Pierre François Verhulst, a Belgian
mathematician, thought the premise wrong: unchecked exponential growth describes
no actual population, because growth slows as a population approaches what its
territory can support.

He wrote down the correction. If growth is proportional both to the current
population *and to the remaining room*, the result is a curve that rises
exponentially, inflects, and flattens toward a ceiling. He named it the
**logistic** curve in 1845.

It reached statistics by a different road. Berkson proposed it in the 1940s as
the natural transformation for dose–response data — the *logit* — arguing against
the then-dominant probit on grounds of tractability and interpretation (Berkson,
1944). Cox gave the regression treatment its modern form (Cox, 1958).

> The function Verhulst wrote to describe populations pressing against a limit is
> now the standard way to model any bounded probability. It is the same
> mathematical situation: something that cannot exceed a ceiling, and that slows
> as it approaches one.

### Odds, log-odds, and the curve

**Odds** are $p/(1-p)$: bounded below by zero, unbounded above.

| $p$ | 0.2 | 0.5 | 0.8 | 0.99 |
|---|---|---|---|---|
| odds | 0.25 | 1 | 4 | 99 |

**Log-odds**, or the logit, run over the whole real line and are symmetric about
zero:

| $p$ | 0.01 | 0.25 | 0.5 | 0.75 | 0.99 |
|---|---|---|---|---|---|
| $\log\frac{p}{1-p}$ | −4.60 | −1.10 | 0 | +1.10 | +4.60 |

So write the linear index as $\eta_i = x_i^{\top}\beta$ and set

$$
\Pr(y_i = 1 \mid x_i) = \sigma(\eta_i) = \frac{1}{1 + e^{-\eta_i}} .
$$

The function is bounded strictly between 0 and 1, so the first failure is gone by
construction. It is symmetric about $\eta = 0$, where $p = 0.5$ and the slope is
steepest, and it flattens in both tails — so the third failure is gone too.

Inverting it gives the form that explains the model's whole vocabulary:

$$ \log\frac{p_i}{1-p_i} = x_i^{\top}\beta .$$

> The model is linear — **in the log-odds**. That is the entire trick, and it is
> why coefficients are not probabilities.

## 3.3 Estimation: no formula, and why that is fine

Least squares had a closed form. This does not. The log-likelihood is

$$
\ell(\beta) = \sum_i \Big[\,y_i\log\sigma(\eta_i) + (1-y_i)\log\big(1 - \sigma(\eta_i)\big)\Big],
$$

and setting its derivative to zero gives the **score condition**

$$ X^{\top}\big(y - \sigma(X\beta)\big) = 0 ,$$

which has no algebraic solution because $\sigma$ is nonlinear in $\beta$. The
estimate is found numerically, by Newton–Raphson — equivalently by iteratively
reweighted least squares, which is the same computation dressed as a sequence of
weighted regressions.

You can check that the optimiser did its job:

```python
import numpy as np
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("mushrooms")
F = ("poisonous ~ cap_diameter_cm + stem_height_cm + odor_index + ring_present"
     " + volva_present + C(spore_print, Treatment('brown'))"
     " + C(gill_spacing, Treatment('close')) + C(habitat, Treatment('grassland'))")

m = smf.logit(F, data=d).fit()
X, y = m.model.exog, m.model.endog
print(np.abs(X.T @ (y - m.predict(d))).max())   # ~1e-12: zero, to rounding
print(round(m.prsquared, 4))                    # 0.5331, McFadden (below)
```

This is less alarming than it sounds: the log-likelihood for logistic regression
is **globally concave**, so there is one maximum and any sensible optimiser finds
it. The exception matters and is §3.9.

### A warning about pseudo-$R^2$

`m.prsquared` reports McFadden's statistic, $1 - \ell_{\text{full}}/\ell_{\text{null}}$
— on this model, 0.5331. It is **not** a proportion of variance explained and does
not behave like $R^2$. Values that would be alarming in a linear model are
ordinary here; 0.2 to 0.4 is generally considered excellent. Comparing it across
models fitted to different samples, or treating it as "53% of something", is a
misreading.

## 3.4 Reading the coefficients

A logistic coefficient is a change in log-odds per unit of the predictor. Nobody
thinks in log-odds. There are three standard translations, in increasing order of
usefulness and decreasing order of popularity.

**Odds ratio** — $e^{\beta_j}$, the multiplicative change in the odds per
one-unit increase in $x_j$. Constant across the range of $x$, which is its
convenience and its trap: constant on the odds scale means *not* constant on the
probability scale.

**Marginal effect** — $\partial p/\partial x_j = \beta_j\,p(1-p)$, the change in
probability per unit of $x_j$. It depends on where you are. The *average marginal
effect* averages it over the observed sample, and is usually the number a reader
wants.

**Predicted probability** — $\sigma(x^{\top}\hat\beta)$ at specified values. The
most honest presentation for a non-technical audience, because it makes the
nonlinearity visible instead of hiding it in a ratio.

On the mushroom model, `volva_present` — the membranous sac at the stem base that
marks *Amanita*, the genus behind most fatal poisonings — comes back at:

| translation | value |
|---|---|
| coefficient (log-odds) | **+4.2925** |
| odds ratio | **×73.1** |
| average change in probability | **+0.4493** |

### The same coefficient at three different places

For the *continuous* predictor `odor_index` (coefficient +0.6310) the derivative
form applies directly:

| at $p$ = | 0.02 | 0.20 | 0.50 |
|---|---|---|---|
| $\partial p/\partial x$ | +0.0124 | +0.1010 | +0.1577 |

Its average marginal effect over the sample is **+0.0548** probability points per
unit of odour.

For a *binary* predictor like `volva_present`, the derivative is the wrong object
— a unit change is the whole switch — so compute the **discrete change**
$p(x=1) - p(x=0)$ instead:

| baseline risk without a volva | with a volva | change |
|---|---|---|
| 0.0014 | 0.0925 | +0.0911 |
| 0.0533 | 0.8047 | **+0.7514** |
| 0.7550 | 0.9956 | +0.2405 |

The three baselines are the 10th, 50th and 90th percentiles of the predicted risk
with the volva switched off:

```python
import numpy as np
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("mushrooms")
F = ("poisonous ~ cap_diameter_cm + stem_height_cm + odor_index + ring_present"
     " + volva_present + C(spore_print, Treatment('brown'))"
     " + C(gill_spacing, Treatment('close')) + C(habitat, Treatment('grassland'))")
m = smf.logit(F, data=d).fit(disp=0)
p = m.predict(d)
b, b_odor = m.params["volva_present"], m.params["odor_index"]

print(round(b, 4), round(np.exp(b), 1))                  # 4.2925 73.1
p1 = m.predict(d.assign(volva_present=1))                # everyone with a volva
p0 = m.predict(d.assign(volva_present=0))                # ... and without
print(round((p1 - p0).mean(), 4))                        # 0.4493, average change

print([float(round(b_odor * q * (1 - q), 4)) for q in (0.02, 0.20, 0.50)])
                                                         # [0.0124, 0.101, 0.1577]
print(round((b_odor * p * (1 - p)).mean(), 4))           # 0.0548, odour AME

for q in (0.1, 0.5, 0.9):
    lo = p0.quantile(q)
    hi = 1 / (1 + np.exp(-(np.log(lo / (1 - lo)) + b)))  # same person, add a volva
    print(round(lo, 4), round(hi, 4), round(hi - lo, 4))
# 0.0014 0.0925 0.0911  /  0.0533 0.8047 0.7514  /  0.755 0.9956 0.2405
```

> One coefficient, one odds ratio, and three completely different consequences.
> The probability change is largest in the middle and small at both ends, because
> that is the shape of the curve. **An odds ratio conceals that; a discrete change
> or a plot of predicted probabilities shows it.**

### The comparability problem nobody mentions

There is a subtler problem that is widely committed. Logistic coefficients are
identified only up to the scale of the unobserved error, and **that scale changes
when predictors are added** — so coefficients from nested logistic models are not
comparable the way linear coefficients are (Mood, 2009). An apparent change on
adding a control may be pure rescaling.

The mushroom data shows it plainly:

| model | $\hat\beta$ on volva | odds ratio | average marginal effect |
|---|---|---|---|
| `volva` only | +1.741 | ×5.7 | +0.3162 |
| `+ odor_index` | +2.257 | ×9.6 | +0.3382 |
| full model | **+4.293** | **×73.1** | **+0.3725** |

The coefficient more than doubles and the odds ratio grows thirteen-fold. The
**average marginal effect moves by 18%**. Most of what looks like a dramatically
strengthening effect is the rescaling Mood describes.

```python
import numpy as np
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("mushrooms")
F = ("poisonous ~ cap_diameter_cm + stem_height_cm + odor_index + ring_present"
     " + volva_present + C(spore_print, Treatment('brown'))"
     " + C(gill_spacing, Treatment('close')) + C(habitat, Treatment('grassland'))")

for f in ["poisonous ~ volva_present", "poisonous ~ volva_present + odor_index", F]:
    m = smf.logit(f, data=d).fit(disp=0)
    b = m.params["volva_present"]
    ame = m.get_margeff().summary_frame().loc["volva_present", "dy/dx"]
    print(f"{b:+.3f}  x{np.exp(b):.1f}  {ame:+.4f}")
# +1.741  x5.7  +0.3162  /  +2.257  x9.6  +0.3382  /  +4.292  x73.1  +0.3725
```

The marginal effects in this table are the derivative form averaged over the
sample, so they are comparable with each other; they are not the discrete change
of +0.4493 above.

> Compare **average marginal effects or predicted probabilities** across
> specifications. Never raw coefficients or odds ratios.

## 3.5 Comparing models: the likelihood-ratio test

With no sums of squares there is no $F$ test, but the likelihood supplies the
analogue. For nested models, twice the difference in log-likelihood is
asymptotically $\chi^2$ with degrees of freedom equal to the number of
restrictions:

$$ \mathrm{LR} = 2\big(\ell_{\text{full}} - \ell_{\text{restricted}}\big) \overset{d}{\longrightarrow} \chi^2_q .$$

```python
from scipy import stats
import statsmodels.formula.api as smf
import essentials as ess

d = ess.load("mushrooms")
F = ("poisonous ~ cap_diameter_cm + stem_height_cm + odor_index + ring_present"
     " + volva_present + C(spore_print, Treatment('brown'))"
     " + C(gill_spacing, Treatment('close')) + C(habitat, Treatment('grassland'))")

full = smf.logit(F, data=d).fit(disp=0)
reduced = smf.logit(F.replace(" + C(habitat, Treatment('grassland'))", ""),
                    data=d).fit(disp=0)            # restriction: habitat is irrelevant

lr = 2 * (full.llf - reduced.llf)
df = int(full.df_model - reduced.df_model)
print(round(lr, 2), df, stats.chi2.sf(lr, df))     # 21.27 2 2.4e-05
```

For a **single** coefficient the output's Wald $z$-statistic serves —
$\hat\beta/\mathrm{se}(\hat\beta)$, compared against a normal rather than a $t$,
because maximum-likelihood standard errors are asymptotic.

Two disciplines this enforces. Both models must be fitted to **the same rows** — a
predictor with extra missing values silently changes the sample and the
comparison becomes meaningless. And the test applies only to **nested** models;
for non-nested candidates, AIC from chapter 1 is the tool.

## 3.6 Discrimination and calibration

Two different questions, routinely conflated.

**Discrimination** — can the model rank? The area under the ROC curve is exactly
the probability that a randomly chosen positive case receives a higher predicted
probability than a randomly chosen negative one (Hanley & McNeil, 1982). That
rank interpretation means it can be computed from a Mann–Whitney statistic, and
on the mushroom model both routes give **0.9382** to four decimals.

**Calibration** — are the probabilities *true*? Among cases assigned 0.30, do
about 30% turn out positive? The **Brier score**, the mean squared difference
between predicted probability and outcome, is the canonical proper scoring rule
(Brier, 1950).

| | Brier score |
|---|---|
| the model | **0.0880** |
| predicting the base rate for everyone | 0.2047 |
| improvement | **57.0%** |

And the calibration itself, by decile of predicted probability:

| predicted | 0.001 | 0.004 | 0.013 | 0.033 | 0.070 | 0.146 | 0.293 | 0.538 | 0.811 | 0.963 |
|---|---|---|---|---|---|---|---|---|---|---|
| **observed** | 0.000 | 0.000 | 0.007 | 0.043 | 0.071 | 0.150 | 0.307 | 0.536 | 0.807 | 0.950 |

Predicted and observed track each other across the whole range. This model is
both discriminating and well calibrated — which is what you would hope, since the
generator *is* a logistic model and we have fitted it correctly.

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.metrics import brier_score_loss, roc_auc_score
import essentials as ess

d = ess.load("mushrooms")
F = ("poisonous ~ cap_diameter_cm + stem_height_cm + odor_index + ring_present"
     " + volva_present + C(spore_print, Treatment('brown'))"
     " + C(gill_spacing, Treatment('close')) + C(habitat, Treatment('grassland'))")
p = smf.logit(F, data=d).fit(disp=0).predict(d)
y = d.poisonous

pos, neg = p[y == 1], p[y == 0]
U = stats.mannwhitneyu(pos, neg).statistic
print(round(roc_auc_score(y, p), 4), round(U / (len(pos) * len(neg)), 4))  # 0.9382 0.9382

model = brier_score_loss(y, p)
base = brier_score_loss(y, np.full(len(y), y.mean()))
print(round(model, 4), round(base, 4), f"{1 - model / base:.1%}")  # 0.088 0.2047 57.0%

decile = pd.qcut(p, 10, labels=False)
print(pd.DataFrame({"predicted": p.groupby(decile).mean(),
                    "observed": y.groupby(decile).mean()}).round(3).T)
```

> A model can discriminate well and be badly calibrated, or the reverse, and
> which matters depends entirely on the use. **A ranking for a limited budget of
> inspections needs discrimination. A price that must reflect risk needs
> calibration.** Report both, each against its baseline, and say which one your
> decision needs.

## 3.7 The confusion matrix, and what accuracy hides

`.predict()` returns **probabilities**, not classes. Turning one into a decision
requires a threshold, and the model has no opinion about it.

Start with the baseline you have to beat. The mushroom data is 71.3% edible, so:

> **"Call everything edible" is 71.3% accurate** — from a rule with no data, no
> model and no thought — and it eats all 402 poisonous mushrooms.

At $t = 0.5$:

|  | called edible | called poisonous |
|---|---|---|
| **truly edible** | 931 | 67 |
| **truly poisonous** | **105** | 297 |

Accuracy 0.877, AUC 0.938 — a real improvement on the baseline. And 105 of 402
poisonous mushrooms called edible: **one meal in four.**

The four cells have names worth using, because accuracy is an average over all
four and hides which one you are paying for:

- **sensitivity / recall** = $TP/(TP+FN)$ — of the poisonous ones, how many did I
  catch?
- **precision** = $TP/(TP+FP)$ — of the ones I rejected, how many deserved it?
- **specificity** = $TN/(TN+FP)$ — of the edible ones, how many did I keep?

## 3.8 The threshold is a decision

The two errors have different consequences:

| | |
|---|---|
| **false negative** | called edible, was poisonous — a hospital visit |
| **false positive** | called poisonous, was edible — a discarded mushroom |

Write down costs $c_{FN}$ and $c_{FP}$ and minimise expected loss. The minimiser
is

$$ t^\star = \frac{c_{FP}}{c_{FP} + c_{FN}} .$$

With a hospital visit 50 times worse than a lost mushroom, $t^\star = 1/51 =
0.0196$ — two orders of magnitude away from 0.5.

| threshold | accuracy | poisonous missed | basket discarded | expected loss |
|---|---|---|---|---|
| always edible | 0.713 | 402 of 402 | 0% | 20,100 |
| 0.50 | **0.877** | **105 (26.1%)** | 26% | 5,317 |
| 0.10 | 0.764 | 17 (4.2%) | 50% | 1,164 |
| **0.0196** | 0.585 | **1 (0.25%)** | 70% | **630** |
| 0.005 | 0.449 | 0 | 84% | 772 |

> The cost-optimal rule has **worse accuracy** — 0.585 against 0.877 — and one
> eighth the expected loss. If that sounds wrong, the metric is wrong. Accuracy is
> the special case where the two errors cost the same, and that assumption is
> almost never stated out loud.

The empirical minimum sits around 0.031 rather than exactly 0.0196, and it
should: with 1,400 discrete cases the loss is a step function, so its minimum is
a flat region. Everything between 0.021 and 0.031 is within 5% of the floor. The
formula names the right neighbourhood, not a value to quote to four decimals.

```python
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix
import essentials as ess

d = ess.load("mushrooms")
F = ("poisonous ~ cap_diameter_cm + stem_height_cm + odor_index + ring_present"
     " + volva_present + C(spore_print, Treatment('brown'))"
     " + C(gill_spacing, Treatment('close')) + C(habitat, Treatment('grassland'))")
p = smf.logit(F, data=d).fit(disp=0).predict(d)
y = d.poisonous
C_FN, C_FP = 50, 1

print(round(1 - y.mean(), 3))                       # 0.713, "call everything edible"
print(confusion_matrix(y, (p >= 0.5).astype(int)))  # [[931 67] [105 297]]

def outcome(t):
    call = (p >= t).astype(int)                     # 1 = called poisonous
    fn = ((call == 0) & (y == 1)).sum()
    fp = ((call == 1) & (y == 0)).sum()
    return (call == y).mean(), fn, call.mean(), C_FN * fn + C_FP * fp

for t in [0.50, 0.10, C_FP / (C_FP + C_FN), 0.005]:
    acc, fn, discarded, loss = outcome(t)
    print(f"t={t:.4f}  acc {acc:.3f}  missed {fn:3d}  discarded {discarded:.0%}  loss {loss}")
# 0.877 105 26% 5317 / 0.764 17 50% 1164 / 0.585 1 70% 630 / 0.449 0 84% 772

ts = np.linspace(0.002, 0.95, 300)
loss = np.array([outcome(t)[3] for t in ts])
near = ts[loss <= 1.05 * loss.min()]
print(round(ts[loss.argmin()], 3), near.min().round(3), near.max().round(3))  # 0.031 0.021 0.031
```

On this 300-point grid. Sweep every distinct fitted probability instead and the
floor is 609 at $t \approx 0.022$: the step function has several near-equal
steps, which is the point.

### The ROC curve is the menu, not the choice

Plotting sensitivity against $1-$specificity across every threshold traces the
ROC curve, and the area under it is a threshold-free summary of the model.

That property is exactly why AUC cannot make your decision. It averages over
thresholds you would never use. **You do not deploy a curve; you deploy a point
on it**, and which point is a question about costs.

## 3.9 Ordinal and multinomial outcomes

### When the categories are ordered

`toxicity` grades the same risk — edible, upset stomach, hospital, fatal — and
collapsing it to a binary throws the ordering away. The **proportional-odds**
model fits one slope and several thresholds:

$$ \Pr(y_i \le j) = \sigma\big(\tau_j - x_i^{\top}\beta\big), \qquad \tau_1 < \tau_2 < \dots < \tau_{J-1} .$$

```python
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
import essentials as ess

d = ess.load("mushrooms")
sub = d.dropna(subset=["toxicity"]).copy()   # edible < upset stomach < hospital < fatal
X = pd.get_dummies(
    sub[["odor_index", "volva_present", "ring_present", "spore_print"]],
    columns=["spore_print"], drop_first=True, dtype=float)

ordinal = OrderedModel(sub.toxicity.cat.codes, X, distr="logit").fit(method="bfgs", disp=0)
print(ordinal.params.round(4))   # volva_present 3.70; then three cut-points
```

Each threshold $\tau_j$ marks a cut on the latent scale, and the **same** $\beta$
shifts every cut together. That is the proportional-odds assumption, and it is a
real restriction: the effect of a volva on moving from "edible" to "upset
stomach" is assumed identical to its effect on moving from "hospital" to "fatal".

The assumption should be **tested rather than assumed** (Brant, 1990). The cheap
version: fit separate binary logits at each cut-point and compare the
coefficients. If they diverge, either a partial-proportional-odds model or the
unordered multinomial is preferable to a restriction the data reject.

In this generator the assumption holds by construction — the grades are cuts
through one latent risk — so the separate logits should agree, and finding that
they do is the correct answer. **The lesson is the check, not the result.**

```python
import pandas as pd
import statsmodels.api as sm
import essentials as ess

d = ess.load("mushrooms")
sub = d.dropna(subset=["toxicity"]).copy()
X = sm.add_constant(pd.get_dummies(
    sub[["odor_index", "volva_present", "ring_present", "spore_print"]],
    columns=["spore_print"], drop_first=True, dtype=float))
grade = sub.toxicity.cat.codes

for j, name in enumerate(sub.toxicity.cat.categories[1:], start=1):
    cut = sm.Logit((grade >= j).astype(int), X).fit(disp=0)
    print(f"at least {name:14s}", cut.params[["odor_index", "volva_present"]].round(2).values)
# volva 3.66, 4.13, 3.58 at the three cuts, against 3.70 from the ordinal model
```

### When the categories have no order

With $J$ unordered categories, one is chosen as baseline and $J-1$ comparisons
are estimated jointly:

$$ \log\frac{\Pr(y_i = j)}{\Pr(y_i = \text{baseline})} = x_i^{\top}\beta_j .$$

Every coefficient is therefore relative to the baseline, and **changing the
baseline changes all of them** — without changing a single fitted probability. A
multinomial result reported without naming its baseline cannot be read.

The multinomial model carries an assumption the ordinal one does not:
**independence of irrelevant alternatives**. The ratio of probabilities for any
two categories must not depend on what other categories exist. When alternatives
are close substitutes the assumption fails, and the classic illustration is a
transport choice between car, red bus and blue bus — introducing a bus of a
different colour should not change the car's share, and in a multinomial logit it
does.

## 3.10 The failure mode worth knowing

Logistic regression has one dramatic failure, and it looks like success.

**Separation** — a predictor, or a combination of them, perfectly predicts the
outcome in the sample. The likelihood then has **no maximum**: pushing the
coefficient toward infinity keeps improving the fit, so the optimiser diverges
(Albert & Anderson, 1984).

It is easy to produce here. Take mushrooms with a volva, a white spore print and
an odour index above 6:

| flag | n | poisonous rate |
|---|---|---|
| 0 | 1,374 | 0.274 |
| **1** | **26** | **1.000** |

Twenty-six mushrooms, all poisonous. Fit a logit including that flag and:

```
LinAlgError: Singular matrix
```

The Hessian is singular because the likelihood has no maximum, so maximum
likelihood **fails outright** rather than returning a suspicious number. In other
software you get an enormous estimate with an enormous standard error and a
$p$-value near 1 — which reads as "not significant" and is in fact "not
estimable".

The wrong response is to report whatever came back. The right one is to recognise
the diagnosis — look for a predictor that separates the outcome, especially a
rare category or a small subgroup — and then drop the variable, merge categories,
or use a **penalised estimator**. Firth's method adds a penalty that guarantees
finite estimates and reduces small-sample bias at the same time (Firth, 1993).

A penalty does give you a number, and it is worth seeing what kind of number:

| penalty strength | coefficient on the flag |
|---|---|
| almost none (`C=1e6`) | +6.04 |
| moderate (`C=1.0`) | +2.44 |
| strong (`C=0.1`) | +0.84 |

```python
import warnings
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
import essentials as ess

d = ess.load("mushrooms")
d["flag"] = ((d.volva_present == 1) & (d.spore_print == "white")
             & (d.odor_index > 6)).astype(int)
print(d.groupby("flag").poisonous.agg(["size", "mean"]))   # 26 cases, all poisonous

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        m = smf.logit("poisonous ~ flag + odor_index", data=d).fit(disp=0)
        print(m.mle_retvals["converged"], m.params["flag"], m.bse["flag"], m.pvalues["flag"])
    except Exception as e:      # older statsmodels: LinAlgError: Singular matrix
        print(type(e).__name__, e)
# recent statsmodels does not crash: converged False, an estimate near 20,
# a standard error near 10,000 and p ~ 1.0 -- "not estimable", read as "not significant"

for C in (1e6, 1.0, 0.1):
    pen = LogisticRegression(C=C, max_iter=10_000).fit(d[["flag", "odor_index"]], d.poisonous)
    print(C, round(pen.coef_[0][0], 2))   # +6.04, +2.43, +0.84
```

At `C=1e6` even the number is an accident of when the optimiser stopped: tighten
its tolerance and the coefficient keeps climbing.

> The estimate is whatever the penalty says it is, because **the data contains no
> information about how large the effect is** — only that it is large enough to
> separate 26 cases. That dependence on the penalty is the honest signal, and it
> is the same phenomenon chapter 4 is built on.

## 3.11 What to report

1. **The base rate.** Every performance number is meaningless without it; 95%
   accuracy on a 5% outcome is achieved by predicting "no" for everyone.
2. **Coefficients as odds ratios, with marginal effects beside them.** The odds
   ratio is constant and the probability change is not, and the reader needs both
   to know what a coefficient means where they intend to use it. For a binary
   predictor, report the discrete change, not the derivative.
3. **Never compare raw coefficients across nested logistic models.** Compare
   average marginal effects.
4. **A nested comparison as a likelihood-ratio test**, fitted to identical rows,
   with the restriction stated.
5. **Discrimination and calibration, separately.** AUC and Brier, each against
   its baseline, with a sentence on which one the intended decision needs.
6. **The threshold, and the costs that justify it.** And which error you
   deliberately chose to make more of.
7. **The assumption the extension rests on.** Proportional odds for ordinal;
   independence of irrelevant alternatives for multinomial. Say that you tested
   it, or say that you did not.
8. **What the model does not license.** These are associations in a sample of
   mushrooms somebody chose to collect — a population already selected by whoever
   was doing the collecting. Chapter 6 gives that problem its name.

## 3.12 Summary

1. The linear probability model fails in three specific ways: out-of-range
   fitted values, structurally non-constant variance, and the wrong shape at the
   ends. Only the third is fatal.
2. Model the log-odds and invert. The logistic curve came from Verhulst's
   population ceiling, which is the same mathematical situation.
3. There is no closed form, but the likelihood is globally concave. Check
   $X^{\top}(y-p) = 0$ at the optimum.
4. $e^\beta$ is an odds ratio: constant, unlike the probability change it implies.
   Report a marginal effect too.
5. **Logistic coefficients are not comparable across nested models.** Average
   marginal effects are.
6. Discrimination and calibration are different questions with different metrics.
7. Accuracy is a trap on imbalanced data. Quote the baseline.
8. The threshold belongs to the costs: $t^\star = c_{FP}/(c_{FP}+c_{FN})$. The
   best decision rule can have the worse accuracy.
9. Ordinal models buy parsimony with proportional odds; multinomial models buy
   flexibility with IIA. Check whichever you relied on.
10. Separation makes the estimate infinite and the software either crashes or
    lies. Recognise it; penalise or restructure.

## Further reading

The written treatment is
[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/),
chapter [4 · Logistic Regression: Binary, Ordinal, and Multinomial](https://warint.github.io/quantitative-methods/book/session-04.html).

The curve predates the statistics by a century. **Berkson (1944)** proposed the
logit for bio-assay against the entrenched probit, and **Cox (1958)** gave binary
regression its modern treatment. The two transformations remain nearly
indistinguishable in fit and differ mainly in the interpretability of their
coefficients.

Assessment splits along the discrimination–calibration line. **Hanley & McNeil
(1982)** gave the AUC its standard exposition and its rank interpretation — the
property that lets it be computed from a Mann–Whitney statistic. **Brier (1950)**,
published in meteorology and largely ignored by econometrics for decades, is the
canonical proper scoring rule for probability forecasts, and remains the right
answer when the probabilities themselves matter rather than their ordering.

The comparability problem is the most under-appreciated result in applied use.
**Mood (2009)** showed that logistic coefficients are confounded with the scale of
the unobserved error, so adding controls rescales them and cross-model comparison
of coefficients or odds ratios is not valid — a practice that remains routine.

For extensions, **Brant (1990)** supplies the test of the proportional-odds
restriction that ordinal models assume and rarely check. On the failure of
estimation, **Albert & Anderson (1984)** characterised exactly when maximum
likelihood estimates fail to exist under separation, and **Firth (1993)** supplied
the penalty that is now the standard fix.

Textbook treatments: **James, Witten, Hastie & Tibshirani (2021)**,
*An Introduction to Statistical Learning*, 2nd ed., ch. 4;
**Hosmer, Lemeshow & Sturdivant (2013)**, *Applied Logistic Regression*, 3rd ed.;
**Agresti (2019)**, *An Introduction to Categorical Data Analysis*, 3rd ed.;
**Provost & Fawcett (2013)**, *Data Science for Business*, ch. 7–8, on
cost-sensitive evaluation.

> A classifier reported without its threshold, or a threshold reported without
> its costs, is an unfinished piece of work.
