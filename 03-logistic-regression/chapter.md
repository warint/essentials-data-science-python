# 3 · Logistic Regression

> A rule that is 87.7% accurate, on data where 71.3% accuracy requires no model
> at all, and which puts a poisonous mushroom on your plate once every four
> meals.

## 3.1 Why not a line

Let $y$ be 0 or 1, and fit $y = \beta_0 + \beta_1 x$ anyway. Three things break,
and only one of them is cosmetic.

- The fitted values are unbounded, so you get **predicted probabilities above 1
  and below 0**.
- The residuals cannot be normal: for any $x$ there are only two possible values
  of $y$.
- A Bernoulli variable has variance $p(1-p)$, which depends on $p$. The variance
  therefore changes with the fitted value **by construction** — heteroskedasticity
  that no transformation removes.

The fix is not a different error assumption. It is to model a quantity that is
allowed to be any real number, and then map it into $[0,1]$.

## 3.2 Odds, log-odds, and the logistic function

**Odds** are $p/(1-p)$: bounded below by zero, unbounded above.

| $p$ | 0.2 | 0.5 | 0.8 | 0.99 |
|---|---|---|---|---|
| odds | 0.25 | 1 | 4 | 99 |

**Log-odds**, or the logit, run over the whole real line and are symmetric about
zero:

$$ \text{logit}(p) = \log\frac{p}{1-p} $$

| $p$ | 0.01 | 0.25 | 0.5 | 0.75 | 0.99 |
|---|---|---|---|---|---|
| logit | −4.60 | −1.10 | 0 | +1.10 | +4.60 |

So model *those*:

$$ \log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k = \eta ,$$

and invert to recover a probability:

$$ p = \frac{1}{1 + e^{-\eta}} .$$

There is no closed form for $\hat\beta$. It is found by maximum likelihood,
numerically, which is why the output reports a log-likelihood instead of an
$R^2$.

### The odds ratio, and the mistake everybody makes

A coefficient is a change in log-odds per unit of $x$. Exponentiate it and you
get an **odds ratio**: the factor by which the odds are multiplied.

```python
import essentials as ess
import numpy as np
import statsmodels.formula.api as smf

d = ess.load("mushrooms")
fit = smf.logit("poisonous ~ volva_present + odor_index + ...", data=d).fit()
np.exp(fit.params)
```

On the mushroom data, `volva_present` has a coefficient of +4.29, so an odds
ratio of **73**. A volva is the sac at the base of the stem that marks *Amanita*,
the genus behind most fatal mushroom poisonings; the model found it without being
told.

It multiplies the **odds** by 73. It does not multiply the probability by
anything:

| starting $p$ | after OR = 73 | change |
|---|---|---|
| 0.01 | 0.4245 | +0.41 |
| 0.10 | 0.8903 | +0.79 |
| 0.50 | 0.9865 | +0.49 |
| 0.90 | 0.9985 | +0.10 |

> One odds ratio, four different probability effects. This is why "the effect" of
> a logistic coefficient is not a single number, and why reporting marginal
> effects at chosen values of the other variables is usually clearer than
> reporting the coefficient.

## 3.3 Inference

For a **single** coefficient, the output gives a Wald $z$-statistic:
$\hat\beta/\mathrm{se}(\hat\beta)$, compared against a normal rather than a $t$,
because maximum-likelihood standard errors are asymptotic.

For a **group** of coefficients — all the levels of a categorical variable, say —
use a likelihood-ratio test:

$$ 2\bigl(\ell_{\text{full}} - \ell_{\text{restricted}}\bigr) \sim \chi^2_{\Delta k} $$

```python
from scipy import stats

lr = 2 * (fit.llf - reduced.llf)
df = int(fit.df_model - reduced.df_model)
stats.chi2.sf(lr, df)
```

McFadden's pseudo-$R^2$ (`fit.prsquared`) is $1 - \ell_{\text{full}}/\ell_{\text{null}}$.
It is **not** a share of variance and is not comparable to a linear model's
$R^2$; 0.2 to 0.4 is considered excellent. Reporting it as "$R^2$" invites a
reader to compare it with a number that means something else.

## 3.4 The confusion matrix, and what accuracy hides

`.predict()` returns **probabilities**, not classes. Turning one into a decision
requires a threshold, and the model has no opinion about it.

Start with the baseline you have to beat. The mushroom data is 71.3% edible, so:

> **"Call everything edible" is 71.3% accurate** — from a rule with no data, no
> model and no thought — and it eats all 402 poisonous mushrooms.

Any accuracy figure has to be read against that number. At $t = 0.5$:

|  | called edible | called poisonous |
|---|---|---|
| **truly edible** | 931 | 67 |
| **truly poisonous** | **105** | 297 |

Accuracy 0.877, AUC 0.938 — a real improvement on the baseline. And 105 of 402
poisonous mushrooms called edible: **one meal in four.**

The four cells have names worth using, because "accuracy" is an average over all
four and hides which one you are paying for:

- **sensitivity / recall** = $TP/(TP+FN)$ — of the poisonous ones, how many did
  I catch?
- **precision** = $TP/(TP+FP)$ — of the ones I rejected, how many deserved it?
- **specificity** = $TN/(TN+FP)$ — of the edible ones, how many did I keep?

## 3.5 The threshold is a decision

The two errors have different consequences:

| | |
|---|---|
| **false negative** | called edible, was poisonous — a hospital visit |
| **false positive** | called poisonous, was edible — a discarded mushroom |

Write down costs $c_{FN}$ and $c_{FP}$ and minimise expected loss. The
minimiser is

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
> eighth the expected loss. If that sounds wrong, the metric is wrong. Accuracy
> is the special case where the two errors cost the same, and that assumption is
> almost never stated out loud.

The empirical minimum sits around 0.031 rather than exactly 0.0196, and it
should: with 1,400 discrete cases the loss is a step function, so its minimum is
a flat region. Everything between 0.021 and 0.031 is within 5% of the floor. The
formula names the right neighbourhood, not a value to quote to four decimals.

### The ROC curve

Plotting sensitivity against $1 -$ specificity across every threshold traces the
ROC curve, and the area under it (AUC = 0.938 here) is a threshold-free summary
of the model.

That property is exactly why AUC cannot make your decision. It averages over
thresholds you would never use. **You do not deploy a curve; you deploy a point
on it**, and which point is a question about costs.

## 3.6 Ordinal outcomes

`toxicity` grades the same risk — edible, upset stomach, hospital, fatal — and
collapsing it to a binary throws the ordering away. The proportional-odds model
uses one coefficient vector and a set of cut-points:

$$ \log\frac{\Pr(Y \le j)}{\Pr(Y > j)} = \alpha_j - \sum_k \beta_k x_k $$

```python
from statsmodels.miscmodels.ordinal_model import OrderedModel

X = pd.get_dummies(sub[[...]], drop_first=True, dtype=float)
OrderedModel(sub.toxicity.cat.codes, X, distr="logit").fit(method="bfgs")
```

The assumption is strong and is what buys the parsimony: **the effect of a
feature is the same at every cut-point**. A volva is assumed to affect the odds
of "at least upset stomach" exactly as it affects the odds of "fatal".

Check it by fitting separate binary logits at each cut-point and comparing the
coefficients. If they diverge, the ordinal model is reporting an average of
effects that differ, and a multinomial model is the honest fallback — at the cost
of giving up the ordering entirely.

## 3.7 What to report

- Coefficients as **odds ratios**, with intervals, and without implying they are
  probability ratios.
- The **confusion matrix**, not accuracy alone, and always beside the
  majority-class baseline.
- The **threshold, and the costs that justify it**.
- Which error you deliberately chose to make more of.
- AUC for the model; a named point on the ROC curve for the decision.

## 3.8 Summary

1. Model the log-odds, invert to get probabilities.
2. $e^\beta$ is an odds ratio: constant, unlike the probability change it implies.
3. Accuracy is a trap on imbalanced data. Quote the baseline.
4. The threshold belongs to the costs: $t^\star = c_{FP}/(c_{FP}+c_{FN})$.
5. The best decision rule can have the worse accuracy.
6. Ordinal models buy parsimony with proportional odds. Check it.

## Further reading

- James, Witten, Hastie & Tibshirani (2021), *An Introduction to Statistical
  Learning*, 2nd ed., ch. 4.
- Hosmer, Lemeshow & Sturdivant (2013), *Applied Logistic Regression*, 3rd ed.
- Agresti (2019), *An Introduction to Categorical Data Analysis*, 3rd ed.
- Provost & Fawcett (2013), *Data Science for Business*, ch. 7–8, on
  cost-sensitive evaluation.
