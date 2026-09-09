# 4 · Ridge, Lasso, Elastic Net

> Sixty bakes, two hundred ingredients. Least squares fits the training data
> *perfectly* — zero residuals — and then predicts new bakes worse than the mean
> of the outcome would.

## 4.1 When there are more columns than rows

Every model so far had many more rows than columns. This chapter starts with the
opposite: $n = 60$, $p = 200$.

$$ \hat\beta_{\text{OLS}} = (X^\top X)^{-1}X^\top y $$

With $p > n$, $X^\top X$ is $200\times200$ with rank at most 60. It is singular
and cannot be inverted. There are **infinitely many** coefficient vectors that
fit the training rows exactly, and the data cannot choose between them.

```python
import essentials as ess
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
y = r.taste_score.to_numpy()
Xs = StandardScaler().fit_transform(X.values)

Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.35, random_state=7)
ols = LinearRegression().fit(Xtr, ytr)
```

| | |
|---|---|
| training MSE | **0.00000000** |
| test MSE | 36.23 |
| variance of `y` | 35.45 |

Test error exceeds the variance of the outcome. **Zero training error is a
symptom, not an achievement**: the model had enough freedom to memorise the noise
and did.

The way out is to stop asking for the best fit and deliberately accept a worse
one. The device is a penalty, and the rest of this chapter is three of them.

## 4.2 Bias on purpose

$$ \mathbb{E}\left[(y - \hat f(x))^2\right] = \text{Bias}^2 + \text{Variance} + \sigma^2 $$

- **Bias** — how far the average fit sits from the truth. High for models too
  simple to represent it.
- **Variance** — how much the fit moves when you resample. High for flexible
  models.
- **$\sigma^2$** — noise. No method removes it, and a method that appears to has
  found something that is not there.

Least squares is *unbiased*. That sounds like a virtue, and as $p$ approaches $n$
it becomes a disaster: the unbiasedness is bought with enormous variance, and
total error is what you actually pay.

> Two estimators. A is unbiased with variance 100, giving error $0 + 100$. B has
> bias 2 and variance 10, giving $4 + 10 = 14$. **The biased estimator wins by a
> factor of seven.** Every method in this chapter is a version of B.

### Standardise first

A penalty sums coefficients, and a coefficient's magnitude depends on its
variable's units. An unstandardised penalty therefore punishes whichever variable
happens to be recorded in small numbers — so the variables it selects are partly
an artefact of somebody's choice of units. Standardising is not a nicety; it is a
precondition for the penalty meaning anything.

## 4.3 Ridge

$$ \hat\beta_{\text{ridge}} = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_2^2
= (X^\top X + \lambda I)^{-1}X^\top y $$

Adding $\lambda$ to the diagonal makes a singular matrix invertible: ridge did not
merely improve the $p>n$ problem, it made it *solvable*.

In the singular-value basis, ridge multiplies the $j$-th component of the OLS
solution by

$$ \frac{d_j^2}{d_j^2 + \lambda},$$

which is strictly between 0 and 1 for any finite $\lambda$. Two consequences:

- Low-variance directions (small $d_j$) are shrunk hardest — precisely where OLS
  was least stable.
- The factor is **never exactly zero**, so ridge keeps all $p$ coefficients.
  Always.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(10, shuffle=True, random_state=0)
ridge = RidgeCV(alphas=np.logspace(-2, 4, 25), cv=kf).fit(Xs, y)
```

Chosen $\lambda = 0.01$, cross-validated MSE **28.50**, **200 of 200**
coefficients non-zero and 135 of them above 0.1. Ridge cannot answer "which
ingredients matter", and it predicts poorly here because the truth is sparse and
ridge has no way to represent sparsity.

## 4.4 Lasso

$$ \hat\beta_{\text{lasso}} = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_1 $$

One character changed, and the behaviour is qualitatively different.

### Where the zeros come from

Both methods minimise squared error subject to a budget on $\beta$. Only the
*shape* of the budget differs.

- The $\ell_2$ budget is a **ball** — smooth, so the point where the error
  contours first touch it is generically off every axis. No zeros.
- The $\ell_1$ budget is a **diamond**, with corners **on** the axes. Contours
  touching a corner give a solution with coefficients exactly zero.

> Sparsity is a fact about the geometry of the constraint region, not a threshold
> applied afterwards.

In the one-variable case the solution is **soft-thresholding**:

$$ \hat\beta = \operatorname{sign}(z)\left(|z| - \lambda\right)_+ $$

where $z$ is the OLS estimate. Everything within $\lambda$ of zero *becomes*
zero; everything else moves towards zero by exactly $\lambda$. Ridge, by
contrast, multiplies by $1/(1+\lambda)$ — always proportional, never arriving.

### On the recipe data

```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=kf, random_state=0, max_iter=200_000).fit(Xs, y)
```

Chosen $\lambda = 0.43$, cross-validated MSE **9.90** — three times better than
ridge — and 25 selected ingredients.

| ingredient | planted | lasso |
|---|---|---|
| `vanilla_extract` | +4.0 | +3.191 |
| `brown_sugar` | +2.5 | +1.341 |
| `sea_salt` | −3.2 | −2.224 |
| `browned_butter` | +1.8 | +1.322 |
| `dutch_cocoa` | +1.1 | +0.042 |
| `baking_soda` | +1.5 | **0.000** |
| `baking_powder` | +1.5 | +1.942 |

Every recovered coefficient is smaller than the truth. That is shrinkage doing
its job, and it is the bias you agreed to.

## 4.5 Two ingredients that are one ingredient

`baking_soda` and `baking_powder` are correlated at **0.963** and have
**identical** true effects. The lasso set one to exactly zero and gave the other
their combined weight.

It has no choice. The $\ell_1$ penalty charges $|\beta_1| + |\beta_2|$, so
loading everything onto one column costs exactly the same as splitting evenly —
and the all-on-one solution sits at a **corner** of the diamond, which is where
solutions land. Ridge, charging $\beta_1^2 + \beta_2^2$, minimises by splitting,
which is why it shares the load and selects neither.

Worse than arbitrary, the lasso is *stable* about it: the same ingredient is
dropped in every cross-validation split, confidently, with nothing in the output
to suggest a second ingredient is hiding behind the first.

### The elastic net

$$ \hat\beta = \arg\min_\beta \|y - X\beta\|^2 + \lambda\left(\alpha\|\beta\|_1 + \tfrac{1-\alpha}{2}\|\beta\|_2^2\right) $$

$\alpha = 1$ is the lasso, $\alpha = 0$ is ridge. The $\ell_1$ part still
produces exact zeros; the $\ell_2$ part has no reason to prefer either of two
identical columns, so it shares between them. This is the **grouping effect**.

> In scikit-learn, $\alpha$ is `l1_ratio` and `alpha` is the overall $\lambda$. In
> `glmnet` the names are swapped. Check which library you are reading.

At `l1_ratio=0.95` the elastic net keeps both: soda +0.374, powder +1.481.

### And then the question that matters

Refit across five cross-validation seeds:

| CV seed | soda | powder | **sum** |
|---|---|---|---|
| 0 | +0.374 | +1.481 | **+1.855** |
| 1 | +0.786 | +1.203 | **+1.989** |
| 2 | +1.318 | +0.891 | **+2.209** |
| 3 | +0.699 | +1.248 | **+1.947** |
| 4 | +0.552 | +1.358 | **+1.910** |

The **sum is stable**; the **split is not**, swinging on an arbitrary seed and
nearly reversing at seed 2.

> Two collinear columns carry **one identifiable quantity** between them. No
> penalty conjures a second. The elastic net's real service is telling you there
> are two ingredients here — it cannot tell you how to divide the credit.

The honest report is *"leavening matters, combined effect about +1.9; the data
cannot apportion it between soda and powder, because the two were varied
together."* The fix is a better experiment, not a better estimator.

## 4.6 What cross-validation is optimising

$k$-fold cross-validation splits the rows into $k$ parts, fits on $k-1$ and
predicts the held-out part, then averages. `shuffle=True` matters: without it
`KFold` takes contiguous blocks, and if the rows are sorted by anything the folds
are not exchangeable.

The lasso selected **25** ingredients. Seven are real. Nineteen are false
positives.

This is not a failure. Cross-validation estimates **prediction** error, and the
penalty that predicts best is smaller than the penalty that selects best. A few
extra small coefficients cost almost nothing predictively.

> **A lasso's non-zero set is not a list of true effects.** "The model identified
> 25 drivers of taste" is the commonest misuse of this method — and it is a misuse
> of a correctly fitted model.

### The signal that got away

`dutch_cocoa` has a true effect of +1.1 and comes back between 0.00 and 0.25
depending on the split. At $n = 60$, with 200 candidates and a noise sd of 2.5,
it sits **below the detection floor**. No method recovers it, and one that
appeared to would have found noise.

Being able to say *"there may be further effects around 1 that this design cannot
detect"* is worth more than selecting variables correctly.

## 4.7 Three limitations to state out loud

1. **Penalised coefficients are biased towards zero, by design.** Do not report
   them as effect sizes without saying so.
2. **The standard errors are not valid.** Any library will print them, computed as
   though $\lambda$ and the selected set had been fixed in advance. They were
   chosen by looking at the data. Honest inference needs post-selection machinery
   or a clean sample split. A table of lasso coefficients with stars beside them
   reports inference that its own selection invalidated.
3. **Selection is unstable under collinearity.** Which of two correlated
   variables survives is close to arbitrary and changes with a resample.

## 4.8 Summary

1. When $p > n$, OLS has no unique solution, fits perfectly and predicts terribly.
2. Every method here is biased on purpose, and that is why it wins.
3. Standardise before penalising.
4. Ridge shrinks and never zeroes; the lasso zeroes, because the $\ell_1$ ball has
   corners.
5. The lasso must choose between collinear columns; the elastic net shares — but
   only the pair's total is identified.
6. Cross-validation optimises prediction. Its selected set is not a list of true
   effects, and its coefficients carry no valid $p$-values.

## Further reading

- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*,
  2nd ed., §3.4.
- Tibshirani (1996), "Regression shrinkage and selection via the lasso",
  *JRSS-B* 58(1), 267–288.
- Zou & Hastie (2005), "Regularization and variable selection via the elastic
  net", *JRSS-B* 67(2), 301–320.
- James, Witten, Hastie & Tibshirani (2021), *An Introduction to Statistical
  Learning*, 2nd ed., ch. 6.
