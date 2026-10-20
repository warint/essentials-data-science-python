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
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import essentials as ess

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
y = r.taste_score.to_numpy()
Xs = StandardScaler().fit_transform(X.values)

Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.35, random_state=7)
ols = LinearRegression().fit(Xtr, ytr)

print(f"{mean_squared_error(ytr, ols.predict(Xtr)):.8f}")   # 0.00000000
print(f"{mean_squared_error(yte, ols.predict(Xte)):.2f}")   # 36.23
print(f"{np.var(y):.2f}")                                    # 35.45
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

## 4.2 The mechanical first attempt

Before the penalties, the procedure they replaced — because everything in this
chapter is a response to the way it fails.

The obvious way to handle 200 candidates is to let the data choose: start from
nothing, add whichever ingredient improves the fit most, stop when nothing
improves it further. That is **forward stepwise selection**, and variants of it
were the standard answer for decades and remain the default in a great deal of
applied software.

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import essentials as ess

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
y = r.taste_score.to_numpy()
Xs = StandardScaler().fit_transform(X.values)


def forward(X, y, k_max):
    chosen, remaining = [], list(range(X.shape[1]))
    for _ in range(k_max):
        best = max(remaining,
                   key=lambda j: LinearRegression()
                                 .fit(X[:, chosen + [j]], y)
                                 .score(X[:, chosen + [j]], y))
        chosen.append(best); remaining.remove(best)
    return chosen


picked = X.columns[forward(Xs, y, 15)]
print(list(picked[:6]))      # vanilla_extract, baking_powder, sea_salt,
                             # brown_sugar, browned_butter, dutch_cocoa
print(sum(c not in ess._RECIPE_TRUTH for c in picked))   # 9 of 15 are noise
```

Run it for fifteen steps and the first six chosen are `vanilla_extract`,
`baking_powder`, `sea_salt`, `brown_sugar`, `browned_butter`, `dutch_cocoa` —
six of the seven planted ingredients, in almost their true order of magnitude.
It looks like a triumph. Of the fifteen selected, **nine are noise.**

The damning result is what happens when the same procedure is run on bootstrap
resamples of the same data — ten steps each time, 120 times (about a minute):

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import essentials as ess

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
y = r.taste_score.to_numpy()
Xs = StandardScaler().fit_transform(X.values)


def forward(X, y, k_max):
    chosen, remaining = [], list(range(X.shape[1]))
    for _ in range(k_max):
        best = max(remaining,
                   key=lambda j: LinearRegression()
                                 .fit(X[:, chosen + [j]], y)
                                 .score(X[:, chosen + [j]], y))
        chosen.append(best); remaining.remove(best)
    return chosen


rng = np.random.default_rng(0)
counts = pd.Series(0, index=X.columns)
for _ in range(120):
    i = rng.choice(60, 60, replace=True)          # a bootstrap resample
    counts.iloc[forward(Xs[i], y[i], 10)] += 1

freq = counts / 120
print((counts > 0).sum())                          # 189 of 200
print(freq[["vanilla_extract", "sea_salt", "brown_sugar",
            "baking_powder", "sour_cream_toasted"]].round(3))
# 0.825, 0.617, 0.542, 0.483, 0.275
```

| | |
|---|---|
| candidates selected in **at least one** of 120 resamples | **189 of 200** |
| selection frequency, `vanilla_extract` (true) | 82.5% |
| `sea_salt` (true) | 61.7% |
| `brown_sugar` (true) | 54.2% |
| `baking_powder` (true) | 48.3% |
| **`sour_cream_toasted` (pure noise)** | **27.5%** |

Nearly every candidate in the table is selected somewhere, and a column of pure
noise is chosen in more than a quarter of resamples.

> **"The data selected these variables" is therefore not a finding.** Perturb the
> data slightly and the data select different ones.

The diagnosis is that stepwise makes a sequence of **discrete** decisions, each
conditional on the last. A variable is either in or out; a marginal difference in
fit at step three changes the entire subsequent path. That discreteness is what
makes the procedure unstable — and it points straight at the fix: **stop making
in-or-out decisions and shrink coefficients continuously instead.**

## 4.3 Bias on purpose

Chapter 1 gave the Gauss–Markov theorem as reassurance: least squares is best
among linear estimators that are **unbiased**. Read it again as a *restriction*.
The theorem says nothing whatever about estimators that accept bias, and it
leaves open whether one of those might have smaller error overall.

Decompose the expected squared error of an estimate $\hat\theta$ of a quantity
$\theta$:

$$
\operatorname{E}\big[(\hat\theta - \theta)^2\big]
= \underbrace{\big(\operatorname{E}[\hat\theta] - \theta\big)^2}_{\text{bias}^2}
+ \underbrace{\operatorname{Var}(\hat\theta)}_{\text{variance}} .
$$

- **Bias** — how far the average fit sits from the truth. An error made
  systematically, in every sample.
- **Variance** — how much the fit moves when you resample. An error made
  differently each time.
- **$\sigma^2$** — irreducible noise, which appears when the target is a new
  observation rather than the function. No method removes it, and a method that
  appears to has found something that is not there.

Least squares sets the bias term to **exactly zero**. If the variance term is
large — and with 200 predictors and 60 observations it is enormous — then
accepting a little bias to remove a lot of variance is a straightforwardly better
trade.

> This is a **decision-theoretic** argument rather than a statistical convention.
> Two estimators: A is unbiased with variance 100, giving error $0 + 100$. B has
> bias 2 and variance 10, giving $4 + 10 = 14$. **The biased estimator wins by a
> factor of seven.** Every method in this chapter is a version of B.

### Standardise first

The penalties are on the *size* of the coefficients, and a coefficient's size
depends on the units of its predictor. Measure an ingredient in milligrams rather
than grams and its coefficient shrinks by a factor of a thousand, so the penalty
barely notices it. **Nothing about the data changed; the answer did.**

Standardising every predictor to mean zero and unit variance before fitting is
therefore not hygiene — **it is part of the estimator's definition.** The
intercept is left unpenalised, since there is no reason to shrink the overall
level toward zero, which is why it is conventionally handled by centring $y$
rather than by including a column of ones in the penalty.

## 4.4 Ridge

$$ \hat\beta_{\text{ridge}} = \arg\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_2^2
= (X^\top X + \lambda I)^{-1}X^\top y $$

Adding $\lambda$ to the diagonal makes a singular matrix invertible: ridge did not
merely improve the $p>n$ problem, it made it *solvable*.

Compare that with $(X^{\top}X)^{-1}X^{\top}y$. The only change is $\lambda I$
added to the diagonal — which is why the method is called *ridge*, and why it
works where least squares cannot.

The shrinkage is not uniform, and the singular value decomposition says exactly
how it is targeted. Ridge multiplies the component of the solution along the
$j$-th singular direction by

$$ \frac{d_j^2}{d_j^2 + \lambda} .$$

Where the data are informative — large $d_j$ — the factor is near 1 and almost
nothing happens. Where the data barely vary, and the least-squares coefficient is
therefore an unstable ratio of noise to a very small number, the factor is near 0
and the coefficient is crushed. **The shrinkage goes exactly where the variance
problem is.**

On the standardised recipe design, $X$ has **59 non-zero singular values**
(largest 21.55, smallest non-zero 7.47, so a condition ratio of only 2.9) and
**141 directions in which the data has no variation at all**:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import essentials as ess

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
Xs = StandardScaler().fit_transform(X.values)

d = np.linalg.svd(Xs, compute_uv=False)
nz = d[d > 1e-8]
print(len(nz), Xs.shape[1] - len(nz))                          # 59, 141
print(f"{nz.max():.2f} {nz.min():.2f}")                         # 21.55 7.47

for lam in [1, 10, 100, 1000]:
    f = nz**2 / (nz**2 + lam)
    print(lam, f"{f.max():.4f} {np.median(f):.4f} {f.min():.4f}")
# 1     0.9979 0.9945 0.9824
# 1000  0.3172 0.1540 0.0528
```

| $\lambda$ | strongest direction | median | weakest non-zero | the 141 null directions |
|---|---|---|---|---|
| 1 | 0.9979 | 0.9945 | 0.9824 | **0** |
| 10 | 0.9789 | 0.9479 | 0.8480 | **0** |
| 100 | 0.8229 | 0.6454 | 0.3581 | **0** |
| 1000 | 0.3172 | 0.1540 | 0.0528 | **0** |

The last column is why ridge is *defined* when $p > n$. In the 141 directions the
data cannot see, the factor is zero for any $\lambda > 0$, so ridge assigns them
no weight — where least squares had infinitely many equally good answers and no
way to choose. **Ridge did not merely improve the $p > n$ problem; it made it
solvable.**

Note also what the condition ratio of 2.9 tells you: on *this* design the
informative directions are all similarly informative, because the predictors were
generated independently. The targeting matters far more when predictors are
badly collinear, which is the situation Hoerl and Kennard (1970) proposed it for.

For any finite $\lambda$ the factor on a non-zero direction is strictly between
0 and 1, so **ridge never reaches exactly zero** and keeps all $p$ coefficients.

```python
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import essentials as ess

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
y = r.taste_score.to_numpy()
Xs = StandardScaler().fit_transform(X.values)

kf = KFold(10, shuffle=True, random_state=0)
ridge = RidgeCV(alphas=np.logspace(-2, 4, 25), cv=kf).fit(Xs, y)

mse = -cross_val_score(Ridge(alpha=ridge.alpha_), Xs, y, cv=kf,
                       scoring="neg_mean_squared_error").mean()
print(ridge.alpha_, f"{mse:.2f}")                  # 0.01 28.50
print((np.abs(ridge.coef_) > 1e-6).sum(),          # 200 non-zero
      (np.abs(ridge.coef_) > 0.1).sum())           # 135 above 0.1
```

Chosen $\lambda = 0.01$, cross-validated MSE **28.50**, **200 of 200**
coefficients non-zero and 135 of them above 0.1. Ridge cannot answer "which
ingredients matter", and it predicts poorly here because the truth is sparse and
ridge has no way to represent sparsity.

## 4.5 Lasso

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

### Derive it once, on an orthonormal design

There is no closed form in general, but there is in the case that explains
everything. Suppose the predictors are orthonormal, so $X^{\top}X = I$ and the
least-squares solution is simply $z_j$. The objective then separates into one
problem per coefficient, each with the solution

$$ \hat\beta_j^{\text{lasso}} = \operatorname{sign}(z_j)\big(|z_j| - \lambda/2\big)_+ $$

— the **soft-thresholding** operator. Ridge, in the same case, gives
$z_j/(1+\lambda)$.

Build a genuinely orthonormal design and check both, at $\lambda = 1$:

```python
import numpy as np
from sklearn.linear_model import Lasso, Ridge

n, lam = 100, 1.0
Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(n, 6)))   # Q'Q = I
z = np.array([3.0, 1.5, 0.4, -0.9, -2.2, 0.05])
y = Q @ z                                          # so the OLS solution is z

# scikit-learn's lasso minimises ||y - Xb||^2 / (2n) + alpha*||b||_1,
# so lambda/2 on our scale is alpha = lambda / (2n) on theirs.
lasso = Lasso(alpha=lam / (2 * n), fit_intercept=False, tol=1e-12).fit(Q, y)
ridge = Ridge(alpha=lam, fit_intercept=False).fit(Q, y)

soft = np.sign(z) * np.maximum(np.abs(z) - lam / 2, 0)
print(lasso.coef_.round(3))                        # [ 2.5  1.   0.  -0.4 -1.7  0. ]
print(ridge.coef_.round(3))                        # [ 1.5  0.75 0.2 -0.45 -1.1 0.025]
print(np.abs(lasso.coef_ - soft).max(),            # ~4e-16
      np.abs(ridge.coef_ - z / (1 + lam)).max())   # ~3e-16
```

| $z$ (OLS) | lasso, fitted | soft-threshold formula | ridge, fitted | $z/(1+\lambda)$ |
|---|---|---|---|---|
| +3.000 | +2.500 | **+2.500** | +1.500 | **+1.500** |
| +1.500 | +1.000 | **+1.000** | +0.750 | **+0.750** |
| +0.400 | **0.000** | **0.000** | +0.200 | +0.200 |
| −0.900 | −0.400 | **−0.400** | −0.450 | **−0.450** |
| −2.200 | −1.700 | **−1.700** | −1.100 | **−1.100** |
| +0.050 | **0.000** | **0.000** | +0.025 | +0.025 |

Both match their formulas to $2\times10^{-16}$.

Read what the operator does. **Every coefficient is pulled toward zero by the
same fixed amount $\lambda/2$**, and any coefficient whose least-squares value
was smaller than that amount is pulled *past* zero and held there — rows three
and six. The zeros are not a numerical accident. They are what happens when a
constant subtraction meets a small number.

Ridge, by contrast, applies a *proportional* shrinkage, which multiplies small
coefficients by a factor and therefore never reaches zero — rows three and six
again, at +0.200 and +0.025.

### On the recipe data

```python
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import essentials as ess

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
y = r.taste_score.to_numpy()
Xs = StandardScaler().fit_transform(X.values)
kf = KFold(10, shuffle=True, random_state=0)

lasso = LassoCV(cv=kf, random_state=0, max_iter=200_000).fit(Xs, y)
coef = pd.Series(lasso.coef_, index=X.columns)

mse = -cross_val_score(Lasso(alpha=lasso.alpha_, max_iter=200_000), Xs, y,
                       cv=kf, scoring="neg_mean_squared_error").mean()
print(f"{lasso.alpha_:.2f} {mse:.2f}")             # 0.43 9.90
print((coef != 0).sum())                           # 25
print(pd.DataFrame({"planted": pd.Series(ess._RECIPE_TRUTH),
                    "lasso": coef[list(ess._RECIPE_TRUTH)].round(3)}))
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

## 4.6 Two ingredients that are one ingredient

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

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import essentials as ess

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
y = r.taste_score.to_numpy()
Xs = StandardScaler().fit_transform(X.values)

print(f"{np.corrcoef(X.baking_soda, X.baking_powder)[0, 1]:.3f}")   # 0.963

pair = ["baking_soda", "baking_powder"]
for seed in range(5):
    kf = KFold(10, shuffle=True, random_state=seed)
    m = ElasticNetCV(l1_ratio=0.95, cv=kf, random_state=0,
                     max_iter=200_000).fit(Xs, y)
    c = pd.Series(m.coef_, index=X.columns)[pair]
    print(seed, c.round(3).tolist(), round(c.sum(), 3))
# 0 [0.374, 1.481] 1.855   ...   2 [1.318, 0.891] 2.209
```

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

## 4.7 What cross-validation is optimising

$k$-fold cross-validation splits the rows into $k$ parts, fits on $k-1$ and
predicts the held-out part, then averages. `shuffle=True` matters: without it
`KFold` takes contiguous blocks, and if the rows are sorted by anything the folds
are not exchangeable.

The lasso selected **25** ingredients. Six are real. Nineteen are false
positives.

This is not a failure. Cross-validation estimates **prediction** error, and the
penalty that predicts best is smaller than the penalty that selects best. A few
extra small coefficients cost almost nothing predictively.

> **A lasso's non-zero set is not a list of true effects.** "The model identified
> 25 drivers of taste" is the commonest misuse of this method — and it is a misuse
> of a correctly fitted model.

### The one-standard-error rule

The cross-validation curve is itself an estimate with noise in it, so its minimum
is **optimistically located**. The standard correction is to take not the
$\lambda$ with the lowest estimated error but the *largest* $\lambda$ whose error
is within one standard error of the minimum:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import essentials as ess

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
y = r.taste_score.to_numpy()
Xs = StandardScaler().fit_transform(X.values)
kf = KFold(10, shuffle=True, random_state=0)

grid = np.logspace(-2, 1, 16)
folds = np.array([-cross_val_score(Lasso(alpha=a, max_iter=200_000), Xs, y,
                                   cv=kf, scoring="neg_mean_squared_error")
                  for a in grid])
mean, se = folds.mean(1), folds.std(1) / np.sqrt(kf.n_splits)
best = mean.argmin()
one_se = grid[mean <= mean[best] + se[best]].max()

for lam in [grid[best], one_se]:
    c = pd.Series(Lasso(alpha=lam, max_iter=200_000).fit(Xs, y).coef_,
                  index=X.columns)
    sel = set(c[c != 0].index)
    print(f"{lam:.3f}", len(sel), len(sel & set(ess._RECIPE_TRUTH)),
          len(sel - set(ess._RECIPE_TRUTH)))
# 0.398 27 6 21
# 0.631 15 5 10
```

| | $\lambda$ | selected | true found | false positives |
|---|---|---|---|---|
| CV minimum | 0.398 | 27 | 6 of 7 | 21 |
| **1-SE rule** | 0.631 | **15** | 5 of 7 | 10 |

The rule buys a model 45% smaller for an error difference the data cannot
resolve — at the cost of one more true ingredient. That is the trade, stated
plainly; neither answer is wrong, and reporting which rule you used is not
optional.

### Report stability, not significance

If a single fit cannot be trusted to name variables, the honest alternative is
to say how *reliably* each one is named. Refit on many resamples and report the
proportion of times each appears (Meinshausen & Bühlmann, 2010):

| ingredient | selection frequency | |
|---|---|---|
| `vanilla_extract` | **100.0%** | true |
| `sea_salt` | 90.5% | true |
| **`sour_cream_toasted`** | **85.5%** | **noise** |
| `brown_sugar` | 81.0% | true |
| `baking_powder` | 75.0% | true |
| `browned_butter` | 71.0% | true |
| **`walnut_toasted`** | **69.0%** | **noise** |
| `baking_soda` | 24.0% | true |
| `dutch_cocoa` | 14.0% | true |

Stability selection is a real improvement on a single fit — the four strongest
signals sit at the top, and the ordering is broadly right. **It is not a
solution.** A pure-noise column ranks third, above two genuine ingredients, and
the two hardest true cases — the collinear leavener and the weak signal — sit at
24% and 14%, below half a dozen noise columns.

> Report the frequencies anyway. A table like this is a statement the data
> supports, and it is far less impressive than the single-fit result — which is
> precisely its value.

### The signal that got away

`dutch_cocoa` has a true effect of +1.1 and comes back between 0.00 and 0.25
depending on the split. At $n = 60$, with 200 candidates and a noise sd of 2.5,
it sits **below the detection floor**. No method recovers it, and one that
appeared to would have found noise.

Being able to say *"there may be further effects around 1 that this design cannot
detect"* is worth more than selecting variables correctly.

## 4.8 What none of this licenses

The most important sentence in this chapter is a prohibition.

> **A $p$-value from a model chosen by the lasso is not a $p$-value.** Fit the
> selected variables by ordinary least squares, report the resulting standard
> errors, and every one of them is wrong — computed as though the specification
> had been fixed in advance, when in fact it was chosen using the same data.

This is chapter 1's Freedman problem with a modern face, and it has been treated
formally. Valid inference after selection requires either conditioning on the
selection event or widening the intervals to cover every model that might have
been selected (Berk et al., 2013). Tests designed specifically for the lasso path
exist (Lockhart et al., 2014) and are not what a naive refit computes.

The practical responses, in order of how often they are available:

**Split the sample.** Select on one half, estimate and test on the other. The
inference is then honest, at the cost of half the data, and it is the only remedy
that needs no theory.

**Report stability rather than significance** — the frequency table in §4.7. That
is a statement the data support.

**Treat selection as exploratory and say so.** A shortlist for further work is a
legitimate output. A shortlist presented as a tested finding is not.

And three properties to state alongside it:

1. **Penalised coefficients are biased towards zero, by design.** Do not report
   them as effect sizes without saying so.
2. **Selection is unstable under collinearity.** Which of two correlated
   variables survives is close to arbitrary and changes with a resample — §4.6
   measured exactly that.
3. **A signal below the detection floor stays there.** `dutch_cocoa` is not
   recoverable at $n = 60$ by any method here, and the honest report says so
   rather than presenting the selected set as complete.

## 4.9 Summary

1. When $p > n$, OLS has no unique solution, fits perfectly and predicts terribly.
   Zero training error is a symptom, not an achievement.
2. **Stepwise selection is unstable**: 189 of 200 candidates are selected in some
   resample. "The data chose these variables" is not a finding.
3. Gauss–Markov is a restriction, not a reassurance. Every method here is biased
   on purpose, and that is why it wins.
4. Standardise before penalising — it is part of the estimator's definition.
5. Ridge shrinks by $d_j^2/(d_j^2+\lambda)$, targeting the unstable directions,
   and assigns exactly zero to the directions the data cannot see — which is why
   it is defined when $p > n$. It never reaches zero on a direction the data *can*
   see, so it never selects.
6. The lasso soft-thresholds: a **constant** subtraction, which is where the zeros
   come from. Ridge shrinks **proportionally**, which is why it has none.
7. The lasso must choose between collinear columns; the elastic net shares — but
   only the pair's total is identified.
8. Cross-validation optimises prediction. Use the one-standard-error rule if you
   want a smaller model, and say which rule you used.
9. Report **selection frequencies**, not $p$-values. And know that even those rank
   a noise column third here.
10. **A $p$-value from a lasso-selected model is not a $p$-value.**

## Further reading

The written treatment is
[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/),
chapter [5 · Regularisation: Ridge, Lasso, and the Elastic Net](https://warint.github.io/quantitative-methods/book/session-05.html).
Chapter [8](https://warint.github.io/quantitative-methods/book/session-08.html)
carries the bias–variance decomposition this one assumes.

Ridge regression begins with **Hoerl & Kennard (1970)**, who proposed biased
estimation explicitly as a response to instability under near-collinearity — the
title says "biased estimation" without apology. **Frank & Friedman (1993)** set
ridge, subset selection and their relatives in a common framework indexed by the
power of the penalty, which is where the lasso's $\ell_1$ sits between ridge's
$\ell_2$ and subset selection's $\ell_0$.

**Tibshirani (1996)** introduced the lasso and observed the property that made it
dominant: continuous shrinkage and automatic selection in the same estimator.
**Efron, Hastie, Johnstone & Tibshirani (2004)** supplied least angle regression,
which computes the entire coefficient path at the cost of a single least-squares
fit and turned the lasso from attractive into practical. **Zou & Hastie (2005)**
diagnosed the lasso's behaviour among correlated predictors and proposed the
elastic net, with the grouping effect as the explicit remedy.

Choosing the penalty rests on **Stone (1974)**, whose formulation of
cross-validation as a general criterion for model choice long predates its use
here.

The inferential warning has the sharpest literature. **Berk, Brown, Buja, Zhang &
Zhao (2013)** established what valid post-selection inference requires and how
much wider the intervals must be; **Lockhart, Taylor, Tibshirani & Tibshirani
(2014)** developed a test for the lasso path specifically, making clear how
different it is from a naive refit. **Meinshausen & Bühlmann (2010)** offers
stability selection as the practical alternative — report how often a variable is
chosen across resamples, rather than a significance level that assumes it was
never chosen at all.

Textbook treatments: **Hastie, Tibshirani & Friedman (2009)**, *The Elements of
Statistical Learning*, 2nd ed., §3.4; **James, Witten, Hastie & Tibshirani
(2021)**, *An Introduction to Statistical Learning*, 2nd ed., ch. 6.
