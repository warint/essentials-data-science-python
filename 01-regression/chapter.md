# 1 · Regression

> There is a law of physics hiding in a spreadsheet of theme-park statistics.
> A regression can find it. The same regression, fitted slightly differently,
> reports a number that means nothing — and reports it with an excellent $R^2$.

## 1.1 The line, and where it came from

Adrien-Marie Legendre published the method of least squares in 1805, in an
appendix to a book about the orbits of comets. Four years later Carl Friedrich
Gauss published it too, in *Theoria Motus*, and added that he had been using it
since 1795. The priority quarrel that followed was bitter and is still not
entirely settled.

What matters here is what Gauss added that Legendre did not have: not the recipe,
but the *justification*. Legendre offered least squares as a sensible way to
reconcile discordant observations. Gauss asked a different question — under what
conditions is this the best you can do? — and that question is the subject of
this chapter.

You have $n$ observations, a response $y$, and a matrix $X$ holding a column of
ones and $p-1$ predictors. Least squares defines "close" as the sum of squared
vertical distances and chooses

$$ \hat\beta = \arg\min_{\beta} \lVert y - X\beta \rVert^2 .$$

Two choices in that sentence are doing work.

**Squared, rather than absolute.** Squaring makes the objective differentiable
everywhere, which is what delivers the closed form below; it also makes the
fitted values a *linear* function of $y$, which is what makes the entire
diagnostic apparatus of this chapter possible. The price is that a doubled error
gets four times the weight of a single one, so least squares is structurally
sensitive to observations far from the line. Every technique below for finding
influential points is paying off the debt incurred by that square.

**Vertical.** The distances are measured in the $y$ direction only, so the method
assumes $x$ is known and $y$ carries the error. If both are noisy, least squares
understates the slope, and the fix is a different method rather than a different
formula.

Differentiating and setting to zero gives the *normal equations*
$X^{\top}X\beta = X^{\top}y$, whose solution — whenever $X^{\top}X$ can be
inverted — is

$$ \hat\beta = (X^{\top}X)^{-1}X^{\top}y .$$

Every equation in this chapter is something you can execute, and executing it is
the fastest way to find out whether you have understood what it claims:

```python
import numpy as np, statsmodels.api as sm
import essentials as ess

d = ess.load("coasters")
g = d[d.propulsion == "gravity"].copy()
g["lv"], g["lh"] = np.log(g.speed_kmh), np.log(g.drop_m)

X = sm.add_constant(g[["lh"]]).to_numpy()
y = g.lv.to_numpy()

np.linalg.solve(X.T @ X, X.T @ y)    # [2.71914274, 0.49705579]
sm.OLS(y, X).fit().params            # [2.71914274, 0.49705579]
```

The two agree to $1.3\times10^{-14}$. Write it as `solve()` rather than `inv()`:
inverting a matrix in order to multiply it by a vector is slower and less stable
than solving the system directly.

> That is an **algebraic fact**. It requires no assumption about the world, no
> probability, no error term. Feed it any numbers and it returns coefficients.
> This is exactly why diagnostics are necessary: **the formula never refuses.** It
> has no way to tell you that the relationship is curved, that six rows are
> dragging the whole line, or that the standard errors it reports are fiction. It
> returns the same confident four decimal places either way.

The statistical content arrives only when you attach a model,
$y = X\beta + \varepsilon$, and make claims about $\varepsilon$. The
**Gauss–Markov theorem** says that if the model is correctly specified, if
$\operatorname{E}[\varepsilon \mid X] = 0$, and if the errors have constant
variance and are mutually uncorrelated, then $\hat\beta$ is the Best Linear
Unbiased Estimator: no other estimator both linear in $y$ and unbiased has
smaller variance.

Note what is *not* on that list. **Normality is not required for
Gauss–Markov.** It is required for something else — the exactness of the $t$ and
$F$ distributions in small samples — and confusing the two is one of the most
common errors in applied work.

### A note on the name

The word "regression" arrived later and by accident. Francis Galton, studying
the heights of parents and children, found that tall parents had children who
were tall but *less* tall, and called it "regression towards mediocrity"
(Galton, 1886). He was naming a substantive biological phenomenon. The name stuck
to the statistical method he had used to find it, which is why a technique for
fitting conditional means is called by a word that means moving backwards.

The phenomenon is not a fact about biology. It follows from the arithmetic of an
imperfect correlation: whenever two measurements of the same thing are less than
perfectly correlated, the extremes of the first are on average less extreme in
the second — not because anything pulled them back, but because part of what made
them extreme was noise, and noise does not repeat.

The warning Galton's contemporaries missed is the one to carry forward. **If you
select the extremes and measure them again, they will improve, and the
improvement will look like the effect of whatever you did in between.** Chapter 6
returns to this, where it has a sharper name: regression to the mean is the
reason a single-group before-and-after comparison is not evidence.

## 1.2 Adequacy, validity, robustness

These three words get used loosely and they are not synonyms. The distinction
organises everything that follows.

**Adequacy** — whether the model describes the data you actually have. An
adequate model leaves residuals with no remaining structure: nothing in what is
left over could have been used to predict $y$ better. Assessed against the
sample, mostly with plots.

**Validity** — whether the inferences are warranted: whether the standard errors,
intervals and $p$-values mean what they claim. A model can be adequate and its
inference invalid. This is a question about the *sampling distribution*, not
about fit.

**Robustness** — whether the conclusion survives reasonable changes to the
specification, the sample or the estimator. A question about *stability*.

Keeping them apart tells you what a given failure actually costs, and this is the
part most often got wrong:

| If this fails | $\hat\beta$ is | Standard errors are | Damage is to |
|---|---|---|---|
| Correct specification | **biased** | meaningless | the estimate itself |
| Constant variance | unbiased | **wrong** | validity only |
| Independent errors | unbiased | **wrong**, usually far too small | validity only |
| Normal errors | unbiased | fine in large samples | almost nothing, if $n$ is large |

Read that twice. **Only the first row damages the coefficient.** Rows two and
three leave your estimate untouched and destroy your uncertainty — the right
number with the wrong error bar, which matters enormously if the question is
whether an effect can be distinguished from zero. Row four, the assumption
students check first, is the one that matters least.

A model with $R^2 = 0.95$ can fail all of it. This is not hypothetical:
Anscombe's four datasets share, to two decimal places, the same means,
variances, correlation, regression line and $R^2 = 0.67$, and they are utterly
different (Anscombe, 1973). In the first the line describes the data. In the
second the relationship is deterministic and curved — more observations of a
parabola still make a parabola. In the third one aberrant point has tilted a line
that would otherwise pass almost exactly through the rest. In the fourth every
$x$ but one is identical, so deleting that one observation makes the slope not
merely different but *undefined*, because $X^{\top}X$ becomes singular.

> Anscombe's point was not that summary statistics are useless. It was that they
> are **insufficient**, and that the missing information is available for the cost
> of a plot.

## 1.3 Reading the output

A fitted model hands you three numbers per variable. You need all three.

```python
import statsmodels.formula.api as smf
lin = smf.ols("speed_kmh ~ drop_m", data=g).fit()
```

The **coefficient** is a quantity with units: 0.94 km/h per metre of drop. A
coaster whose first drop is ten metres taller is predicted about 9.4 km/h faster.
A coefficient reported without its units is not a finding, and the units are the
first thing a referee will ask for.

The **standard error** is the honest part of the output. It says how much the
coefficient would wobble if you collected the data again from the same process.
Everything you may claim about precision comes from here.

The **$t$-statistic** is the coefficient divided by its standard error: how many
wobbles the estimate sits from zero. It is a ratio, so it has no units, and on
its own it says nothing about whether an effect is large enough to care about.
With enough rows, an effect too small to matter clears any threshold you like.

### The $p$-value, stated carefully

> If the true effect were exactly zero, and this study were repeated many times,
> what fraction of those repetitions would produce a coefficient at least this
> far from zero?

That is the whole content of a $p$-value. It is a statement about *data under an
assumption* — not about the probability that the assumption is true, and not
about the size of anything. Two studies reporting $p = 0.04$ and $p = 0.002$ tell
you nothing about which effect is larger.

## 1.4 Goodness of fit, and what it cannot tell you

The total variation in $y$ splits into the part the model accounts for and the
part it does not:

$$
\underbrace{\textstyle\sum_i (y_i - \bar y)^2}_{\text{TSS}}
=
\underbrace{\textstyle\sum_i (\hat y_i - \bar y)^2}_{\text{ESS}}
+
\underbrace{\textstyle\sum_i e_i^2}_{\text{RSS}},
\qquad
R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}} .
$$

On the log–log coaster model: TSS 8.373582, ESS 8.255655, RSS 0.117927, and
ESS + RSS reproduces TSS to $1.8\times10^{-15}$.

That decomposition holds **only** because the residuals are orthogonal to the
fitted values, which is a property of least squares *with an intercept* and not a
law of nature. Fit without an intercept and $R^2$ can be negative.

Its defect is the one Anscombe exploited: **$R^2$ cannot fall when a predictor is
added**, however irrelevant, because the extra column can always be given a
coefficient of zero. Two corrections are standard.

**Adjusted $R^2$** charges for the parameters spent:

$$ R^2_{\text{adj}} = 1 - \frac{\mathrm{RSS}/(n-p)}{\mathrm{TSS}/(n-1)} $$

It *can* decrease when a useless variable is added, which is the point. It is
still not a model-selection criterion.

**Residual standard error** is the typical size of a residual, in the units of
$y$: $\mathrm{RSE} = s = \sqrt{\mathrm{RSS}/(n-p)}$. Unlike $R^2$ it is not
unitless, which makes it the more honest number to report:

| model | $R^2$ | RSE |
|---|---|---|
| `speed_kmh ~ drop_m` | 0.9635 | **6.84 km/h** |
| `speed_kmh ~ sqrt(drop_m)` | 0.9837 | **4.57 km/h** |
| *(the standard deviation of `speed_kmh` itself)* | | 35.55 km/h |

"Typically wrong by 6.8 km/h" means something. "$R^2 = 0.96$" means something
only relative to the variance of this particular sample — and the third row is
what that variance happens to be.

> $R^2$ measures how **close** the line is to the points. It says nothing about
> whether the line is the **right shape**. When those two come apart, $R^2$ gives
> no signal at all — and on this data they come apart badly.

## 1.5 What the residual knows

Write $\hat y = X\hat\beta$. Substituting the estimator,

$$ \hat y = X(X^{\top}X)^{-1}X^{\top}y = Hy ,$$

where $H = X(X^{\top}X)^{-1}X^{\top}$ is the **hat matrix** — so named because it
puts the hat on $y$. Geometrically it is the orthogonal projection onto the
column space of $X$: it takes the $n$-dimensional vector of observations and casts
its shadow onto the $p$-dimensional subspace the model can reach. That picture is
worth holding, because it makes two properties obvious rather than algebraic. $H$
is symmetric and **idempotent** ($HH = H$) — projecting a shadow again does
nothing, since it is already flat.

The residual vector is what the projection could not reach,

$$ e = y - \hat y = (I - H)y ,$$

and it is orthogonal to every column of $X$ **by construction**. That
orthogonality is the reason residual plots are informative *and* the reason they
can mislead: $e$ is guaranteed uncorrelated with each predictor whether or not
the model is right, so a flat plot of residuals against $x$ is not evidence of
anything. What is *not* guaranteed is the absence of non-linear structure — which
is exactly what you are looking for.

Now the consequence that governs every diagnostic below. If
$\operatorname{Var}(\varepsilon) = \sigma^2 I$, then

$$
\operatorname{Var}(e) = \sigma^2 (I - H),
\qquad\text{so}\qquad
\operatorname{Var}(e_i) = \sigma^2 (1 - h_{ii}).
$$

**The residuals do not have constant variance even when the errors do.** An
observation with large $h_{ii}$ has a residual squeezed towards zero as a matter
of arithmetic, not because the model fits it well.

> The most dangerous points therefore look like the best-behaved ones.

This is why raw residuals are the wrong thing to plot, and why everything below
is built on the **standardised residual**

$$ r_i = \frac{e_i}{s\sqrt{1 - h_{ii}}}, \qquad s^2 = \frac{e^{\top}e}{n-p},$$

which divides out the arithmetic squeeze and restores comparability.

### Residuals versus fitted

Plotting $r_i$ against $\hat y_i$ is the single most informative diagnostic there
is (Anscombe, 1963). Under an adequate model the cloud is structureless: flat,
centred on zero, of even width. **Curvature means the functional form is wrong.
A widening fan means the variance is not constant.** Both are visible instantly
and neither is visible in $R^2$.

Why against $\hat y$ rather than against $x$? Because with several predictors
there is no single $x$ to use, and $\hat y$ is the one-dimensional summary the
model actually commits to. It is also the axis along which heteroscedasticity
usually shows up, since the spread of a quantity so often scales with its level.

```python
import matplotlib.pyplot as plt
plt.scatter(lin.fittedvalues, lin.resid, s=16)
plt.axhline(0)
```

The coaster residuals are not formless. They **arch**: negative at small drops,
positive through the middle, negative again at the top. Binning converts the
impression into a table you can put in a report:

| drop band (m) | 0–25 | 25–45 | 45–65 | 65–85 | 85–105 | 105+ |
|---|---|---|---|---|---|---|
| mean residual (km/h) | −7.33 | +1.45 | +3.82 | +6.44 | +0.61 | −4.52 |

The sign changes twice. A straight line is being asked to trace a curve and is
splitting the difference. **More data would not fix this.** A different functional
form would.

Confirm it with a test rather than an eye:

```python
quad = smf.ols("speed_kmh ~ drop_m + I(drop_m**2)", data=g).fit()
quad.tvalues["I(drop_m ** 2)"]     # -8.57, p ~ 1e-12
```

`I(...)` tells the formula parser to treat its contents as arithmetic; without
it, `**` is read as formula syntax and you get an interaction.

> **The habit to build.** Never report a linear model without having looked at
> this plot, and never describe it as "fine". Say what you looked for — a change
> of sign, a fan, a cluster — and say whether you found it.

## 1.6 Logs, and the exponent they reveal

The curvature has a reason. A car falling a height $h$ converts potential energy
into kinetic energy, and at the bottom

$$ mgh = \tfrac{1}{2}mv^2 \quad\Longrightarrow\quad v = \sqrt{2gh} .$$

Speed goes as the *square root* of the drop. No straight line was ever going to
fit it.

The general form is $v = c\,h^{k}$, and logs turn it into something we can
already estimate:

$$ \log v = \log c + k \log h .$$

A straight line **whose slope is the exponent**:

```python
power = smf.ols("lv ~ lh", data=g).fit()
power.params["log_drop"]           # 0.4971
power.conf_int().loc["log_drop"]   # [0.4835, 0.5106]
np.exp(power.params["Intercept"])  # 15.17
```

The interval contains 0.5 and excludes 1. A regression has recovered a physical
law from a table of theme-park statistics.

The constant deserves a sentence too. Frictionless, it would be
$3.6\sqrt{2g} = 15.95$ with height in metres and speed in km/h. The data says
15.17, because real track loses energy. **Friction costs speed; it does not change
the exponent.** Being able to say which part of a discrepancy matters — and which
part is a nuisance that leaves your conclusion intact — is most of what
interpreting a model consists of.

This is the general lesson about transformations. A log is not a trick for making
a plot look nicer. In log–log form the coefficient is an **elasticity**: a 1% rise
in $x$ is associated with a $k$% rise in $y$. In log–linear form it is an
approximate proportional change per unit of $x$. Choose the form because of what
you want the coefficient to *mean*, and the diagnostics will usually agree with
you (Box & Cox, 1964).

## 1.7 Leverage: which observations *can* move the line

**Leverage** is the diagonal element $h_{ii}$ of the hat matrix, equal to
$\partial \hat y_i / \partial y_i$: how much the fitted value at observation $i$
responds to a change in the observed value at $i$.

That derivative reading is the one to carry. If $h_{ii} = 0.9$, moving that
observation's $y$ up by one unit drags its own fitted value up by 0.9 — the line
follows it almost exactly, which is another way of saying the line is not being
constrained by anything else in that region.

**Leverage is a property of $X$ alone.** It does not involve $y$ at all, so an
observation can have high leverage and fit perfectly.

Because $H$ projects onto a $p$-dimensional space,

$$ \sum_i h_{ii} = \operatorname{tr}(H) = p ,$$

so leverage is a **fixed budget of $p$** shared among $n$ observations. The
average is exactly $p/n$ and each $h_{ii}$ lies in $[0,1]$. The conventional flag
is $h_{ii} > 2p/n$ — twice the average — which is a rule of thumb and nothing
more (Hoaglin & Welsch, 1978).

In simple regression it has a transparent closed form,

$$ h_{ii} = \frac{1}{n} + \frac{(x_i - \bar x)^2}{\sum_j (x_j - \bar x)^2},$$

which says exactly what leverage means: a floor of $1/n$ everyone gets, plus a
term growing with squared distance from the centre of the predictor space,
measured in units of the predictors' own spread. On the coaster model the closed
form and the hat-matrix diagonal agree to $2\times10^{-16}$, and:

```
sum 2.0000000000  = p = 2      mean 0.025641 = p/n = 0.025641
min 0.01283 (at the x closest to the mean)      max 0.07042
```

Two consequences follow. Leverage is minimised at $x_i = \bar x$ and grows
quadratically away from it. And it is **relative** — the same observation is
high-leverage in a narrow sample and unremarkable in a wide one, so leverage is a
statement about the *design*, not about the observation.

> $\sum_i h_{ii} = p$ is not an empirical finding about this dataset. It is the
> trace of a projection onto a $p$-dimensional space, and it must hold exactly for
> every regression you ever fit. If your own implementation does not reproduce it,
> the implementation is wrong.

There is a fixed budget of influence and every observation gets a share.
Collecting more data does not create more; it divides the same total more finely,
which is the real reason large samples are robust to individual weird rows.

## 1.8 Influence: separating the outlier from the point that matters

**Cook's distance** asks how much the entire vector of fitted values moves when
observation $i$ is deleted:

$$
D_i = \frac{(\hat y - \hat y_{(i)})^{\top}(\hat y - \hat y_{(i)})}{p\,s^2}
= \frac{r_i^{2}}{p}\cdot\frac{h_{ii}}{1 - h_{ii}} .
$$

The second form is the one to internalise (Cook, 1977). Cook's distance is a
**product** of two quantities: how badly the point is fitted, $r_i^2$, and how
much power it has to pull, $h_{ii}/(1-h_{ii})$. On the coaster model the product
form reproduces `cooks_distance` exactly — to zero, not to floating point.

Because it is a product, either factor being near zero makes the whole thing near
zero — which is why the two things students conflate are genuinely different:

- **High residual, low leverage** — an outlier in the middle of the predictor
  range. It inflates $s$ and therefore widens every standard error in the model,
  but it barely moves $\hat\beta$, because it has no lever to pull on.
- **Low residual, high leverage** — a point far out in $X$ that the line happens
  to pass through. It is *supporting* your slope rather than distorting it, and no
  misfit diagnostic will flag it. But your slope now rests on it, which is a
  robustness problem precisely because nothing looks wrong.
- **High on both** — the case that changes conclusions, and what Cook's distance
  is built to find.

Note how the leverage factor behaves:

| $h_{ii}$ | 0.05 | 0.10 | 0.50 | 0.90 | 0.95 |
|---|---|---|---|---|---|
| $h/(1-h)$ | 0.05 | 0.11 | 1.00 | **9.00** | **19.00** |

Influence does not rise gently with leverage. **It explodes as $h_{ii}$
approaches 1.** Common cutoffs are $D_i > 1$ or $D_i > 4/n$; both are conventions,
and Belsley, Kuh & Welsch (1980) are properly sceptical about treating any of them
as a test.

The useful discipline is not the threshold. It is that you *look*, you *name* the
observations that stand out, and you report what happens to your conclusion with
and without them.

### And then go and read the rows

```python
inf = full.get_influence()
cooks = inf.cooks_distance[0]
d.iloc[np.argsort(cooks)[-6:][::-1]][["coaster", "drop_m", "speed_kmh", "propulsion"]]
```

Five of the six most influential coasters are **launched** — accelerated by
hydraulics or linear induction motors instead of being dropped. They reach
150–240 km/h from structures 12 to 46 metres high.

Energy conservation says nothing about them, because a motor put the energy in.
They are not bad measurements. They are a *different physical system*, and they
should never have been in the same regression.

Keeping them costs the entire result:

| | exponent | 95% interval |
|---|---|---|
| gravity only, $n = 78$ | **0.4971** | [0.4835, 0.5106] |
| all 84 coasters | 0.3722 | [0.2935, 0.4508] |

Six rows — seven per cent of the data — move the exponent from a law of physics to
a number corresponding to nothing, and the interval around 0.37 is narrow.
**Confidently wrong is the worst state a model can be in**, and it is the state
that reporting $R^2$ alone will leave you in.

> **The sequence to internalise.** A diagnostic points at rows. You read the rows.
> The rows reveal a mechanism. The mechanism justifies a decision. Cook's distance
> never told anyone to delete anything — it said *go and look at these six*.

Deleting an influential point silently is misconduct. Reporting that the six
launched coasters are influential, and that the exponent falls to 0.37 with them,
is a finding — and a more interesting one than the coefficient.

Note too that exclusion is not the only defensible answer. Fitting `propulsion`
as a group, and estimating how the two systems differ, uses all the data and
assumes less. It is arguably the better analysis. What is not defensible is
deleting rows because they were far from the line.

## 1.9 Normality, and why it matters least

Of the four assumptions this is the one checked first and the one that matters
least. It is not needed for unbiasedness, not needed for Gauss–Markov, and not
needed for the standard errors to be right. It is needed for the $t$ and $F$
statistics to follow *exactly* the distributions their tables assume — and only in
small samples, because the central limit theorem takes over as $n$ grows.
$\hat\beta$ is a weighted sum of the $y_i$, and sums tend to normality whatever
they are sums of.

The right check is a normal quantile–quantile plot, and what you look for is not
perfection but the **shape** of the departure. Points curving above the line at
the right and below at the left mean heavy tails — more extreme observations than
a normal would give — which is a robustness concern, since least squares weights
those heavily. A systematic S-shape means skew.

On the log–log coaster residuals, Shapiro–Wilk gives $W = 0.977$, $p = 0.175$: no
evidence against normality. Formal tests exist (Shapiro & Wilk, 1965) and their
weakness is instructive: with $n = 400$ they reject on a departure far too small
to affect any inference, while with $n = 20$ they fail to reject a departure large
enough to matter.

> This is the general pathology of diagnostic testing — the null is never exactly
> true, so the test measures sample size as much as misspecification. It is the
> reason this chapter keeps returning to plots.

## 1.10 When the variance is not constant

Be precise about what heteroscedasticity costs. Under it, $\hat\beta$ remains
**unbiased** and **consistent**. The coefficients are not wrong. What breaks is
$\operatorname{Var}(\hat\beta)$: the usual $s^2(X^{\top}X)^{-1}$ is no longer the
right formula, so the standard errors, $t$ statistics, intervals and $p$-values
are all computed from an expression that does not apply.

The modern response is not to model the variance but to stop relying on the
assumption. White's heteroscedasticity-consistent estimator replaces the
misapplied middle of the sandwich with the squared residuals themselves (White,
1980):

$$
\widehat{\operatorname{Var}}(\hat\beta)
= (X^{\top}X)^{-1}\Big(\textstyle\sum_i e_i^2\,x_i x_i^{\top}\Big)(X^{\top}X)^{-1}.
$$

The classical formula assumes every observation contributes the same $\sigma^2$
to the middle; the sandwich lets each contribute its own $e_i^2$. It is consistent
whatever the pattern of the variance, without your having to know or model that
pattern — which is why it is called *robust* rather than *efficient*. Its
small-sample behaviour is poor, because $e_i^2$ understates the variance at
high-leverage points, and the HC1/HC2/HC3 corrections exist to compensate
(MacKinnon & White, 1985). The recommendation is settled: **use HC3 by default**,
certainly whenever $n < 250$ (Long & Ervin, 2000). In `statsmodels` that is
`fit(cov_type="HC3")` — one argument, no good reason to omit it.

### The gap between the two is itself a diagnostic

The coaster data makes this vivid. On the **misspecified** model — all 84 rows,
launched coasters included — the robust standard error is far larger:

| all 84 coasters | $\hat\beta$ | se | 95% interval |
|---|---|---|---|
| classical | 0.3722 | 0.0395 | [0.2935, 0.4508] |
| HC3 | 0.3722 | **0.0643** | [0.2463, 0.4981] |
| clustered by park | 0.3722 | 0.0590 | [0.2565, 0.4878] |

On the **correct** model — gravity coasters only — all three agree:

| gravity only | $\hat\beta$ | se | 95% interval |
|---|---|---|---|
| classical | 0.4971 | 0.0068 | [0.4835, 0.5106] |
| HC3 | 0.4971 | 0.0065 | [0.4844, 0.5098] |
| clustered by park | 0.4971 | 0.0066 | [0.4842, 0.5099] |

> When the model is right, robust and classical standard errors agree. **A large
> gap between them is a signal that something is wrong with the specification**,
> not merely a reason to prefer the robust one. Report both and the reader learns
> something; report only the robust one and you have hidden a diagnostic.

A caution about tests. The Breusch–Pagan test regresses squared residuals on the
predictors and asks whether they explain anything (Breusch & Pagan, 1979). On the
linear coaster model it returns $p = 0.958$ — no heteroscedasticity detected — and
that model is badly wrong. **It is testing for the wrong failure.** The problem
there is curvature, and a variance test has no way to see it. For that, the RESET
procedure adds powers of the fitted values back into the regression and asks
whether they matter (Ramsey, 1969).

## 1.11 The assumption nobody plots

Independence has no standard diagnostic plot, which is most of why it goes
unexamined.

The coaster table is 84 rows, but it is not 84 independent draws: they sit in 12
parks, and coasters in the same park share a designer, a budget and a piece of
land. If the errors within a park are positively correlated, the sample carries
less information than its row count suggests, and the standard errors — computed
from that row count — come out too small.

```python
smf.ols("lv ~ lh", data=a).fit(cov_type="cluster",
                               cov_kwds={"groups": a.park})
```

The coefficient is identical in every row of both tables above — as promised,
dependence does not bias it. What changes is the interval.

This is Moulton's warning (1990), and it is the single most consequential
correction in applied panel work. Bertrand, Duflo & Mullainathan (2004) showed
the same mechanism producing spurious significance in difference-in-differences
studies at alarming rates — which chapter 6 returns to. Cameron & Miller (2015)
is the practitioner's reference for what to cluster on and when the cluster count
is too small to trust. The rule of thumb is to cluster at the level at which
treatment is assigned or shocks arrive.

> **Some assumptions are checked by looking at the data. Independence is checked
> by knowing how the data were collected.** A diagnostic chapter that stopped at
> residual plots would miss it entirely, because it does not show up in one.

## 1.12 Comparing models: what AIC is, and what it is not

Having diagnosed a problem you will fit an alternative and need a basis for
preferring one. $R^2$ cannot supply it.

Akaike's criterion comes at it from prediction rather than fit. It estimates the
expected Kullback–Leibler divergence between the fitted model and the process that
generated the data, and reduces to

$$ \mathrm{AIC} = -2\log\hat L + 2k $$

for a model with $k$ estimated parameters and maximised likelihood $\hat L$
(Akaike, 1974). Be careful with $k$: the variance $\sigma^2$ was estimated from
the data too, so a strict count includes it, while `statsmodels` counts only the
regression coefficients for OLS.

```
llf by hand -> 142.604279            statsmodels -> 142.604279
AIC with k=2 (coefficients only)  -281.2086     statsmodels -281.2086
AIC with k=3 (counting sigma^2)   -279.2086     <- differs by exactly 2
```

The two conventions differ by a constant, which is invisible in any comparison
and alarming the first time your own calculation disagrees with the library.

BIC replaces the penalty with $k\log n$ — harsher for any $n > 7$ — and aims at a
different target: the true model, if one is among the candidates, rather than the
best predictor (Schwarz, 1978). They disagree by construction, and which you use
should follow from which question you are asking.

Candidates on identical rows:

| AIC | adj $R^2$ | $R^2$ | model |
|---|---|---|---|
| **460.33** | 0.9835 | 0.9837 | `speed_kmh ~ sqrt(drop_m)` |
| 461.32 | 0.9835 | 0.9839 | `+ queue_min` |
| 461.72 | 0.9838 | **0.9846** | `+ length_m + inversions + queue_min` |
| 471.97 | 0.9811 | 0.9816 | `drop_m + I(drop_m**2)` |
| 523.22 | 0.9630 | 0.9635 | `drop_m` |

Read the second and third rows. `queue_min` is pure noise — how long the queue was
when somebody measured it — and adding it **raises $R^2$** from 0.9837 to 0.9839
while making AIC *worse*. The kitchen-sink model has the **highest $R^2$ on the
table** and is beaten on AIC by the one-variable square-root model. The
square-root form also beats the quadratic on one fewer parameter, which is the
tidier argument for it.

Three cautions, in order of how often they are violated.

**AIC is comparative and unitless.** A single value means nothing. Only
differences between models fitted to *the same observations* are interpretable —
so dropping rows through missing data silently invalidates the comparison, a
mistake that is easy to make and invisible afterwards. By convention
$\Delta_i = \mathrm{AIC}_i - \mathrm{AIC}_{\min}$ below about 2 is weak evidence
of any difference (Burnham & Anderson, 2004).

**AIC does not check adequacy.** It ranks the candidates you supply. If every one
is misspecified it will rank them and hand you a winner, in the same confident
tone it uses when one of them is right.

**Selecting a model and then reporting its $p$-values as though the specification
had been chosen in advance is invalid.** Freedman (1983) demonstrated the scale of
it with pure noise: regress a random $y$ on 50 random predictors, keep the
significant ones, refit, and the second regression looks impressive. Nothing was
there. The same mechanism operates whenever the data guide the specification —
the "garden of forking paths", which requires no dishonesty at all, only the
ordinary practice of looking at your data before deciding what to fit (Gelman &
Loken, 2014). If you select on AIC, say so, and treat the resulting inference as
exploratory.

## 1.13 When a diagnostic fails, what to actually do

Diagnosis without a remedy is just anxiety. The response depends entirely on
which assumption failed, which is why the table in §1.2 matters.

**Curvature in the residuals** — the functional form is wrong, and this is the
only failure that biases the coefficient, so it is the only one you must *fix*
rather than accommodate. Add a quadratic term, take logs of a variable whose
effect is plausibly proportional rather than additive, or fit a spline. Report
that you transformed, and why, before you report the result.

**Non-constant variance** — do not transform to fix it, since that changes the
quantity you are modelling in order to repair its error term. Use HC3 and say so.

**Dependence** — cluster at the level at which the dependence arises, or model it
explicitly with fixed effects. Chapter 2 does the latter.

**Heavy tails or an influential point** — first identify the observation and ask
what it *is*. An influential point is frequently the most informative observation
in the dataset rather than an error: the launched coasters are strange because
they are a different propulsion system, which is a fact about engineering, not a
data problem. Report the result with and without it. Delete only for a reason you
can state in the text, and state it there.

**Everything at once** — reconsider whether one linear model over pooled
heterogeneous units is the right object. Often it is not, and the honest answer is
a different specification rather than a patched one.

## 1.14 What to report

A diagnostic section that earns its place answers five questions in order, and
takes about a page.

1. **Is the functional form right?** Residuals against fitted, with a smoother.
   Say what structure you looked for and whether you found it.
2. **Is the variance constant?** Scale–location. If not, report HC3 and say that
   you did — do not quietly switch and leave the reader to notice the numbers
   moved.
3. **Are the observations independent?** Answered from how the data were
   collected, not from a plot. If they cluster, cluster the standard errors and
   report both.
4. **Does any observation carry the result?** Leverage against standardised
   residual, sized by Cook's distance. Name what stands out, identify it
   substantively, and give the coefficient with and without it.
5. **Would a different specification do better?** Compare on AIC over the same
   rows, report $\Delta$, and state plainly whether the specification was chosen
   before or after seeing the data.

The failure mode this chapter exists to prevent is reporting $R^2$ and stopping.
Anscombe's four datasets all have $R^2 = 0.67$. Three of them should never be
described by a straight line, and no amount of staring at 0.67 will tell you
which three.

## 1.15 Summary

1. A coefficient is a quantity with units. Say them. RSE is the more honest
   summary than $R^2$ because it has units too.
2. **The formula never refuses.** Least squares returns coefficients for any
   numbers you feed it, which is why diagnostics exist.
3. Normality is not needed for Gauss–Markov. Specification is the only failure
   that biases $\hat\beta$; non-constant variance and dependence destroy the
   uncertainty and leave the estimate alone.
4. Residuals do not have constant variance even when errors do. Plot
   **standardised** residuals against **fitted values**.
5. Leverage is capacity — a fixed budget summing to $p$ — and Cook's distance is
   damage, the product of misfit and leverage. Both are invitations to read the
   data, not instructions to change it.
6. Use HC3 by default, and treat a large gap between classical and robust
   standard errors as a diagnostic in its own right.
7. AIC ranks the candidates you supply and cannot tell you they are all wrong.
   Inference after selection is not the inference the $p$-value describes.

## Further reading

The written treatment of this chapter is
[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/),
chapters [2](https://warint.github.io/quantitative-methods/book/session-02.html)
and [3](https://warint.github.io/quantitative-methods/book/session-03.html).

The diagnostic tradition begins with **Anscombe (1963)**, who argued that
residuals should be examined systematically rather than summarised, and
**Anscombe (1973)**, whose four datasets share every conventional summary while
differing completely. The smoother that makes such plots readable is **Cleveland
(1979)**.

The geometry was formalised in the 1970s. **Hoaglin & Welsch (1978)** gave the hat
matrix its expository treatment and the $2p/n$ convention; **Cook (1977)**
introduced the distance separating outlying from influential observations; and
**Belsley, Kuh & Welsch (1980)** assembled the apparatus into a book, with a
scepticism about mechanical cutoffs that has aged well.

Robust inference developed in parallel: **White (1980)** for the covariance
estimator, **Breusch & Pagan (1979)** for the test, **MacKinnon & White (1985)**
for the small-sample corrections, and **Long & Ervin (2000)** for the evaluation
that made HC3 the applied default. Specification error is treated by **Ramsey
(1969)** and transformation by **Box & Cox (1964)**.

Dependence has a separate and more consequential literature: **Durbin & Watson
(1950)** for serial correlation, **Moulton (1990)** for clustering within groups,
**Bertrand, Duflo & Mullainathan (2004)** for the same failure in
difference-in-differences, and **Cameron & Miller (2015)** as the practical
reference.

Model comparison follows **Akaike (1974)** and **Schwarz (1978)**, whose criteria
differ in penalty and in target — prediction against identification — a
distinction **Burnham & Anderson (2004)** make carefully and much applied work
does not. The cost of ignoring it was quantified by **Freedman (1983)** and framed
for a modern audience by **Gelman & Loken (2014)**.

For the textbook treatments: **James, Witten, Hastie & Tibshirani (2021)**,
*An Introduction to Statistical Learning*, 2nd ed., ch. 3;
**Kutner, Nachtsheim, Neter & Li (2005)**, *Applied Linear Statistical Models*,
5th ed., ch. 3 and 10; **Wooldridge (2019)**, *Introductory Econometrics*, 7th
ed., ch. 2, 3 and 9.

> The through-line, from 1963 to now, is a single claim: **a fitted model is a
> hypothesis about the data, not a summary of it, and the residuals are where that
> hypothesis is tested.**
