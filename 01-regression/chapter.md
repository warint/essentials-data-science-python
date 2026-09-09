# 1 · Regression

> There is a law of physics hiding in a spreadsheet of theme-park statistics.
> A regression can find it. The same regression, fitted slightly differently,
> reports a number that means nothing — and reports it with an excellent $R^2$.

## 1.1 What least squares is doing

We assume a straight-line relationship, and allow every observation to miss it:

$$ y_i = \beta_0 + \beta_1 x_i + \varepsilon_i .$$

Least squares chooses the intercept and slope that minimise the sum of squared
vertical distances between the points and the line:

$$ \hat\beta = \arg\min_{\beta}\sum_{i=1}^{n}\bigl(y_i - \beta_0 - \beta_1 x_i\bigr)^2 .$$

Two choices in that sentence are doing work, and both are worth noticing before
you use the result.

**Squared.** A miss of 4 counts four times as much as a miss of 2, not twice.
This is what makes the arithmetic tractable, and it is also why a single badly
predicted observation can matter so much — a theme this chapter returns to at
length.

**Vertical.** The distances are measured in the $y$ direction only. The method
assumes $x$ is known and $y$ carries the error. If both are measured with noise,
least squares under-states the slope, and the fix is a different method rather
than a different formula.

## 1.2 Reading the output

A fitted model hands you three numbers per variable. You need all three.

```python
import essentials as ess
import statsmodels.formula.api as smf

d = ess.load("coasters")
g = d[d.propulsion == "gravity"].copy()

lin = smf.ols("speed_kmh ~ drop_m", data=g).fit()
print(lin.summary().tables[1])
```

The **coefficient** is a quantity with units: 0.94 km/h per metre of drop. It
says that a coaster whose first drop is ten metres taller is predicted to be
about 9.4 km/h faster. A coefficient reported without its units is not a
finding, and the units are the first thing a referee will ask for.

The **standard error** is the honest part of the output. It says how much the
coefficient would wobble if you collected the data again from the same process.
Everything you are entitled to claim about precision comes from here.

The **$t$-statistic** is the coefficient divided by its standard error: how many
wobbles the estimate sits from zero. It is a ratio, so it has no units, and on
its own it tells you nothing about whether an effect is large enough to care
about. With enough rows, an effect too small to matter clears any threshold you
like.

### The $p$-value, stated carefully

> If the true effect were exactly zero, and this study were repeated many times,
> what fraction of those repetitions would produce a coefficient at least this
> far from zero?

That is the whole content of a $p$-value. It is a statement about data under an
assumption — not about the probability that the assumption is true, and not
about the size of anything. Two studies reporting $p = 0.04$ and $p = 0.002$
tell you nothing about which effect is larger.

## 1.3 $R^2$ measures closeness, not correctness

$R^2$ is the share of the variance in $y$ that the fitted values reproduce:

$$ R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2} .$$

It runs from 0 to 1, and it rises whenever you add a variable — including a
column of random numbers. Adjusted $R^2$ penalises the extra parameter, which
helps a little and is not a defence.

The deeper problem is not inflation. It is that $R^2$ answers a question about
*distance* and people read it as an answer about *form*. A model can sit very
close to the data and still have the shape wrong, and when that happens $R^2$
gives no signal at all.

On the 78 gravity-driven coasters, the straight line achieves $R^2 = 0.9635$.
The model is wrong. Here is how you find that out.

## 1.4 Residuals versus fitted

A residual is what the model failed to explain: $e_i = y_i - \hat y_i$. Plot the
residuals against the fitted values, and ask one question: **is there a
pattern?**

If the model has the form right, there is none. The residuals are a formless
band around zero, because everything with structure in it has already been
absorbed by the fit. Anything you can see in that plot is signal the model left
behind.

```python
import matplotlib.pyplot as plt

plt.scatter(lin.fittedvalues, lin.resid, s=16)
plt.axhline(0)
plt.xlabel("Fitted top speed (km/h)")
plt.ylabel("Residual (km/h)")
plt.show()
```

The coaster residuals are not formless. They arch: negative at small drops,
positive through the middle, negative again at the top. Binning makes it plain,
and binning is worth doing because it converts an impression into a table you
can put in a report:

| drop band (m) | 0–25 | 25–45 | 45–65 | 65–85 | 85–105 | 105+ |
|---|---|---|---|---|---|---|
| mean residual (km/h) | −7.33 | +1.45 | +3.82 | +6.44 | +0.61 | −4.52 |

The sign changes twice. A straight line is being asked to trace a curve and is
splitting the difference — over-predicting at the ends, under-predicting in the
middle. More data would not fix this. A different functional form would.

Confirm it with a test rather than an eye:

```python
quad = smf.ols("speed_kmh ~ drop_m + I(drop_m**2)", data=g).fit()
quad.tvalues["I(drop_m ** 2)"]     # -8.57, p ~ 1e-12
```

`I(...)` tells the formula parser to treat what is inside as arithmetic. Without
it, `**` would be interpreted as formula syntax and you would get an
interaction.

> **The habit to build.** Never report a linear model without having looked at
> this plot, and never describe it as "fine". Say what you looked for — a change
> of sign, a fan, a cluster — and say whether you found it.

## 1.5 Logs, and the exponent they reveal

The curvature has a reason. A car falling a height $h$ converts potential energy
into kinetic energy, and at the bottom

$$ mgh = \tfrac{1}{2}mv^2 \quad\Longrightarrow\quad v = \sqrt{2gh} .$$

Speed goes as the *square root* of the drop. No straight line was ever going to
fit that.

The general form is $v = c\,h^{k}$, and logs turn it into something we can
already estimate:

$$ \log v = \log c + k \log h .$$

A straight line, **whose slope is the exponent**. So:

```python
import numpy as np

g["log_speed"], g["log_drop"] = np.log(g.speed_kmh), np.log(g.drop_m)
power = smf.ols("log_speed ~ log_drop", data=g).fit()

power.params["log_drop"]           # 0.4971
power.conf_int().loc["log_drop"]   # [0.4835, 0.5106]
np.exp(power.params["Intercept"])  # 15.17
```

The interval contains 0.5 and excludes 1. A regression has recovered a physical
law from a table of theme-park statistics.

The constant deserves a sentence too. Frictionless, it would be
$3.6\sqrt{2g} = 15.95$ with height in metres and speed in km/h. The data says
15.17, because real track loses energy to friction. **Friction costs speed; it
does not change the exponent.** Being able to say which part of a discrepancy
matters — and which part is a nuisance that leaves your conclusion intact — is
most of what interpreting a model consists of.

This is also the general lesson about transformations. A log is not a trick for
making a plot look nicer. In `log-log` form, the coefficient is an **elasticity**:
a 1% rise in $x$ is associated with a $k$% rise in $y$. In `log-linear` form, the
coefficient is an approximate proportional change per unit of $x$. Choose the
form because of what you want the coefficient to *mean*, and the diagnostics will
usually agree with you.

## 1.6 Leverage, outliers, and influence

Three different ideas that get collapsed into the word "outlier":

- an **outlier** has an unusual $y$ given its $x$ — it is badly predicted;
- a **leverage point** has an unusual $x$ — it has the *capacity* to move the line;
- an **influential** observation actually moves it, which needs both.

A point far from the centre of $x$ that sits exactly on the line does no damage.
A badly predicted point in the middle of the data does very little either. Damage
requires both properties at once, which is why they are diagnosed together.

### Leverage

Leverage is the diagonal of the hat matrix $H = X(X^\top X)^{-1}X^\top$, and
$h_{ii}$ measures how strongly $\hat y_i$ responds to $y_i$. Its total is not a
matter of the data:

$$ \sum_i h_{ii} = \operatorname{trace}(H) = p .$$

```python
inf = power.get_influence()
inf.hat_matrix_diag.sum()      # 2.000000, exactly
```

There is a **fixed budget of influence**, and every observation gets a share of
it. Collecting more data does not create more; it divides the same total more
finely, which is the real reason large samples are robust to individual weird
rows. The conventional screen is $h_{ii} > 2p/n$ — twice the average.

### Cook's distance

Cook's distance asks the question you actually care about: if this row were
deleted, how far would *all* the fitted values move?

$$ D_i = \frac{\sum_j\bigl(\hat y_j - \hat y_{j(i)}\bigr)^2}{p\,\hat\sigma^2} $$

It combines leverage with residual size, which is why it is the single number to
look at. The usual screen is $D_i > 4/n$, and it is a prompt to investigate, not
a rule for deleting.

```python
cooks = inf.cooks_distance[0]
top = np.argsort(cooks)[-6:][::-1]
d.iloc[top][["coaster", "drop_m", "speed_kmh", "propulsion"]]
```

### And then go and read the rows

Five of the six most influential coasters are **launched** — accelerated by
hydraulics or linear induction motors instead of being dropped. They reach
150–240 km/h from structures 12 to 46 metres high.

Energy conservation says nothing about them, because a motor put the energy in.
They are not bad measurements. They are a *different physical system*, and they
should never have been in the same regression.

Keeping them costs the entire result:

| | exponent |
|---|---|
| gravity only, $n = 78$ | **0.4971** |
| all 84 coasters | 0.3722 |

Six rows — seven per cent of the data — move the exponent from a law of physics
to a number corresponding to nothing, and the confidence interval around 0.37 is
narrow. **Confidently wrong is the worst state a model can be in**, and it is the
state that reporting $R^2$ alone will leave you in.

> **The sequence to internalise.** A diagnostic points at rows. You read the
> rows. The rows reveal a mechanism. The mechanism justifies a decision. Cook's
> distance never told anyone to delete anything — it said *go and look at these
> six*.

Note that exclusion is not the only defensible answer here. Fitting `propulsion`
as a group, and estimating how the two systems differ, uses all the data and
assumes less. It is arguably the better analysis. What is not defensible is
deleting rows because they were far from the line.

## 1.7 What to report

- The coefficient, in units, with its standard error or interval.
- The functional form, and why that one — not merely that it fit better.
- What you looked for in the residuals, and what you found.
- Which observations you investigated, what they turned out to be, and what you
  decided.
- What the result does not license you to claim.

$R^2$ is the least informative number in the output and the one most often
reported on its own.

## 1.8 Summary

1. A coefficient is a quantity with units. Say them.
2. $R^2$ measures closeness, not correctness.
3. Residuals versus fitted, every time. Curvature there means the form is wrong,
   and more data will not help.
4. Logs turn a power law into a line whose slope is the exponent, and make the
   coefficient an elasticity.
5. Leverage is capacity, Cook's distance is damage, and both are invitations to
   read the data rather than instructions to change it.

## Further reading

- James, Witten, Hastie & Tibshirani (2021), *An Introduction to Statistical
  Learning*, 2nd ed., ch. 3.
- Kutner, Nachtsheim, Neter & Li (2005), *Applied Linear Statistical Models*,
  5th ed., ch. 3 and 10.
- Wooldridge (2019), *Introductory Econometrics*, 7th ed., ch. 2, 3 and 9.
