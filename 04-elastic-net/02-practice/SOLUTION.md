# Session 4 · The Secret Recipe — answer key

*Read this after you have tried. Every number is asserted by
`scripts/verify_puzzles.py`.*

## The planted recipe

Seven ingredients, on standardised columns:

| ingredient | true effect |
|---|---|
| `vanilla_extract` | **+4.0** |
| `sea_salt` | **−3.2** |
| `brown_sugar` | +2.5 |
| `browned_butter` | +1.8 |
| `baking_soda` | +1.5 |
| `baking_powder` | +1.5 |
| `dutch_cocoa` | +1.1 |

Intercept 50, noise sd 2.5. `baking_soda` and `baking_powder` are correlated at
**0.9632** and have **identical** effects. The other 193 ingredients do nothing.

## Stage 1 · OLS

| | |
|---|---|
| training MSE | **0.00000000** |
| test MSE | 36.23 |
| variance of `y` | 35.45 |

$p/n = 3.3$, so $X^\top X$ is singular: there are infinitely many coefficient
vectors that fit the 39 training rows perfectly, and the fitted model interpolates
every point. **Test error exceeds the variance of the outcome** — this model
predicts worse than reporting the mean would.

## Stage 2 · Ridge

Chosen $\lambda = 0.01$, cross-validated MSE **28.50**.

- **200 of 200** coefficients non-zero
- **135** exceed 0.1 in absolute value

Ridge cannot answer "which ingredients matter", and not because it was tuned
badly. In the singular-value basis it multiplies each component by
$d_j^2/(d_j^2+\lambda)$, which is strictly between 0 and 1 for any finite
$\lambda$. **It can never reach zero.** It gives you a prediction and no list —
and here, a poor prediction, because the truth is sparse and ridge cannot
represent sparsity.

## Stage 3 · Lasso

Chosen $\lambda = 0.43$, cross-validated MSE **9.90** — about three times better
than ridge.

| ingredient | planted | lasso |
|---|---|---|
| `vanilla_extract` | +4.0 | +3.191 |
| `brown_sugar` | +2.5 | +1.341 |
| `sea_salt` | −3.2 | −2.224 |
| `browned_butter` | +1.8 | +1.322 |
| `dutch_cocoa` | +1.1 | **+0.042** |
| `baking_soda` | +1.5 | **0.000** |
| `baking_powder` | +1.5 | +1.942 |

**Selected 25 ingredients. Six of the seven true ones. Nineteen false positives.**

Every recovered coefficient is smaller than its true value. That is not an error
— shrinkage is what a penalty *does*, and it is the bias you accepted in exchange
for lower variance.

The 19 false positives are also not an error. **Cross-validation optimises
prediction**, and the penalty that predicts best is smaller than the penalty that
selects best. Keeping a handful of small spurious coefficients costs almost
nothing in predictive error.

> A lasso's non-zero set is not a list of true effects. "The model identified 25
> drivers of taste" is the commonest misuse of this method, and it is a misuse of
> a *correctly fitted* model.

## Stage 4 · The two leaveners — the heart of the exercise

The lasso gave `baking_soda` exactly **0.000** and `baking_powder` **+1.942**,
roughly the pair's combined shrunken effect loaded onto one member. It does this
in every split, with no sign of difficulty. Nothing in the output suggests a
second ingredient is hiding behind the first.

The geometry forces it: the $\ell_1$ penalty charges $|\beta_1|+|\beta_2|$, so
loading everything onto one column costs the same as splitting — and the
all-on-one solution is a **corner** of the $\ell_1$ diamond, which is where
solutions land. Ridge, charging $\beta_1^2+\beta_2^2$, does the opposite and
splits.

The elastic net at `l1_ratio=0.95` retains **both**: soda +0.374, powder +1.481.
That is the grouping effect.

**And then the question that matters.** Across five cross-validation seeds:

| CV seed | soda | powder | **sum** |
|---|---|---|---|
| 0 | +0.374 | +1.481 | **+1.855** |
| 1 | +0.786 | +1.203 | **+1.989** |
| 2 | +1.318 | +0.891 | **+2.209** |
| 3 | +0.699 | +1.248 | **+1.947** |
| 4 | +0.552 | +1.358 | **+1.910** |

The **sum is stable**. The **split is not** — it swings on an arbitrary choice of
random seed, at seed 2 nearly reversing. The lasso's total is +1.942, within
noise of the elastic net's.

> Two collinear columns carry **one identifiable quantity** between them. No
> penalty conjures a second. The elastic net's real service is telling you there
> are two ingredients here; it cannot tell you how to divide the credit, and a
> group that reported "soda +0.37, powder +1.48" as two findings has over-claimed.

The honest report: *"Leavening matters, with a combined effect of about +1.9. The
data cannot apportion it between soda and powder, because the two were varied
together."* The fix is not a better estimator — it is a better experiment.

## Stage 5 · What you cannot claim

**`dutch_cocoa` is the signal that got away.** True effect +1.1, estimated
between 0.00 and 0.25 depending on the split. At $n = 60$ with 200 candidates and
a noise sd of 2.5, it sits **below the detection floor**. No choice of method
recovers it, and any method that appeared to would have found noise.

Recognising this is worth more than correctly selecting variables. The right
sentence is *"there may be further ingredients with effects around 1 that this
design cannot detect; ruling them out would need more bakes, not a different
estimator."*

**Direction of bias:** towards zero, always. Penalised coefficients understate
effect sizes by construction.

**$p$-values:** there are none. The standard errors any library will print are
computed as though $\lambda$ and the selected set had been fixed in advance. They
were chosen by looking at the data, and that selection consumed evidence the
standard errors know nothing about. Honest inference needs post-selection
machinery or a clean sample split. **A table of lasso coefficients with stars
beside them is reporting inference that its own selection invalidated.**

## What full marks look like

- A shortlist naming vanilla, salt, brown sugar, browned butter, and leavening
- The leavening pair reported as **one combined quantity**, with the seed
  instability as evidence
- `dutch_cocoa` — or the general possibility of undetectable weak effects —
  flagged as beyond the design
- Shrinkage acknowledged, so coefficients are not read as effect sizes
- No $p$-values, with a sentence on why
- The 19 false positives explained as CV optimising prediction, not truth

## Common wrong answers

| Reported | What happened |
|---|---|
| "25 ingredients drive taste" | read the selected set as a list of true effects |
| "baking powder matters, soda does not" | believed the lasso's arbitrary choice |
| "soda +0.37 and powder +1.48" | reported an unidentified split as two findings |
| ridge coefficients as a ranking | ridge cannot select; the ranking is unstable |
| coefficients with stars | inference invalidated by the selection |
| "the method found everything" | never checked for the signal it missed |
