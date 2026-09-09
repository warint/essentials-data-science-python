# 5 · Classification

> The only chapter where you can see the right answer. This data was generated
> from known distributions, so the best achievable accuracy is computable and the
> true decision boundary can be drawn underneath your own.

## 5.1 The best possible classifier

If you knew the true class probabilities at every point, the best rule would be
obvious: predict the most likely class.

$$ \hat y(x) = \arg\max_j \Pr(Y = j \mid X = x) $$

That is the **Bayes classifier**. The proof that nothing beats it is a one-liner:
at each $x$, any rule either picks the most probable class or does not, and
picking it minimises the chance of being wrong *there*; integrating over $x$ gives
the result.

It is not a method, because computing it requires $\Pr(Y \mid X)$ — which is the
thing you are trying to estimate. Its value is conceptual and it is considerable:

> **A non-zero Bayes rate means that error is not evidence of a bad model.** If
> the outcome is genuinely uncertain given the predictors, no amount of
> flexibility, data or cleverness removes the remaining error, and a practitioner
> who keeps adding complexity to chase it is chasing noise.

The regression analogue is the irreducible error $\sigma^2$: the variance of $y$
around $f(x)$, which no estimator of $f$ can touch.

```python
import essentials as ess
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

d = ess.load("bestiary")
species = ess.bestiary_species()
post = ess.bestiary_posterior(d.wing_span_cm, d.tail_length_cm)
bayes = np.array(species)[post.argmax(1)]

accuracy_score(d.species.astype(str).to_numpy(), bayes)   # 0.8625
```

| | |
|---|---|
| **Bayes accuracy** | **0.8625** |
| Bayes balanced accuracy | 0.8876 |

Two distinct reasons it is not 1.0:

1. **The classes overlap.** The four Gaussians share support, so some points are
   genuinely more likely to have come from a species other than the one that
   produced them. No amount of information resolves that.
2. **7% of the labels are wrong.** Even a perfect rule disagrees with the
   recorded label that often.

The two averages differ because accuracy weights classes by size while balanced
accuracy weights them equally — so the 2% species contributes 2% of one number
and 25% of the other.

> **A test accuracy of 0.98 here would be a bug report, not an achievement.** It
> would mean test rows leaked into training, or a feature encodes the answer.
> Knowing the ceiling turns a high score into a question.

Most real problems do not hand you the floor. Estimating one anyway — from a
majority-class baseline, from measurement reliability, from what a trained human
achieves — turns an unanchored number into an interpretable one.

## 5.2 $k$-nearest neighbours

The method is due to Evelyn Fix and Joseph Hodges, in a 1951 technical report for
the US Air Force School of Aviation Medicine that went unpublished for nearly
forty years. Cover and Hart (1967) then proved the result that made it
respectable: **as the sample grows, the error rate of the 1-nearest-neighbour
rule is bounded above by twice the Bayes rate.** The simplest possible classifier,
using one point, is asymptotically within a factor of two of the best anything can
do.

You can check the bound holds here:

| | |
|---|---|
| Bayes error | 0.1375 |
| twice the Bayes error | 0.2750 |
| **1-NN test error** | **0.2375** |

Satisfied, and not by much — which is the honest reading. A guarantee of "within
a factor of two of optimal" is a real guarantee and a poor target.

The whole algorithm: to classify a new point, find the $k$ closest training
points and take a vote.

$$ \hat{\Pr}(Y = j \mid x) = \frac{1}{k}\sum_{i \in N_k(x)} \mathbb{1}(y_i = j) $$

- **No fitting.** No coefficients, no likelihood, no optimisation.
- **No functional form.** It assumes nothing about linearity or normality.
- **Everything is stored.** Prediction costs a search over the training set.

The training data *is* the model.

This looks like getting something for nothing, and it is worth being precise
about what it actually costs. **The method is not assumption-free; it has moved
its assumption.** Averaging the neighbours presumes the function is smooth enough
that points close in $x$ have similar $y$, and it presumes **you know what
"close" means**. Both presumptions can fail badly, and neither shows up as a
diagnostic plot.

### Effective degrees of freedom

For $k$-nearest neighbours the effective number of parameters is roughly $n/k$:

| $k$ | 1 | 5 | 21 | 45 | 125 | 400 | $n$ |
|---|---|---|---|---|---|---|---|
| $n/k$ | **1120** | 224 | 53 | 25 | 9 | 3 | **1** |

At $k = 1$ the method has as many effective parameters as observations, which is
why it interpolates the training data exactly; at $k = n$ it has one, and predicts
the global majority everywhere. It is the same quantity chapter 4's penalty
controlled continuously.

### Standardise, or the units decide

$k$-NN is a distance method, and in this data `mass_g` has a standard deviation
nearly forty times that of tail length. In Euclidean distance it would dominate
every other feature — and it is pure noise. **A distance method on
unstandardised features is measuring whichever variable happens to have the
biggest numbers.**

Fit the scaler on the **training** data only. Fitting it on everything leaks the
test set's mean and variance into the model.

## 5.3 Where prediction error comes from

Take a target $y = f(x) + \varepsilon$ with $\operatorname{E}[\varepsilon] = 0$
and $\operatorname{Var}(\varepsilon) = \sigma^2$, and let $\hat f$ be an
estimate built from a random training sample. The expected squared error at a
point $x_0$ splits into three:

$$
\operatorname{E}\big[(y_0 - \hat f(x_0))^2\big]
= \underbrace{\sigma^2}_{\text{irreducible}}
+ \underbrace{\big(\operatorname{E}[\hat f(x_0)] - f(x_0)\big)^2}_{\text{bias}^2}
+ \underbrace{\operatorname{Var}\big(\hat f(x_0)\big)}_{\text{variance}} .
$$

Chapter 4 used the last two terms to argue for a biased estimator. Here all three
matter, and **the first is what makes the others meaningful**: because
$\sigma^2 > 0$ the total error has a floor, and the only quantity a modeller
controls is how the remaining budget is split.

The words are worth being precise about. **Bias** is how far the method's
*average* prediction is from the truth — an error it makes systematically, in
every sample. **Variance** is how much the prediction moves when the training
sample changes — an error it makes differently each time. A rigid method has high
bias and low variance; a flexible one has the reverse.

For nearest neighbours, $k$ is the dial. Small $k$ averages few points, follows
the data closely, and moves a lot when the data change: low bias, high variance.
Large $k$ averages many points, smooths across regions where the truth differs,
and barely moves between samples: high bias, low variance.

### Measured, not illustrated

The decomposition is usually presented as algebra and left there. Here the truth
is known, so it can be measured. Draw 40 independent training samples from the
same generator, fit each, and predict at 300 fixed query points:

```python
for r in range(40):
    samp = ess._make_bestiary(seed=90000 + r)     # a fresh draw from the same world
    ...
modal = the most common prediction across the 40 fits, at each query point
bias  = how often that modal prediction differs from the Bayes label
var   = how often a single fit differs from its own modal prediction
```

| $k$ | bias | variance | total disagreement with the Bayes label |
|---|---|---|---|
| 1 | 0.0867 | **0.1462** | 0.1778 |
| 5 | 0.0900 | 0.0469 | 0.1098 |
| **15** | 0.1000 | 0.0279 | **0.1032** |
| 45 | 0.1100 | 0.0197 | 0.1194 |
| 125 | 0.1967 | 0.0126 | 0.1988 |
| 400 | **0.2133** | 0.0126 | 0.2139 |

Read the columns rather than the shape. **Variance falls by a factor of twelve**
between $k=1$ and $k=400$; **bias rises by a factor of two and a half**. The total
is U-shaped with a minimum in between.

Neither extreme is a modelling failure. Both are the same dial at different
settings — and the dial is the only thing you control.

## 5.4 The trade, made visible on one split

| $k$ | train | test |
|---|---|---|
| 1 | **0.999** | **0.762** |
| 5 | 0.866 | 0.850 |
| 9 | 0.860 | 0.871 |
| **45** | 0.855 | **0.873** |
| 125 | 0.838 | 0.854 |
| 400 | 0.829 | 0.850 |

The two columns have **different shapes**, and that is the whole lesson. Training
accuracy falls monotonically as $k$ rises: more smoothing always fits the
training data worse. Test accuracy is a hill.

Drawing the boundary on a mesh, with the true boundary contoured on top, turns
those numbers into something you can look at:

- **$k = 1$** — the regions shatter into islands, many of them a single
  mislabelled creature. The classifier has fitted the 7% label noise. Perfect on
  training data, worst on new data. **Variance.**
- **$k = 45$** — the regions track the true boundary. Close to the truth.
- **$k = 400$** — smooth, in the wrong place, and the rare species has vanished
  from the map. **Bias.**

The peak (0.873) sits just above the 0.8625 floor. That is sampling noise on a
480-row test set — one standard error is about 0.015 — not a violation. The floor
is a population quantity; a test estimate wobbles around it.

### Choosing $k$ honestly

You cannot choose $k$ on the test set: doing so makes the test score an estimate
of nothing. Cross-validate within the training data.

```python
from sklearn.model_selection import GridSearchCV, KFold

grid = GridSearchCV(KNeighborsClassifier(),
                    {"n_neighbors": list(range(1, 151, 2))},
                    cv=KFold(10, shuffle=True, random_state=0)).fit(A, ytr)
grid.best_params_      # 21
```

CV picked 21, not the 45 that happened to win on the test set. Both land within a
point of the floor and the difference is noise — so picking the test-set winner
would have been cheating for nothing.

## 5.5 The confusion matrix, and the 2% species

Accuracy at $k = 45$ is 0.873. The matrix (rows = truth):

| | → dusk_moth | → ember_drake | → glimmerwing | → thornback |
|---|---|---|---|---|
| **dusk_moth** | 125 | 0 | 6 | 17 |
| **ember_drake** | 0 | **8** | 0 | 2 |
| **glimmerwing** | 3 | 0 | 141 | 14 |
| **thornback** | 15 | 0 | 4 | 145 |

Now the measurement that matters. Take the predictions and replace every
`ember_drake` with the most common class:

| | accuracy |
|---|---|
| the model as fitted | 0.8729 |
| the same model, blind to ember drakes | 0.8562 |
| **cost of total blindness** | **0.0167** |

> **Complete blindness to the rare species costs 1.7 percentage points.** Anything
> optimising accuracy has almost no incentive to find it. If that species is why
> you built the classifier, accuracy is measuring the wrong thing.

Note too that the test set holds only **10** drakes. A recall of 0.800 means
"8 out of 10", and its standard error is enormous. **You cannot evaluate a rare
class on a small test set**, and saying so is part of an honest report.

### Precision, recall, and which one your problem needs

| | | |
|---|---|---|
| **Precision** | $TP/(TP+FP)$ | of the ones I flagged, how many deserved it? |
| **Recall** | $TP/(TP+FN)$ | of the ones that existed, how many did I find? |

For `ember_drake`: precision 1.000, recall 0.800. Every drake it named was a
drake; it found four in five. Whether that is good depends on what happens next:

- Cataloguing a rare species for conservation → **recall**. A missed drake goes
  unprotected; a false alarm costs a field visit.
- Triggering an expensive survey → **precision**. A false alarm burns budget the
  real drakes needed.

`macro avg` weights classes equally; `weighted avg` reproduces accuracy. On
imbalanced data, the macro average or balanced accuracy is usually the honest
headline.

## 5.6 Distance is chosen, not given

Nearest neighbours needs a definition of "near". The default is Euclidean
distance,

$$ d(x_0, x_i) = \sqrt{\textstyle\sum_{j=1}^{p}(x_{0j} - x_{ij})^2},$$

and that formula **adds squared differences measured in different units.** A
difference of one gram and a difference of one centimetre are treated as the same
quantity. In practice the variable with the largest numerical range determines the
neighbours entirely and every other predictor is decoration — and in this data
`mass_g` has a standard deviation nearly forty times that of tail length, and is
pure noise.

> This is the third time the same conclusion has arrived — the penalties in
> chapter 4, the units in chapter 1, and now here — and the reason is always the
> same. **Any method that compares magnitudes across variables must first make
> them comparable.** The alternative is a method whose answer depends on whether
> you recorded mass in grams or kilograms.

Fit the scaler on the **training** data only. Fitting it on everything leaks the
test set's mean and variance into the model.

## 5.7 The curse of dimensionality

The deeper problem is that "near" stops existing as the number of predictors
grows, and it does so much faster than intuition suggests.

Draw 600 points and one query point from a standard normal in $p$ dimensions, and
ask how much further the farthest point is than the nearest:

| $p$ | nearest | farthest | $(\text{far}-\text{near})/\text{near}$ |
|---|---|---|---|
| 1 | 0.001 | 3.594 | **2414** |
| 2 | 0.084 | 4.863 | 56.8 |
| 5 | 0.526 | 5.447 | 9.4 |
| 10 | 1.545 | 7.175 | 3.7 |
| 50 | 6.759 | 12.109 | 0.79 |
| 200 | 16.188 | 22.358 | 0.38 |
| 1000 | 41.448 | 47.261 | **0.14** |

At $p = 1000$ the farthest of 600 points is only 14% further away than the
nearest. All the points are approximately equidistant from the query, so "the ten
nearest" is close to **a random selection of ten** (Beyer et al., 1999).

There is a second, geometric way to see the same thing. To capture a fraction $r$
of the data in a uniform hypercube of $p$ dimensions, the neighbourhood must span
$r^{1/p}$ of the range of **each** variable:

| $p$ | 1 | 2 | 5 | 10 | 50 | 100 |
|---|---|---|---|---|---|---|
| range needed for 1% of the data | 0.010 | 0.100 | 0.398 | 0.631 | **0.912** | **0.955** |
| range needed for 10% | 0.100 | 0.316 | 0.631 | 0.794 | 0.955 | 0.977 |

Capturing 1% of the data needs 10% of the range in two dimensions, 63% in ten,
and **95% in a hundred** — a "neighbourhood" covering almost the entire space,
which is not local in any useful sense.

> The consequence for practice is blunt. Nearest neighbours works well with few
> predictors and abundant data, and degrades quickly as predictors are added.
> **Adding a variable to a linear model costs one parameter; adding one to a
> nearest-neighbour method costs a dimension of the space in which distance must
> remain meaningful.**

## 5.8 What three useless columns cost

`mass_g`, `antenna_mm` and `scale_count` contain no information about species.

| $k$ | 2 real features | + 3 noise columns |
|---|---|---|
| 5 | 0.850 | 0.838 |
| **45** | **0.873** | **0.846** |
| 125 | 0.854 | 0.846 |

Nearly three percentage points at the optimal $k$, from columns that say nothing.

The reason is specifically about distance. $k$-NN asks "who is nearby?" In five
dimensions, two creatures of the same species can be pushed apart by nothing but
a difference in mass, and **distance in an irrelevant dimension is still
distance**. The neighbourhood fills with the wrong neighbours. This is the curse
of dimensionality in miniature: as dimensions grow, all points become roughly
equidistant and "nearby" stops meaning anything.

A penalised regression would have shrunk those coefficients towards zero and lost
almost nothing. **$k$-NN has no coefficients, so it cannot ignore a feature you
hand it.** Feature selection therefore matters more for distance methods than for
penalised linear ones.

### Importance without coefficients

To learn what a feature is worth to a model with no coefficients, scramble it and
see what breaks:

```python
from sklearn.inspection import permutation_importance

imp = permutation_importance(model, X_test, y_test, n_repeats=12, random_state=0)
```

| feature | importance |
|---|---|
| `tail_length_cm` | 0.313 |
| `wing_span_cm` | 0.263 |
| `mass_g` | 0.006 |
| `antenna_mm` | 0.004 |
| `scale_count` | 0.001 |

Permutation importance is model-agnostic: it works for anything with a
`.predict`, which is exactly what you need here.

## 5.9 Did the flexibility earn its keep?

Now the real question. Nearest neighbours is flexible; flexibility costs variance;
and **the only justification for paying is that the truth is non-linear in a way a
rigid model misses.** Whether that holds is an empirical question, answered by
comparison rather than by assertion.

| model | train | test |
|---|---|---|
| logistic regression (linear) | 0.854 | 0.867 |
| **QDA (the correct family)** | 0.853 | **0.875** |
| tree, depth 3 | 0.825 | 0.844 |
| tree, unlimited | **0.999** | **0.765** |
| $k$-NN, $k=45$ | 0.855 | 0.873 |

- **QDA is the right family** — the generator is Gaussians with unequal
  covariances — and it reaches the floor by knowing the answer's shape.
- **$k$-NN gets there without knowing anything.** That is the case for
  non-parametric methods, and it is a real one.
- **The linear boundary loses only 0.006.** Worth a sentence, because it
  contradicts the obvious inference: "the true boundary is curved" does not
  automatically mean "a linear model will fail". Curvature only costs you where
  the data actually is.
- **The unlimited tree is $k=1$ in different clothing.** Two models can score
  identically for opposite reasons, and a single number hides which.

So on this data the flexibility roughly broke even: $k$-NN matched QDA and beat
the linear model by six thousandths, which is inside the noise of a 480-row test
set. **The honest report is that it did not clearly earn its keep** — and that is
a finding, not a failure of the exercise. Had the comparison been made on
accuracy alone, all five models would have looked within three points of each
other and the conclusion would have been that the choice does not matter, which
is a different and less useful claim.

## 5.10 What nearest neighbours cannot do

**It cannot explain.** There are no coefficients, so there is nothing to
interpret, no marginal effect, no test of a hypothesis about a mechanism. It is a
prediction machine.

**It cannot extrapolate.** A query outside the range of the training data is
answered by whichever training points happen to be least far away — which will be
the boundary points, no matter how far outside you go. The prediction is
confidently wrong and nothing signals it.

**It is slow where it matters.** Fitting is free and predicting is expensive: the
entire training set must be searched for every query. That is the opposite of the
profile you usually want in production.

**It is exposed to irrelevant variables.** Adding a predictor that carries no
information does not merely fail to help — it actively corrupts the distances,
diluting the informative dimensions. §5.8 measured that at nearly three points of
accuracy. Chapter 4's penalties can shrink a useless variable to zero; **a
distance cannot.**

## 5.11 What to report

- The **confusion matrix**, always. Accuracy alone is not a result.
- Accuracy against a **baseline** — the majority class, and a computed floor
  where one exists.
- **Per-class precision and recall**, and which of the two your problem needs.
- How the hyperparameter was chosen, and on **which** data.
- What the model cannot do.

## 5.12 Summary

1. Know the ceiling if you can compute one. **A non-zero Bayes rate means error
   is not evidence of a bad model.**
2. $k$-NN has no fit, no form and no coefficients — but it is not
   assumption-free. It assumes smoothness, and it assumes you know what "close"
   means.
3. $k$ is the bias–variance dial, with roughly $n/k$ effective parameters. Here,
   measured over 40 independent samples: **variance falls twelvefold, bias rises
   two-and-a-half-fold**, and the total is U-shaped.
4. Cover & Hart: 1-NN is asymptotically within **twice** the Bayes rate. A real
   guarantee and a poor target.
5. **Standardise.** Distance adds squared differences in different units, so
   without it the biggest numbers decide. Fit the scaler on training data only.
6. "Near" stops existing as dimensions grow. At $p = 1000$ the farthest of 600
   points is 14% further than the nearest, and a 1% neighbourhood spans 95% of
   every variable's range at $p = 100$.
7. Accuracy hides rare classes; ignoring a 2% class costs 1.7 points.
8. Precision or recall — the problem decides.
9. Irrelevant features punish distance methods, and **$k$-NN cannot ignore a
   feature you hand it.** A penalty can shrink it to zero; a distance cannot.
10. Flexibility is a purchase. Say what it bought — and if a rigid model does as
    well, that is the finding.

## Further reading

The written treatment is
[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/),
chapter [8 · K-Nearest Neighbours and the Bias–Variance Trade-off](https://warint.github.io/quantitative-methods/book/session-08.html).

The method originates in an unpublished 1951 technical report by **Fix & Hodges**,
whose belated reprinting in 1989 is the citable version. **Cover & Hart (1967)**
proved the result that made it respectable: as the sample grows, the error rate of
the 1-nearest-neighbour rule is bounded above by twice the Bayes rate — so the
simplest possible classifier, using one point, is asymptotically within a factor
of two of the best any method can do. **Stone (1977)** established the conditions
under which nearest-neighbour regression is consistent, completing the
theoretical case.

The trade-off itself is stated in its modern form by **Geman, Bienenstock &
Doursat (1992)**, who framed it as a *dilemma* rather than a choice: any method
flexible enough to fit an arbitrary function is flexible enough to fit the noise,
and the resolution has to come from outside the data — as a restriction, a
penalty, or a tuning parameter chosen by resampling.

**Beyer, Goldstein, Ramakrishnan & Shaft (1999)** is the reference for the failure
in high dimensions, showing that under broad conditions the distance to the
nearest neighbour approaches the distance to the farthest as dimension grows — so
the query on which the whole method depends becomes meaningless, not merely
imprecise.

Textbook treatments: **James, Witten, Hastie & Tibshirani (2021)**,
*An Introduction to Statistical Learning*, 2nd ed., ch. 2 and 4;
**Hastie, Tibshirani & Friedman (2009)**, *The Elements of Statistical Learning*,
2nd ed., ch. 13; **Provost & Fawcett (2013)**, *Data Science for Business*,
ch. 7–8. **Breiman (2001)**, "Statistical modeling: the two cultures",
*Statistical Science* 16(3), 199–231, is the essay on why the choice between a
rigid model and a flexible one is a choice about what you want from it.
