# 5 · Classification

> The only chapter where you can see the right answer. This data was generated
> from known distributions, so the best achievable accuracy is computable and the
> true decision boundary can be drawn underneath your own.

## 5.1 The best possible classifier

If you knew the true class probabilities at every point, the best rule would be
obvious: predict the most likely class.

$$ \hat y(x) = \arg\max_j \Pr(Y = j \mid X = x) $$

That is the **Bayes classifier**. No method beats it, and its error rate — the
**Bayes rate** — is usually not zero, because classes overlap and labels contain
mistakes.

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

The whole algorithm: to classify a new point, find the $k$ closest training
points and take a vote.

$$ \hat{\Pr}(Y = j \mid x) = \frac{1}{k}\sum_{i \in N_k(x)} \mathbb{1}(y_i = j) $$

- **No fitting.** No coefficients, no likelihood, no optimisation.
- **No functional form.** It assumes nothing about linearity or normality.
- **Everything is stored.** Prediction costs a search over the training set.

The training data *is* the model.

### Standardise, or the units decide

$k$-NN is a distance method, and in this data `mass_g` has a standard deviation
nearly forty times that of tail length. In Euclidean distance it would dominate
every other feature — and it is pure noise. **A distance method on
unstandardised features is measuring whichever variable happens to have the
biggest numbers.**

Fit the scaler on the **training** data only. Fitting it on everything leaks the
test set's mean and variance into the model.

## 5.3 Bias and variance, made visible

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

## 5.4 The confusion matrix, and the 2% species

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

## 5.5 What three useless columns cost

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

## 5.6 Four estimators, and why the scores mislead

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

## 5.7 What to report

- The **confusion matrix**, always. Accuracy alone is not a result.
- Accuracy against a **baseline** — the majority class, and a computed floor
  where one exists.
- **Per-class precision and recall**, and which of the two your problem needs.
- How the hyperparameter was chosen, and on **which** data.
- What the model cannot do.

## 5.8 Summary

1. Know the ceiling if you can compute one.
2. $k$-NN has no fit, no form and no coefficients.
3. $k$ is the bias–variance dial: $k=1$ memorises, $k=n$ is the majority class.
4. Standardise, and fit the scaler on training data only.
5. Accuracy hides rare classes; ignoring a 2% class costs 1.7 points.
6. Precision or recall — the problem decides.
7. Irrelevant features punish distance methods, and $k$-NN cannot ignore them.

## Further reading

- James, Witten, Hastie & Tibshirani (2021), *An Introduction to Statistical
  Learning*, 2nd ed., ch. 2 and 4.
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*,
  2nd ed., ch. 13.
- Provost & Fawcett (2013), *Data Science for Business*, ch. 7–8.
- Breiman (2001), "Statistical modeling: the two cultures", *Statistical Science*
  16(3), 199–231.
