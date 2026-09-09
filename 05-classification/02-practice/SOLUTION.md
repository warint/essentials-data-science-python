# Session 5 · The Bestiary — answer key

*Read this after you have tried. Every number is asserted by
`scripts/verify_puzzles.py`.*

## The generator

Four species, each a 2-D Gaussian in (wing span, tail length) with its own mean
**and its own covariance** — so the Bayes-optimal boundary is genuinely curved,
and no straight line can reach it.

| species | prior | mean | share of records |
|---|---|---|---|
| `glimmerwing` | 0.34 | (6.5, 3.0) | 527 |
| `thornback` | 0.34 | (10.5, 5.5) | 545 |
| `dusk_moth` | 0.30 | (8.0, 8.5) | 494 |
| `ember_drake` | **0.02** | (13.5, 9.5) | **34 (2.12%)** |

Plus **7% label noise**, applied only among the three common species — an ember
drake is unmistakable once seen, so its labels are clean and the 2% class stays a
genuine 2%.

`mass_g`, `antenna_mm` and `scale_count` are pure noise.

## Stage 1 · The ceiling

| | |
|---|---|
| **Bayes accuracy** | **0.8625** |
| Bayes balanced accuracy | 0.8876 |

Two distinct reasons it is not 1.0, and both need naming:

1. **The classes overlap.** The Gaussians have common support, so some points are
   genuinely more likely to belong to a species other than the one that produced
   them. No information could resolve this.
2. **7% of the labels are wrong.** Even a perfect rule disagrees with the
   recorded label 7% of the time among the common species.

The two averages differ because accuracy weights classes by size and balanced
accuracy weights them equally — so the 2% species contributes 2% of one number
and 25% of the other.

> **A test accuracy of 0.98 would be a bug report, not an achievement.** It would
> mean test rows leaked into training, or a feature encodes the answer.

## Stage 2 · The bias–variance curve

| $k$ | train | test |
|---|---|---|
| 1 | **0.999** | **0.762** |
| 3 | 0.868 | 0.817 |
| 5 | 0.866 | 0.850 |
| 9 | 0.860 | 0.871 |
| 25 | 0.860 | 0.869 |
| **45** | 0.855 | **0.873** |
| 125 | 0.838 | 0.854 |
| 400 | 0.829 | 0.850 |

The two curves are **not the same shape**. Training accuracy falls monotonically
as $k$ rises — more smoothing always fits the training data worse. Test accuracy
is a hill: it rises to a peak around $k = 9$–45 and then declines.

The peak (0.873) sits just above the 0.8625 floor. That is sampling noise on a
480-row test set (one standard error is about 0.015), not a violation — the floor
is a population quantity and a test estimate wobbles around it.

Cross-validation on the training data picks **$k = 21$** (CV accuracy 0.858, test
0.8688). It did not pick 45, and that is correct behaviour: the difference is
noise, and picking the $k$ that maximised the *test* score would have been
cheating for nothing.

## Stage 3 · The boundary

- **$k = 1$** — the regions shatter into islands, many of them a single
  mislabelled creature. The classifier has fitted the **7% label noise**, which
  is why training accuracy is 0.999 and test accuracy is the worst on the table.
  This is variance.
- **$k \approx 45$** — the coloured regions track the true black boundary
  closely. Close to the truth.
- **$k = 400$** — smooth, in the wrong place, and the rare species has vanished
  from the map entirely. This is bias.

## Stage 4 · The rare species — the number that matters

Confusion matrix at $k = 45$ (rows = truth):

| | → dusk_moth | → ember_drake | → glimmerwing | → thornback |
|---|---|---|---|---|
| **dusk_moth** | 125 | 0 | 6 | 17 |
| **ember_drake** | 0 | **8** | 0 | 2 |
| **glimmerwing** | 3 | 0 | 141 | 14 |
| **thornback** | 15 | 0 | 4 | 145 |

Accuracy **0.873**.

| class | precision | recall | support |
|---|---|---|---|
| dusk_moth | 0.874 | 0.845 | 148 |
| **ember_drake** | **1.000** | **0.800** | **10** |
| glimmerwing | 0.934 | 0.892 | 158 |
| thornback | 0.815 | 0.884 | 164 |

**The key measurement.** Replace every `ember_drake` prediction with the most
common class:

| | accuracy |
|---|---|
| the model as fitted | 0.8729 |
| the same model, blind to ember drakes | 0.8562 |
| **cost of total blindness** | **0.0167** |

> **Complete blindness to the rare species costs 1.7 percentage points.** Anything
> optimising accuracy has almost no incentive to find it. If the rare species is
> why you built the classifier, accuracy is measuring the wrong thing — and the
> confusion matrix is what shows you.

Note also that the test set contains only **10** drakes, so recall of 0.800 means
"8 out of 10". Its standard error is enormous. You cannot evaluate a rare class on
a small test set, and saying so is part of the answer.

**Precision or recall?** Both are defensible with a scenario:

- **Recall** — cataloguing a rare species for conservation. A missed drake goes
  unprotected; a false alarm costs a field visit.
- **Precision** — the sighting triggers an expensive survey. A false alarm burns
  budget that the real drakes needed.

## Stage 5 · Three useless columns

| $k$ | 2 real features | + 3 noise columns | cost |
|---|---|---|---|
| 5 | 0.850 | 0.838 | −0.012 |
| 15 | 0.867 | 0.867 | 0.000 |
| **45** | **0.873** | **0.846** | **−0.027** |
| 125 | 0.854 | 0.846 | −0.008 |

Nearly three percentage points at the optimal $k$, from columns containing **no
information whatsoever**.

The reason is specifically about distance. $k$-NN asks "who is nearby?", and in
five dimensions two creatures of the same species can be pushed apart by nothing
but a difference in mass. **Distance in an irrelevant dimension is still
distance**, so the neighbourhood fills up with the wrong neighbours. This is the
curse of dimensionality in miniature.

Permutation importance finds them:

| feature | importance |
|---|---|
| `tail_length_cm` | 0.313 |
| `wing_span_cm` | 0.263 |
| `mass_g` | 0.006 |
| `antenna_mm` | 0.004 |
| `scale_count` | 0.001 |

A penalised regression would have shrunk the noise coefficients towards zero and
lost almost nothing. **$k$-NN has no coefficients, so it cannot ignore a feature
you hand it** — which is why feature selection matters more for distance methods
than for penalised linear ones.

## The optional extras

| model | train | test |
|---|---|---|
| logistic regression (linear) | 0.854 | 0.867 |
| **QDA (the correct family)** | 0.853 | **0.875** |
| tree, depth 3 | 0.825 | 0.844 |
| tree, unlimited | **0.999** | **0.765** |
| $k$-NN, $k$ = 45 | 0.855 | 0.873 |

- **QDA is the right family** — the generator really is Gaussians with unequal
  covariances — and it lands at the floor by knowing the answer's shape.
- **$k$-NN gets there without knowing anything.** That is the case for
  non-parametric methods.
- **The linear boundary loses only 0.006.** Worth a sentence: curvature only
  costs you where the data actually is, and here it barely is. "The true boundary
  is curved" does not automatically mean "a linear model will fail".
- **The unlimited tree is $k = 1$ in different clothing**: 0.999 train, 0.765
  test. Two models can score identically for opposite reasons, and reporting only
  the score hides which.

## Common wrong answers

| Reported | What happened |
|---|---|
| "87% accurate — good model" | never computed the 86.25% ceiling |
| $k$ chosen on the test set | the test score now estimates nothing |
| scaler fitted before the split | test mean and variance leaked into training |
| "the model handles all four species" | never replaced the rare-class predictions |
| "recall 0.80 on ember_drake" as a firm figure | that is 8 out of 10; the interval is huge |
| "the noise columns did not matter" | did not test at the optimal $k$ |
