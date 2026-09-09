"""Session 5 practice — The Bestiary.

    python session05_starter.py

The only puzzle in this course where you can see the right answer:
`ess.bestiary_posterior()` returns the true class probabilities at any point, so
you can compute the best achievable accuracy in advance and draw the true
decision boundary underneath your own.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import essentials as ess  # noqa: E402

d = ess.load("bestiary")
SPECIES = ess.bestiary_species()
y = d.species.astype(str).to_numpy()

REAL = ["wing_span_cm", "tail_length_cm"]
NOISE = ["mass_g", "antenna_mm", "scale_count"]

print(d.species.value_counts().to_string())
print(f"rarest: {(d.species == 'ember_drake').mean():.2%}\n")


# ---------------------------------------------------------------- stage 1
post = ess.bestiary_posterior(d.wing_span_cm, d.tail_length_cm)
bayes = np.array(SPECIES)[post.argmax(1)]
FLOOR = accuracy_score(y, bayes)
print(f"stage 1 · Bayes accuracy          {FLOOR:.4f}   <- the ceiling")
print(f"          Bayes balanced accuracy {balanced_accuracy_score(y, bayes):.4f}")
# TODO: why do those two differ?
# TODO: why is the ceiling not 1.0? Name TWO reasons from the generator.
# TODO: if your test accuracy came out at 0.98, what would you conclude?


# ---------------------------------------------------------------- stage 2
# stratify=y matters: without it a random split of a 2% class can leave almost
# none of it in the test set, and your estimate of the class you care about
# would rest on four creatures.
X = d[REAL].to_numpy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1,
                                      stratify=y)
# Fit the scaler on TRAINING data only. Fitting it on everything leaks the test
# set's mean and variance into the model.
sc = StandardScaler().fit(Xtr)
A, B = sc.transform(Xtr), sc.transform(Xte)

print("\nstage 2 · k, train, test")
for k in (1, 3, 5, 9, 15, 25, 45, 75, 125, 250, 400):
    m = KNeighborsClassifier(k).fit(A, ytr)
    print(f"          {k:4d}   {m.score(A, ytr):.3f}   {m.score(B, yte):.3f}")
# TODO: plot both curves against k, with FLOOR as a horizontal line.
# TODO: describe each curve in one sentence. They are not the same shape.


# ---------------------------------------------------------------- stage 3
# TODO: build a mesh over the two real features:
#         GX, GT = np.meshgrid(np.linspace(...), np.linspace(...))
#         truth = ess.bestiary_posterior(GX, GT).argmax(1).reshape(GX.shape)
# TODO: for k = 1, your best k, and k = 400: contourf your predictions and
#       contour `truth` on top. Three panels.
# TODO: what exactly is k=1 fitting? What has k=400 lost?


# ---------------------------------------------------------------- stage 4
# TODO: choose k by cross-validation ON THE TRAINING DATA:
#         GridSearchCV(KNeighborsClassifier(),
#                      {"n_neighbors": list(range(1, 151, 2))},
#                      cv=KFold(10, shuffle=True, random_state=0)).fit(A, ytr)
# TODO: confusion matrix and classification_report at that k.
# TODO: THE KEY MEASUREMENT — replace every ember_drake prediction with your
#       most common class, and recompute accuracy. How much did total blindness
#       to the rare species cost?
# TODO: precision or recall on ember_drake? Give a scenario for each.


# ---------------------------------------------------------------- stage 5
# TODO: refit with REAL + NOISE. Compare accuracy at several k.
# TODO: explain the drop in terms of DISTANCE.
# TODO: from sklearn.inspection import permutation_importance — does it find
#       the noise columns?
# TODO: why would a penalised regression have coped better?


print("\nWork down the TODOs. Answer key in ../SOLUTION.md — after you try.")
