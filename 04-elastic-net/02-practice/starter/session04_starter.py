"""Session 4 practice — The Secret Recipe.

    python session04_starter.py

Deliverable: a named shortlist of ingredients, with signs and rough sizes, and
an honest split between the ones you are confident about and the ones you are
not. Two of the planted ingredients are near-identical to each other; one is
below the detection floor at n = 60. Both facts are part of the answer.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import essentials as ess  # noqa: E402

r = ess.load("secret_recipe")
X = r.drop(columns=["bake_id", "taste_score"])
y = r.taste_score.to_numpy()

# Standardise BEFORE penalising. A penalty sums coefficients, and a
# coefficient's size depends on its variable's units, so an unstandardised
# penalty punishes whichever ingredient happens to be measured in small numbers.
Xs = StandardScaler().fit_transform(X.values)
names = X.columns
kf = KFold(10, shuffle=True, random_state=0)


def cv_mse(model):
    """Cross-validated mean squared error, on one fixed set of folds."""
    return -cross_val_score(model, Xs, y, cv=kf,
                            scoring="neg_mean_squared_error").mean()


print(f"{r.shape[0]} bakes, {len(names)} candidate ingredients "
      f"-> p/n = {len(names) / r.shape[0]:.1f}")


# ---------------------------------------------------------------- stage 1
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.35, random_state=7)
ols = LinearRegression().fit(Xtr, ytr)
print(f"\nstage 1 · OLS  train MSE {mean_squared_error(ytr, ols.predict(Xtr)):.8f}"
      f"   test MSE {mean_squared_error(yte, ols.predict(Xte)):.2f}"
      f"   var(y) {y.var():.2f}")
# TODO: which is bigger, test MSE or var(y)? What does that mean?
# TODO: explain the training MSE in one sentence.


# ---------------------------------------------------------------- stage 2
# TODO: ridge = RidgeCV(alphas=np.logspace(-2, 4, 25), cv=kf).fit(Xs, y)
# TODO: how many coefficients are exactly zero? How many exceed 0.1?
# TODO: cv_mse(Ridge(alpha=ridge.alpha_))
# TODO: answer the client's question — which ingredients matter? Then say why
#       ridge cannot answer it.


# ---------------------------------------------------------------- stage 3
# TODO: lasso = LassoCV(cv=kf, random_state=0, max_iter=200_000).fit(Xs, y)
# TODO: print the selected ingredients sorted by |coefficient|.
# TODO: cross-validated MSE, against ridge.
# TODO: plot the path — from sklearn.linear_model import lasso_path
# TODO: is your selected set the right SIZE? What is CV optimising?


# ---------------------------------------------------------------- stage 4
c = X.corr().abs().to_numpy(copy=True)
np.fill_diagonal(c, 0)                       # ignore each column against itself
i, j = np.unravel_index(np.argmax(c), c.shape)
print(f"\nstage 4 · most correlated pair: {names[i]} & {names[j]} "
      f"(r = {X.iloc[:, i].corr(X.iloc[:, j]):.4f})")
# TODO: what did the lasso give each of them?
# TODO: fit ElasticNetCV(l1_ratio=0.95, cv=kf, ...). What does it give each?
# TODO: re-fit with five different KFold seeds. Print both coefficients AND
#       their sum. WHICH OF THE THREE NUMBERS IS STABLE? That determines what
#       you are entitled to claim.


# ---------------------------------------------------------------- stage 5
# TODO: which of your selections would you report, and which would you flag?
# TODO: in which direction are penalised coefficients biased, and why?
# TODO: can you put a p-value on any of them? Why not?
# TODO: is there an ingredient you suspect is real and cannot demonstrate?


print("\nWork down the TODOs. Answer key in ../SOLUTION.md — after you try.")
