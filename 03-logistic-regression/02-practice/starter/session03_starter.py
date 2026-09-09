"""Session 3 practice — The Mushroom Forager.

    python session03_starter.py

Deliverable: a decision rule you would eat from. A threshold, the two costs that
justify it, and the number of poisonous mushrooms it lets through.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import essentials as ess  # noqa: E402

d = ess.load("mushrooms")


# ---------------------------------------------------------------- stage 1
base = 1 - d.poisonous.mean()
print(f"stage 1 · {len(d)} mushrooms, {d.poisonous.mean():.1%} poisonous")
print(f"          'always edible' scores {base:.1%} accuracy "
      f"and eats all {int(d.poisonous.sum())} poisonous ones")
print("          ^ every accuracy you report today goes next to that number\n")


# ---------------------------------------------------------------- stage 2
# TODO: fit the logistic regression. Wrap categoricals in C(...), e.g.
#       C(spore_print, Treatment('brown')) to set the reference level.
# TODO: report np.exp(fit.params) — odds ratios, sorted.
# TODO: one feature dominates. Which? Look it up afterwards; it has a name.
# TODO: take its odds ratio and apply it to a mushroom at p = 0.01 and one at
#       p = 0.90. The probability changes differ. Explain why.


# ---------------------------------------------------------------- stage 3
# TODO: p_hat = fit.predict(d)          # probabilities, NOT classes
# TODO: confusion matrix at t = 0.5; label the four cells in words.
# TODO: accuracy, with the baseline beside it.
# TODO: how many poisonous mushrooms did it call edible? One meal in how many?
# TODO: AUC — and one sentence on why it cannot choose a threshold for you.


# ---------------------------------------------------------------- stage 4
# The stage that carries the marks.
C_FN, C_FP = 50, 1          # <- CHANGE THESE, and justify whatever you choose


def outcomes(p_hat, t, c_fn=C_FN, c_fp=C_FP):
    """Accuracy, misses, basket discarded and expected loss at threshold t."""
    tn, fp, fn, tp = confusion_matrix(d.poisonous, (p_hat > t).astype(int)).ravel()
    return {"t": t, "accuracy": (tp + tn) / len(d), "missed": fn,
            "discarded": (tp + fp) / len(d), "loss": c_fn * fn + c_fp * fp}


# TODO: build a grid of thresholds, compute outcomes() at each, and plot loss.
# TODO: where is the minimum? Compare with c_fp / (c_fp + c_fn).
# TODO: report accuracy / missed / discarded at your threshold AND at 0.5.
# TODO: say which error you chose to make more of.


# ---------------------------------------------------------------- stage 5
# TODO: from statsmodels.miscmodels.ordinal_model import OrderedModel
#       Build a numeric design matrix (pd.get_dummies with dtype=float) and fit
#       OrderedModel(sub.toxicity.cat.codes, X, distr="logit").
# TODO: what does proportional odds claim about your dominant feature?
# TODO: check it — separate binary logits at each cut-point, compare coefficients.


print("Work down the TODOs. Answer key in ../SOLUTION.md — after you try.")
