#!/usr/bin/env python3
"""Write a one-page-ish Python cheatsheet, as a PDF, for every session.

    python3 scripts/build_cheatsheets.py             # all six sessions
    python3 scripts/build_cheatsheets.py 03          # one session
    python3 scripts/build_cheatsheets.py --no-render # write the .qmd, skip LaTeX

Why this exists
---------------
The decks execute their own code, so every name a student needs is somewhere in
eighteen reveal.js files. That is the wrong shape for reference material. A
student stuck on `get_influence` at 21:00 does not want to arrow through a deck;
they want a page beside the keyboard.

So each session gets `PYTHON-CHEATSHEET.pdf`: the environment, the four things
you always do, how to install what you are missing, and then every name taught
up to *and including* that session, one runnable line each. Session 4's sheet
covers sessions 1 to 4 — cumulative, because a student in session 4 has to
remember `get_influence` from session 1 as much as `LassoCV` from today, and a
sheet carrying only this week would send them hunting through three other PDFs.

What is in the registry
-----------------------
TOOLKIT below maps a name to `(session, group, snippet)`. The snippets are
written the way they appear in the session that introduced them, against the
course data, so a line copied off the sheet is a line that has run — not a
signature copied out of the documentation. A name appears once, in the session
that first teaches it.

The session list itself is imported from `build_index.py` rather than repeated,
so the sheets and the landing page can never disagree about what session 4 is.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_index import SESSIONS  # noqa: E402

FILENAME = "PYTHON-CHEATSHEET"

# The order the groups print in. Teaching order, not alphabetical: you load
# before you describe, describe before you fit, and read a fit before you
# diagnose it.
GROUPS = ["Load and shape", "Describe", "Fit", "Read the fit", "Diagnose",
          "Choose and validate", "Plot"]


# ---------------------------------------------------------------------------
# The registry: name -> (session, group, one runnable line)
#
# Registry order is preserved inside each group, so `smf.ols` comes before
# `.fit()` and `.get_influence()` before the three things you pull off it.

TOOLKIT = OrderedDict([

    # ---- Session 01 · regression, on the coasters --------------------------
    ("ess.load", ("01", "Load and shape",
        'import essentials as ess\n'
        'd = ess.load("coasters")          # generated locally, no network')),
    ("ess.catalog", ("01", "Load and shape",
        "ess.catalog()                     # every dataset, and its puzzle")),
    (".shape", ("01", "Load and shape",
        "n_rows, n_cols = d.shape")),
    (".head", ("01", "Load and shape",
        'd[["coaster", "drop_m", "speed_kmh"]].head()')),
    (".copy", ("01", "Load and shape",
        'g = d[d.propulsion == "gravity"].copy()\n'
        "# .copy() before you modify a subset, or pandas warns and you guess")),
    ("np.log / np.exp", ("01", "Load and shape",
        'g["log_speed"] = np.log(g.speed_kmh)      # a power law becomes a line\n'
        'c = np.exp(power.params["Intercept"])     # and back again')),
    ("pd.cut", ("01", "Load and shape",
        "bands = pd.cut(g.drop_m, [0, 25, 45, 65, 85, 105, 200])")),
    (".assign / .groupby / .agg", ("01", "Load and shape",
        "(g.assign(resid=lin.resid)\n"
        '  .groupby(bands, observed=True)["resid"].agg(["mean", "count"]))\n'
        "# the sign of the mean residual, band by band, is the whole diagnosis")),

    (".describe", ("01", "Describe",
        "d.describe()                      # count, mean, sd, quartiles")),
    (".value_counts", ("01", "Describe",
        "d.propulsion.value_counts()       # how many of each")),

    ("smf.ols", ("01", "Fit",
        "import statsmodels.formula.api as smf\n"
        'lin = smf.ols("speed_kmh ~ drop_m", data=g).fit()\n'
        '# read the ~ as "explained by"; the intercept is added for you')),
    (".fit()", ("01", "Fit",
        "fit = model.fit()                 # .fit() does the arithmetic")),

    (".params / .bse", ("01", "Read the fit",
        'lin.params["drop_m"], lin.bse["drop_m"]   # estimate, and its wobble')),
    (".tvalues / .pvalues", ("01", "Read the fit",
        'lin.tvalues["drop_m"], lin.pvalues["drop_m"]')),
    (".conf_int", ("01", "Read the fit",
        'lo, hi = lin.conf_int().loc["drop_m"]     # 95% by default')),
    (".rsquared / .rsquared_adj", ("01", "Read the fit",
        "lin.rsquared, lin.rsquared_adj    # the second penalises extra columns")),
    (".resid / .fittedvalues", ("01", "Read the fit",
        "lin.resid, lin.fittedvalues       # what is left, and what was predicted")),
    (".summary()", ("01", "Read the fit",
        "print(lin.summary().tables[1])    # the coefficient table on its own")),

    (".get_influence()", ("01", "Diagnose",
        "inf = lin.get_influence()         # the whole diagnostic bundle")),
    (".hat_matrix_diag", ("01", "Diagnose",
        "h = inf.hat_matrix_diag           # leverage; it sums to p, always\n"
        "(h > 2 * 2 / len(d)).sum()        # the 2p/n screen")),
    (".cooks_distance", ("01", "Diagnose",
        "cooks = inf.cooks_distance[0]     # leverage x residual; screen at 4/n")),

    ("plt.subplots", ("01", "Plot",
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(7.2, 3.2))")),
    ("ax.scatter", ("01", "Plot",
        "ax.scatter(g.drop_m, g.speed_kmh, s=20, alpha=0.8)")),
    ("ax.set_xlabel / set_ylabel", ("01", "Plot",
        'ax.set_xlabel("Drop (m)"); ax.set_ylabel("Top speed (km/h)")\n'
        "# say the units, every time")),
    ("ax.axhline / ax.axvline", ("01", "Plot",
        'ax.axhline(0, lw=1)               # the line you are judging against\n'
        'ax.axhline(4 / len(d), ls="--", lw=1)     # the Cook\'s screen')),
    ("ax.vlines", ("01", "Plot",
        "ax.vlines(range(len(cooks)), 0, cooks, lw=1.4)   # one spike per row")),
    ("plt.show", ("01", "Plot",
        "plt.show()                        # last line of every figure")),

    # ---- Session 02 · non-linearity, interactions, fixed effects -----------
    (".nunique", ("02", "Load and shape",
        "d.baker.nunique()                 # eight bakers")),
    (".to_numpy / .reshape / .ravel", ("02", "Load and shape",
        "Z = full.predict(grid).to_numpy().reshape(HH.shape)\n"
        "HH.ravel()                        # a grid, flattened into a column")),

    ("I(x**2) in a formula", ("02", "Fit",
        'quad = smf.ols("rise_mm ~ hydration_pct + I(hydration_pct**2)",\n'
        "               data=d).fit()\n"
        "# I() means: treat what is inside as arithmetic, not as formula syntax")),
    ("C(group)", ("02", "Fit",
        'fe = smf.ols("rise_mm ~ hydration_pct + oven_c + C(baker)",\n'
        "             data=d).fit()\n"
        "# one dummy per baker, one held back as the reference")),
    ("a:b interaction", ("02", "Fit",
        '"rise_mm ~ hydration_pct + oven_c + hydration_pct:oven_c"\n'
        "# the slope on water now depends on the oven, and vice versa")),

    (".predict", ("02", "Read the fit",
        'row = pd.DataFrame([{"hydration_pct": 78, "oven_c": 228, "baker": "D"}])\n'
        "fit.predict(row)                  # a fitted value at a point you choose")),
    (".get_prediction / .summary_frame", ("02", "Read the fit",
        "pred = fit.get_prediction(grid).summary_frame()\n"
        'pred["mean"], pred["mean_ci_lower"], pred["mean_ci_upper"]')),
    (".aic / .bic", ("02", "Read the fit",
        "fit.aic, fit.bic                  # lower is better, and only relatively")),

    (".compare_f_test", ("02", "Choose and validate",
        "f_stat, p_val, _ = big.compare_f_test(small)\n"
        "# call it on the bigger model; small must be big with terms removed")),
    ("minimize", ("02", "Choose and validate",
        "from scipy.optimize import minimize\n"
        "r = minimize(neg_predicted, [75, 225], bounds=[(55, 95), (190, 260)])\n"
        "# find the peak of a fitted surface numerically, inside the design")),
    ("ess.sourdough_optimum", ("02", "Choose and validate",
        "H_TRUE, T_TRUE = ess.sourdough_optimum()  # the planted answer, to score against")),

    ("np.linspace / np.meshgrid", ("02", "Plot",
        "hs = np.linspace(58, 94, 60); ts = np.linspace(195, 258, 60)\n"
        "HH, TT = np.meshgrid(hs, ts)")),
    ("ax.contour / ax.contourf / ax.clabel", ("02", "Plot",
        "cs = ax.contour(HH, TT, Z, levels=12, linewidths=0.7)\n"
        'ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")\n'
        'ax.contourf(HH, TT, Z, levels=24, cmap="YlOrBr", alpha=0.45)')),
    ("ax.fill_between", ("02", "Plot",
        'ax.fill_between(hs, pred["mean_ci_lower"], pred["mean_ci_upper"],\n'
        "                alpha=0.18)       # the interval, as a band")),
    ("ax.annotate", ("02", "Plot",
        'ax.annotate("planted optimum", (H_TRUE, T_TRUE), (H_TRUE - 17, T_TRUE + 9))')),

    # ---- Session 03 · logistic and ordinal regression ----------------------
    (".astype", ("03", "Load and shape",
        "(p_hat > t).astype(int)           # a probability becomes a call")),
    (".dropna", ("03", "Load and shape",
        'sub = d.dropna(subset=["toxicity"]).copy()\n'
        "# subset= matters: without it a row missing ANY column goes")),
    ("pd.get_dummies", ("03", "Load and shape",
        'X = pd.get_dummies(sub[["odor_index", "spore_print"]],\n'
        '                   columns=["spore_print"], drop_first=True, dtype=float)')),
    (".cat.codes", ("03", "Load and shape",
        "sub.toxicity.cat.codes            # an ordered category as 0, 1, 2, 3")),
    (".reindex", ("03", "Load and shape",
        'counts.reindex(["edible", "upset stomach", "hospital", "fatal"])\n'
        "# print an ordered category in its own order, not alphabetically")),

    ("smf.logit", ("03", "Fit",
        'fit = smf.logit("poisonous ~ odor_index + volva_present",\n'
        "                data=d).fit(disp=0)")),
    ("C(x, Treatment(ref))", ("03", "Fit",
        '"... + C(spore_print, Treatment(\'brown\'))"\n'
        "# choose the reference level yourself, rather than taking the first")),
    ("OrderedModel", ("03", "Fit",
        "from statsmodels.miscmodels.ordinal_model import OrderedModel\n"
        'ord_fit = OrderedModel(sub.toxicity.cat.codes, X,\n'
        '                       distr="logit").fit(method="bfgs", disp=0)')),

    ("np.exp on coefficients", ("03", "Read the fit",
        "np.exp(fit.params)                # a log-odds becomes an odds ratio\n"
        "# an odds ratio is not a probability: what it does to p depends on p")),
    (".predict (probabilities)", ("03", "Read the fit",
        "p_hat = fit.predict(d)            # probabilities, not classes")),
    (".summary2", ("03", "Read the fit",
        'fit.summary2().tables[1].loc[rows, ["Coef.", "Std.Err.", "z", "P>|z|"]]')),
    (".llf / .prsquared", ("03", "Read the fit",
        "lr = 2 * (fit.llf - reduced.llf)  # the likelihood-ratio statistic\n"
        "fit.prsquared                     # McFadden's pseudo R2")),
    ("stats.chi2.sf", ("03", "Read the fit",
        "from scipy import stats\n"
        "stats.chi2.sf(lr, df)             # its p-value")),

    ("confusion_matrix", ("03", "Choose and validate",
        "from sklearn.metrics import confusion_matrix\n"
        "tn, fp, fn, tp = confusion_matrix(d.poisonous, calls).ravel()\n"
        "# fn is the one that matters here: a poisonous mushroom called edible")),
    ("roc_auc_score / roc_curve", ("03", "Choose and validate",
        "from sklearn.metrics import roc_auc_score, roc_curve\n"
        "roc_auc_score(d.poisonous, p_hat)\n"
        "fpr, tpr, thr = roc_curve(d.poisonous, p_hat)")),
    ("a threshold from costs", ("03", "Choose and validate",
        "C_FN, C_FP = 50, 1                # a hospital visit against a lost meal\n"
        "t_star = C_FP / (C_FP + C_FN)     # = 1/51, not 0.5\n"
        "loss = C_FN * fn + C_FP * fp      # what you are actually minimising")),

    # ---- Session 04 · ridge, lasso, elastic net ----------------------------
    (".drop(columns=)", ("04", "Load and shape",
        'X = r.drop(columns=["bake_id", "taste_score"])\n'
        "y = r.taste_score.to_numpy()")),
    ("StandardScaler", ("04", "Load and shape",
        "from sklearn.preprocessing import StandardScaler\n"
        "Xs = StandardScaler().fit_transform(X.values)\n"
        "# a penalty on raw columns penalises whatever is measured in small units")),
    ("train_test_split", ("04", "Load and shape",
        "from sklearn.model_selection import train_test_split\n"
        "Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.35,\n"
        "                                      random_state=7)")),

    ("LinearRegression", ("04", "Fit",
        "from sklearn.linear_model import LinearRegression\n"
        "ols = LinearRegression().fit(Xtr, ytr)   # p > n: it fits perfectly, and lies")),
    ("Ridge / Lasso / ElasticNet", ("04", "Fit",
        "from sklearn.linear_model import Ridge, Lasso, ElasticNet\n"
        "Ridge(alpha=lam)                  # shrinks everything, zeroes nothing\n"
        "Lasso(alpha=lam)                  # shrinks, and zeroes\n"
        "ElasticNet(alpha=lam, l1_ratio=0.95)     # both, and keeps pairs together")),
    ("RidgeCV / LassoCV / ElasticNetCV", ("04", "Fit",
        "from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV\n"
        "lasso = LassoCV(cv=kf, random_state=0, max_iter=200_000).fit(Xs, y)\n"
        "lasso.alpha_                      # the lambda cross-validation chose")),
    ("lasso_path", ("04", "Fit",
        "from sklearn.linear_model import lasso_path\n"
        "lam, coefs, _ = lasso_path(Xs, y, n_alphas=120, eps=1e-3)")),

    (".coef_", ("04", "Read the fit",
        "coef = pd.Series(model.coef_, index=X.columns)\n"
        "sel = coef[coef.abs() > 1e-6]     # what survived")),

    ("KFold / cross_val_score", ("04", "Choose and validate",
        "from sklearn.model_selection import KFold, cross_val_score\n"
        "kf = KFold(10, shuffle=True, random_state=0)\n"
        '-cross_val_score(model, Xs, y, cv=kf,\n'
        '                 scoring="neg_mean_squared_error").mean()')),
    ("mean_squared_error", ("04", "Choose and validate",
        "from sklearn.metrics import mean_squared_error\n"
        "mean_squared_error(yte, ols.predict(Xte))   # against y.var(), not against 0")),

    (".corr", ("04", "Describe",
        "X.baking_soda.corr(X.baking_powder)      # 0.96: the lasso will pick one")),

    ("np.logspace / np.log10", ("04", "Plot",
        "alphas = np.logspace(-2, 4, 25)   # a penalty grid is log-spaced\n"
        "ax.plot(np.log10(lam), coefs[j])  # so plot it on a log axis")),

    # ---- Session 05 · classification ---------------------------------------
    ("ess.bestiary_posterior", ("05", "Load and shape",
        "post = ess.bestiary_posterior(d.wing_span_cm, d.tail_length_cm)\n"
        "SPECIES = ess.bestiary_species()\n"
        "# the true class probabilities: the Bayes floor no model can beat")),
    ("StandardScaler, fit on train only", ("05", "Load and shape",
        "sc = StandardScaler().fit(Xtr)    # fit on TRAIN\n"
        "A, B = sc.transform(Xtr), sc.transform(Xte)   # transform both\n"
        "# fitting the scaler on everything leaks the test set into the model")),
    ("stratify=", ("05", "Load and shape",
        "train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)\n"
        "# without it, a 2% species can miss the test set entirely")),
    ("np.column_stack", ("05", "Load and shape",
        "mesh = np.column_stack([GX.ravel(), GT.ravel()])   # a grid to predict on")),
    ("pd.Categorical", ("05", "Load and shape",
        "pd.Categorical(pred, categories=SPECIES).codes     # labels -> integers")),

    ("KNeighborsClassifier", ("05", "Fit",
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "m = KNeighborsClassifier(45).fit(A, ytr)\n"
        "# k is the whole model: k=1 memorises, k=n predicts the majority")),
    ("other classifiers", ("05", "Fit",
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "LogisticRegression(max_iter=5000)        # a straight boundary\n"
        "QuadraticDiscriminantAnalysis()          # a curved one\n"
        "DecisionTreeClassifier(max_depth=3, random_state=0)")),

    (".score / .predict", ("05", "Read the fit",
        "m.score(A, ytr), m.score(B, yte)  # train and test, always both\n"
        "pred = m.predict(B)")),

    ("GridSearchCV", ("05", "Choose and validate",
        "from sklearn.model_selection import GridSearchCV\n"
        "grid = GridSearchCV(KNeighborsClassifier(),\n"
        '                    {"n_neighbors": list(range(1, 151, 2))},\n'
        "                    cv=KFold(10, shuffle=True, random_state=0)).fit(A, ytr)\n"
        "grid.best_params_, grid.best_score_")),
    ("accuracy vs balanced accuracy", ("05", "Choose and validate",
        "from sklearn.metrics import accuracy_score, balanced_accuracy_score\n"
        "accuracy_score(yte, pred)         # can be high while a class is invisible\n"
        "balanced_accuracy_score(yte, pred)       # averages recall over classes")),
    ("classification_report", ("05", "Choose and validate",
        "from sklearn.metrics import classification_report\n"
        "print(classification_report(yte, pred, labels=LABS, digits=3))\n"
        "# read the per-class recall column before you read accuracy")),
    ("permutation_importance", ("05", "Choose and validate",
        "from sklearn.inspection import permutation_importance\n"
        "imp = permutation_importance(m, Xte_s, yte, n_repeats=12, random_state=0)\n"
        "pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)")),

    # ---- Session 06 · modern difference-in-differences ---------------------
    ("ess.cat_cafe_truth", ("06", "Load and shape",
        "TRUTH = ess.cat_cafe_truth()      # the planted ATT, cohort by quarter")),
    (".merge", ("06", "Load and shape",
        'cells = cells.merge(TRUTH, on=["cohort", "quarter"], how="left")')),
    (".notna / .isna / .fillna", ("06", "Load and shape",
        "cells.true_att = cells.true_att.fillna(0.0)\n"
        "ok = cells[cells.est.notna()]     # a cell with no clean control is NaN")),

    ("two-way fixed effects", ("06", "Fit",
        'twfe = smf.ols("dwell_min ~ has_cat + C(cafe_id) + C(quarter)",\n'
        "               data=d).fit()\n"
        "# the trap: with staggered timing this is a weighted average that can\n"
        "# carry NEGATIVE weights, and return the wrong sign at p = 1e-5")),
    ("dd.BaconDecomposition", ("06", "Fit",
        "import diff_diff as dd\n"
        "bacon = dd.BaconDecomposition().fit(\n"
        '    d, outcome="dwell_min", unit="cafe_id", time="quarter",\n'
        '    first_treat="cat_arrival_quarter")\n'
        "# shows you exactly which comparisons TWFE made, and with what weight")),
    ("dd.CallawaySantAnna", ("06", "Fit",
        'cs = dd.CallawaySantAnna(control_group="not_yet_treated").fit(\n'
        '    d, outcome="dwell_min", unit="cafe_id", time="quarter",\n'
        '    first_treat="cat_arrival_quarter", aggregate="group")')),
    ("the rest of the toolkit", ("06", "Fit",
        "dd.SunAbraham(...)                # interaction-weighted event study\n"
        "dd.ImputationDiD(...)             # Borusyak-Jaravel-Spiess\n"
        "dd.TwoStageDiD(...)               # Gardner\n"
        "dd.HonestDiD(...)                 # Rambachan-Roth sensitivity")),

    (".to_dataframe()", ("06", "Read the fit",
        'bacon.to_dataframe()[["treated_group", "control_group",\n'
        '                      "comparison_type", "estimate", "weight"]]\n'
        "bacon.twfe_estimate, bacon.decomposition_error")),
    (".overall_att / .group_effects", ("06", "Read the fit",
        "cs.overall_att                    # one number, correctly weighted\n"
        'cs.group_effects[4]["effect"], cs.group_effects[4]["se"]\n'
        "cs.group_time_effects[(4, 6)]     # one cohort, one quarter")),

    ("np.average with weights", ("06", "Choose and validate",
        "np.average(s.estimate, weights=s.weight)   # a weighted average, by hand")),
    ("a placebo check", ("06", "Choose and validate",
        "pre = cells[(cells.event_time < 0) & cells.est.notna()]\n"
        "(pre.est.abs() > 1.96 * pre.se).sum()      # how many pre-periods fire")),
    ("np.polyfit", ("06", "Choose and validate",
        "slope = np.polyfit(pre.event_time, pre.est, 1)[0]\n"
        "# the pre-trend slope you would have to rule out")),

    ("ax.errorbar", ("06", "Plot",
        "ax.errorbar(ev.event_time, ev.est, yerr=1.96 * ev.se,\n"
        '            fmt="o", ms=4, capsize=2, lw=1.2)')),
])


# ---------------------------------------------------------------------------
# The parts that are the same on every sheet

PREAMBLE = r"""\usepackage{titling}
\usepackage{xcolor}
\definecolor{accent}{HTML}{E3120B}
\definecolor{ink}{HTML}{222222}
\definecolor{muted}{HTML}{6B6B67}

% A modest title block: the sheet is a reference, not a paper.
\pretitle{\begin{flushleft}\large\bfseries\color{ink}}
\posttitle{\end{flushleft}\vspace{-0.6em}%
  {\color{accent}\rule{3.2em}{2pt}}\vspace{0.4em}}
\preauthor{\begin{flushleft}\footnotesize\color{muted}}
\postauthor{\end{flushleft}}
\predate{\begin{flushleft}\footnotesize\color{muted}}
\postdate{\end{flushleft}}
\setlength{\droptitle}{-3.2em}

\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1.1em}{0.35em}
\titlespacing*{\subsection}{0pt}{0.8em}{0.25em}
\titleformat{\section}{\normalsize\bfseries\color{ink}}{}{0pt}{}
\titleformat{\subsection}{\small\bfseries\color{accent}}{}{0pt}{}
"""

GETTING_STARTED = """## Starting Python in VS Codium

Open the course folder once — **File > Open Folder**, then pick the repository. Everything below
assumes you are inside it, and in **the VS Codium terminal** (**View > Terminal**, or `Ctrl` +
`` ` ``) rather than the macOS Terminal or PowerShell.

Create the environment the first time only:

```bash
python3 -m venv .venv      # macOS
python3 -m venv .venv      # Linux
py -m venv .venv           # Windows (PowerShell)
```

Then activate it — **every time you open a new terminal**:

```bash
source .venv/bin/activate  # macOS
source .venv/bin/activate  # Linux
.venv\\Scripts\\activate     # Windows (PowerShell)
```

You will see `(.venv)` at the start of the prompt. If you do not, nothing below will find pandas.
Install the packages once, with the environment active, and check the course data is intact:

```bash
pip install -r requirements.txt
python scripts/verify_puzzles.py    # 39 lines of [  ok  ], in about twenty seconds
```

Finally tell the editor which Python to use: `Ctrl`/`Cmd` + `Shift` + `P`, type
**Python: Select Interpreter**, and choose the one inside `.venv`.

## A new file, and running it

**File > New File**, then save it with a `.py` ending — `sessionSESSIONNUM.py`, say. Write your
code, save it, and in the terminal:

```bash
python sessionSESSIONNUM.py     # macOS, Linux and Windows alike, inside .venv
```

Inside the environment the command is `python` on all three platforms. Outside it, macOS and Linux
need `python3` and Windows needs `py`, which is one more reason to activate first. For trying one
line at a time, type `python` on its own to get the `>>>` prompt, and `exit()` to leave.

## The four things you always do

```python
import numpy as np
import pandas as pd
import essentials as ess

# 1 - load a course dataset, or build a frame by hand
d = ess.load("DATASETNAME")                 # generated locally; no network needed
ess.catalog()                               # the other five, and their puzzles
df = pd.DataFrame({"country": ["FR", "DE"], "gdp_pc_eur": [37_000, 46_000]})

# 2 - look at what is in it
d.head()           # the first five rows
d.shape            # (rows, columns)
d.dtypes           # what type each column holds
d.info()           # types, and how many values are missing
d.describe()       # count, mean, sd and quartiles, per numeric column

# 3 - read a file in
df = pd.read_csv("data.csv")
df = pd.read_excel("data.xlsx")             # needs openpyxl - see below
df = pd.read_parquet("data/coasters.parquet")

# 4 - write a CSV out
d.to_csv("out.csv", index=False)            # index=False, or you get a stray column
```

`import essentials as ess` works from the repository root. From a subfolder — a starter script in
`SESSIONDIR/02-practice/starter/`, say — point Python back at the root first, which is what the
two lines at the top of every starter do:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import essentials as ess
```

## Installing a package you do not have

Everything this course needs is already in `requirements.txt`. When you meet a
`ModuleNotFoundError` anyway, or a line above says a reader needs something extra, install it
**into the environment** — never outside it:

```bash
pip install openpyxl                # one package, by name
pip install -r requirements.txt     # or the whole course list again
```

Two rules, and both are about where it lands:

- Your prompt must show `(.venv)` **first**. `pip -V` prints the path it is about to install into;
  that path has to contain `.venv`.
- Type `pip`, not `pip3`, and never `sudo`. Inside an activated environment `pip` is already the
  environment's own.

Then check it arrived:

```bash
pip show openpyxl                   # name, version and location - or nothing at all
python -c "import openpyxl; print(openpyxl.__version__)"
```
"""

# The dataset each session works on, for the placeholder in section 3.
DATASETS = {"01": "coasters", "02": "sourdough", "03": "mushrooms",
            "04": "secret_recipe", "05": "bestiary", "06": "cat_cafe"}


def sessions():
    """[(number, title, slug), ...] in order, from the landing-page spec."""
    return [(slug[:2], title, slug) for slug, title, *_ in SESSIONS]


def session_block(num):
    """One fenced block for a session, subdivided by what the names are for."""
    names = [k for k, v in TOOLKIT.items() if v[0] == num]
    if not names:
        return None

    by_group = OrderedDict((g, []) for g in GROUPS)
    for name in names:
        group = TOOLKIT[name][1]
        if group not in by_group:
            raise SystemExit(f"{name!r} is in group {group!r}, which is not in GROUPS.")
        by_group[group].append(name)

    lines = []
    for group, group_names in by_group.items():
        if not group_names:
            continue
        rule = "-" * max(3, 62 - len(group))
        lines.append(f"# -- {group} {rule}")
        lines.extend(TOOLKIT[n][2] for n in group_names)
        lines.append("")
    return "```python\n" + "\n".join(lines).rstrip() + "\n```\n"


def recap(num):
    """Everything taught up to and including this session, session by session."""
    n = int(num)
    out = ["## What you have learned so far\n"]
    out.append(
        f"Session 1, in the order you met it. Every line below has run on the course data.\n"
        if n == 1 else
        f"Sessions 1 to {n}, in the order you met them. Every line below has run on the "
        f"course data.\n")

    for sess, title, _slug in sessions():
        if int(sess) > n:
            break
        head = f"### Session {int(sess)} · {title}\n"
        block = session_block(sess)
        out.append(head + "\n" + block if block else
                   head + f"\nNo new Python: session {int(sess)} reuses what you already have.\n")
    return "\n".join(out)


def sheet(num, title, slug):
    n = int(num)
    lead = ("Everything you need to type, in one place. The first four sections are the same "
            f"every week; the last one is every name from sessions 1 to {n}."
            if n > 1 else "Everything you need to type, in one place.")
    started = (GETTING_STARTED
               .replace("SESSIONNUM", num)
               .replace("DATASETNAME", DATASETS[num])
               .replace("SESSIONDIR", slug))
    header = "\n".join("        " + line for line in PREAMBLE.splitlines())
    return f"""---
title: "Python Cheatsheet"
subtitle: "Essentials of Data Science in Python · Session {n} · {title}"
author: "Thierry Warin, PhD · HEC Montréal"
format:
  pdf:
    documentclass: article
    geometry: [top=2cm, bottom=2cm, left=2.2cm, right=2.2cm]
    fontsize: 9pt
    linestretch: 1.05
    colorlinks: true
    linkcolor: accent
    urlcolor: accent
    highlight-style: github
    include-in-header:
      text: |
{header}
---

<!-- Generated by scripts/build_cheatsheets.py — edit that, not this. -->

{lead}

{started}
{recap(num)}"""


# LaTeX picks a random six-letter tag for every embedded font subset, seeded
# from the clock, so two builds of an unchanged sheet differ in a dozen places
# and rewrite six binaries into the history for nothing. These are TeX Live's
# documented reproducible-build switches; with them set, an unchanged sheet
# rebuilds byte-identical. The epoch is arbitrary and fixed — 1 January 2026.
REPRODUCIBLE = {"SOURCE_DATE_EPOCH": "1767225600", "FORCE_SOURCE_DATE": "1"}


def render(qmd):
    r = subprocess.run(["quarto", "render", qmd.name, "--to", "pdf"],
                       cwd=qmd.parent, capture_output=True, text=True,
                       env={**os.environ, **REPRODUCIBLE})
    if r.returncode:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
    return r.returncode == 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_render = "--no-render" not in sys.argv
    wanted = set(args) or None

    unknown = wanted - {n for n, _, _ in sessions()} if wanted else set()
    if unknown:
        raise SystemExit(f"No such session: {', '.join(sorted(unknown))}")

    written = failed = 0
    for num, title, slug in sessions():
        if wanted and num not in wanted:
            continue
        qmd = ROOT / slug / f"{FILENAME}.qmd"
        qmd.write_text(sheet(num, title, slug), encoding="utf-8")
        written += 1
        if not do_render:
            print(f"  session {num}: {qmd.relative_to(ROOT)}")
            continue
        ok = render(qmd)
        pdf = qmd.with_suffix(".pdf")
        size = f"{pdf.stat().st_size / 1024:.0f} KB" if pdf.exists() else "—"
        print(f"  session {num}: {'ok  ' if ok else 'FAIL'} {size:>7}  "
              f"{pdf.relative_to(ROOT)}")
        failed += not ok

    print(f"\n{written} cheatsheet(s) written"
          + (f", {failed} failed to render" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
