"""Session 2 practice — The Sourdough Optimum.

    python session02_starter.py

Deliverable: a recipe. One hydration percentage, one oven temperature, and an
honest interval. There is a planted optimum; it is an interior point, and it
takes quadratics, an interaction and baker fixed effects together to find it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import essentials as ess  # noqa: E402

d = ess.load("sourdough")
print(f"{len(d)} loaves, {d.baker.nunique()} bakers")


def peak(fit, baker="D", bounds=((55, 95), (190, 260))):
    """Numerically maximise a fitted surface over the design region.

    Useful as a check on the algebra: if -b1/2b2 and this disagree, one of them
    is wrong and it is worth knowing which.
    """
    def neg(z):
        row = pd.DataFrame([{"hydration_pct": z[0], "oven_c": z[1], "baker": baker}])
        return -fit.predict(row).iloc[0]
    r = minimize(neg, [75, 225], bounds=list(bounds))
    return r.x[0], r.x[1]


# ---------------------------------------------------------------- stage 1
flat = smf.ols("rise_mm ~ hydration_pct + oven_c", data=d).fit()
print(f"\nstage 1 · linear model recommends: "
      f"{peak(flat)[0]:.1f}% hydration, {peak(flat)[1]:.0f} C  "
      f"(R2 {flat.rsquared:.3f})")
# TODO: where did that land relative to the design? Why can a linear model
#       never do anything else?


# ---------------------------------------------------------------- stage 2
# TODO: add I(hydration_pct**2). Is the coefficient negative?
# TODO: compute the peak as -b1 / (2*b2) and check it against peak().
# TODO: report what one more point of hydration is worth at 65, 80 and 92 per
#       cent. The three numbers differ — say why.


# ---------------------------------------------------------------- stage 3
# TODO: add hydration_pct:oven_c. Report the coefficient and its t.
# TODO: differentiate with respect to oven_c and solve for the best temperature
#       AT A GIVEN HYDRATION. Do it at 65% and at 90%.


# ---------------------------------------------------------------- stage 4
print("\nstage 4 · look at these two columns together before fitting anything:")
print(d.groupby("baker")[["hydration_pct", "rise_mm"]].mean().round(1))
# TODO: state the confounding in one sentence.
# TODO: fit with and without C(baker). Report the optimum from each, and the
#       R^2 of the model WITHOUT the baker effects. That number is the lesson.


# ---------------------------------------------------------------- stage 5
# TODO: put a confidence band on predicted rise across hydration, holding the
#       oven at your estimated optimum:
#           full.get_prediction(grid).summary_frame()
# TODO: over what range is the prediction indistinguishable from its maximum?
# TODO: rewrite your recommendation as that range.


print("\nWork down the TODOs. Answer key in ../SOLUTION.md — after you try.")
