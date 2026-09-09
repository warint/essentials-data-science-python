"""Session 1 practice — The Roller Coaster Problem.

Run it, then work down the TODOs. It loads the data and gets out of the way.

    python session01_starter.py

There is a specific right answer: an exponent you can look up in a physics
textbook. Recover it from the data first, then check.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import essentials as ess  # noqa: E402

d = ess.load("coasters")

print(f"{len(d)} coasters, {d.shape[1]} columns")
print(d[["coaster", "drop_m", "speed_kmh", "propulsion"]].head(), "\n")


# ---------------------------------------------------------------- stage 1
# Plot speed against drop BEFORE fitting anything. Then fit the obvious model
# and write down its R^2 — you will want it later.
#
# TODO: plot, fit `speed_kmh ~ drop_m`, print the coefficient with its units.

lin = smf.ols("speed_kmh ~ drop_m", data=d).fit()
print(f"stage 1 · all 84 rows: slope {lin.params['drop_m']:.3f} km/h per metre, "
      f"R2 {lin.rsquared:.4f}")


# ---------------------------------------------------------------- stage 2
# Residuals versus fitted. Name the shape you see.
#
# TODO: plot lin.resid against lin.fittedvalues.
# TODO: bin the residuals by drop height and take the mean in each bin. Watch
#       what the SIGN of the mean residual does as you move across the range.
# TODO: add I(drop_m**2) and test whether it belongs.


# ---------------------------------------------------------------- stage 3
# You are looking for v = c * h^k. Logs make that a straight line whose slope
# is k and whose intercept is log(c).
#
# TODO: build log columns, fit, report k with a 95% interval, and report c.
# TODO: does your interval contain a simple fraction? Which physical statement
#       does that fraction correspond to?


# ---------------------------------------------------------------- stage 4
# Which rows are running the regression?
#
# TODO: fit = <your log-log model on all 84 rows>
#       inf = fit.get_influence()
#       leverage = inf.hat_matrix_diag        # confirm: leverage.sum() == p
#       cooks    = inf.cooks_distance[0]
# TODO: print the six rows with the largest Cook's distance. READ them. What do
#       they have in common that the model does not know about?
# TODO: re-estimate k with and without them. Report both.
# TODO: decide — and write down the argument, not just the decision.


print("\nWork down the TODOs. The answer key is in ../SOLUTION.md — after you try.")
