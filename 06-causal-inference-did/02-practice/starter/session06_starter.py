"""Session 6 practice — The Cat Café Effect.

    python session06_starter.py

Every planted effect in this dataset is POSITIVE and GROWING. The estimator most
people reach for first returns a significant NEGATIVE number. The exercise is to
get the wrong answer, understand exactly why, and then get the right one.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import essentials as ess  # noqa: E402

pd.set_option("display.width", 120)

d = ess.load("cat_cafe")
truth = ess.cat_cafe_truth()

print(d.groupby("cat_arrival_quarter").cafe_id.nunique().to_string())
print(f"never-treated cafés: {(d.cat_arrival_quarter == 0).sum()}\n")


# ---------------------------------------------------------------- stage 1
# TODO: plot mean dwell_min by quarter, one line per cohort. Do this FIRST —
#       the raw data looks entirely reasonable, and that is the lesson.
twfe = smf.ols("dwell_min ~ has_cat + C(cafe_id) + C(quarter)", data=d).fit()
print(f"stage 1 · TWFE has_cat = {twfe.params['has_cat']:+.4f}  "
      f"se {twfe.bse['has_cat']:.4f}  t {twfe.tvalues['has_cat']:+.2f}  "
      f"p {twfe.pvalues['has_cat']:.1e}")
print(f"          planted effects range "
      f"{truth.true_att.min():+.2f} to {truth.true_att.max():+.2f}, "
      f"mean {truth.true_att.mean():+.2f}")
# TODO: write one sentence describing what a careless analyst would publish.


# ---------------------------------------------------------------- stage 2
import diff_diff as dd  # noqa: E402

bacon = dd.BaconDecomposition().fit(
    d, outcome="dwell_min", unit="cafe_id", time="quarter",
    first_treat="cat_arrival_quarter")
print("\nstage 2 · Goodman-Bacon decomposition")
print(bacon.to_dataframe()[["treated_group", "control_group", "comparison_type",
                            "estimate", "weight"]].round(4).to_string(index=False))
print(f"\n  TWFE {bacon.twfe_estimate:+.4f} = weighted sum of its 2x2s "
      f"(error {bacon.decomposition_error:.1e})")
print(f"  weight on later_vs_earlier (forbidden): "
      f"{bacon.total_weight_later_vs_earlier:.1%}")
# TODO: group by comparison_type. What weight and average estimate for each?
# TODO: in the later_vs_earlier rows, WHICH CAFÉS are the control group?
# TODO: explain in 3-4 sentences why that yields a negative number when the
#       true effects are positive and growing. This is the whole session.


# ---------------------------------------------------------------- stage 3
# TODO: cs = dd.CallawaySantAnna(control_group="not_yet_treated").fit(
#           d, outcome="dwell_min", unit="cafe_id", time="quarter",
#           first_treat="cat_arrival_quarter", aggregate="group")
# TODO: report cs.overall_att. Compare its SIGN with TWFE's.
# TODO: build a table of every ATT(g,t) from cs.group_time_effects — a dict
#       keyed by (cohort, quarter) whose values hold "effect", "se" and
#       "skip_reason" — and merge it against `truth`.
# TODO: how many post-treatment cells land within 2 se of the planted value?
# TODO: report cs.group_effects against the planted per-cohort averages.
# TODO: what does control_group="not_yet_treated" change, mechanically?


# ---------------------------------------------------------------- stage 4
# TODO: add event_time = quarter - cohort.
# TODO: average estimates by event_time; plot with CIs and the planted truth.
# TODO: are the pre-treatment estimates flat? Mean, and largest absolute value?
# TODO: how many placebo cells are individually significant at 5%? How many
#       would you EXPECT, given the number of tests? That is the real lesson.


# ---------------------------------------------------------------- stage 5
# TODO: which post-treatment cells come back NA? How many?
# TODO: how many cafés are untreated in quarter 11? (Count it.)
# TODO: TWFE reported a number for those quarters. Where did it get it?
# TODO: breakdown value — a differential trend of delta per quarter contaminates
#       ATT(g,t) by roughly delta * (t - g + 1), so solve for the delta that
#       drives your cohort estimate to zero. Compare with the pre-trend slope
#       you can actually measure.
# TODO: what should the cafés have done differently, at the design stage?


print("\nWork down the TODOs. Answer key in ../SOLUTION.md — after you try.")
