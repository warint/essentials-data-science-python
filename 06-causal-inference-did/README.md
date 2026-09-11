# Session 6 · Causal Inference

**Where difference-in-differences goes wrong, and what replaced it.**

The last session, and the one with the most consequential trap. Potential
outcomes, the 2×2 design, two-way fixed effects and why it fails under staggered
adoption, the Goodman-Bacon decomposition, group-time estimators, event studies,
and sensitivity analysis instead of assertion.

## The puzzle

**The Cat Café Effect.** 186 cafés over 12 quarters, each adopting a resident cat
in quarter 4, 7 or 10. Did the cat work?

Every planted effect is **positive** and **grows** with time since adoption,
averaging +11.92 minutes. And then:

```
two-way fixed effects:  has_cat = -2.0877   t = -4.42   p = 0.00001
```

**The wrong sign, with three stars.** The Goodman-Bacon decomposition shows
exactly where it came from: TWFE is a weighted average of six 2×2 comparisons,
and **50.3% of the weight** sits on comparisons that use *already-treated* cafés
as the control group. Those average **−12.56**. When effects grow over time, the
control group's own treatment effect is being subtracted from the treated
group's.

Callaway–Sant'Anna, using only not-yet-treated controls, recovers every
identified cell **within two standard errors** of the planted truth — cohort 7
comes back at +3.009 against a planted +3.000.

And it returns **NA** for nine post-treatment cells, because by quarter 10 every
café has a cat and nothing identifies them. TWFE reported numbers for those same
quarters. *An estimator that refuses is more useful than one that invents.*

## What you should be able to do afterwards

- Read a 2×2 DiD as an interaction, and every coefficient in it
- Decompose a TWFE estimate and identify forbidden comparisons
- Explain why already-treated controls bias the estimate when effects grow
- Estimate group-time ATTs, and say what the control group is for each
- Build an event study and read the pre-period as multiple tests
- Say what your design cannot identify, and what to do about it at the design stage
- Report a breakdown value instead of asserting parallel trends

## The written treatment

[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/) (Warin),
chapters [10 · Causal Inference I: Counterfactuals, Randomisation, Matching](https://warint.github.io/quantitative-methods/book/session-10.html)
and [11 · Causal Inference II: Difference-in-Differences](https://warint.github.io/quantitative-methods/book/session-11.html).
Assigned in the pre-session. Chapter 11 is the **classical** treatment;
[`chapter.md`](chapter.md) and the lecture start where it stops.

## Contents

| | |
|---|---|
| [`00-pre-session/`](00-pre-session/) | Potential outcomes, the 2×2 by hand. **70 min** — it carries the session's concepts. |
| [`01-lecture/`](01-lecture/) | The deck. Executes its own Python. |
| [`chapter.md`](chapter.md) | The lecture notes, as prose. |
| [`02-practice/`](02-practice/) | The workshop, a starter script, and `SOLUTION.md`. |
| [`PYTHON-CHEATSHEET.pdf`](PYTHON-CHEATSHEET.pdf) | Sessions 1 to 6, on a page: the environment, and every name so far. |

## Data

```python
import essentials as ess
d = ess.load("cat_cafe")       # 2,232 rows: 186 cafés x 12 quarters
ess.cat_cafe_truth()           # every planted ATT(g, t)
```

Needs `diff-diff` (in `requirements.txt`) for `BaconDecomposition`,
`CallawaySantAnna`, `SunAbraham`, `ImputationDiD` and `HonestDiD`.
