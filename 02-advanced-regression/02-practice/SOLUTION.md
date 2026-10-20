# Session 2 · The Sourdough Optimum — answer key

*Read this after you have tried. Every number is asserted by
`scripts/verify_puzzles.py`.*

## The planted answer

The generator's response surface, with $u = \text{hydration} - 75$ and
$v = \text{oven} - 225$:

$$
\text{rise} = 40 + 1.10u - 0.055u^2 + 0.30v - 0.020v^2 + 0.015uv + \text{baker} + \varepsilon
$$

Setting both partial derivatives to zero gives the optimum:

| | |
|---|---|
| **hydration** | **86.62 %** |
| **oven** | **236.86 °C** |

The Hessian is negative definite, so this is a genuine peak rather than a saddle.
Anything from roughly **82% to 91%** hydration is statistically
indistinguishable from the maximum, and that range — not the point — is the
honest recommendation.

## What each stage should have produced

**Stage 1.** The linear model recommends **95% hydration and 260 °C**: the
top-right corner of the design. Both coefficients are positive, and a plane has
no interior maximum, so the advice is always "more of everything". If a
recommendation lands on the boundary of your data, that is a fact about your
model, not about baking.

**Stage 2.** `I(hydration_pct**2)` comes in at −0.0557 (planted: −0.055). The
naive peak $-\beta_1/2\beta_2$ from the hydration-only quadratic is **81.1%** —
already much better than the corner, still two points off, because two pieces
are still missing.

The marginal value of water depends on where you are:

| at hydration | one more point is worth |
|---|---|
| 65% | **+2.2 mm** |
| 80% | **+0.6 mm** |
| 92% | **−0.8 mm** |

**Stage 3.** The interaction is **+0.01506** (planted 0.015) with **t = 17.00**.
Solving $\partial y/\partial v = 0$ for the best temperature at a given
hydration:

| hydration | best oven |
|---|---|
| 65% | **229.0 °C** |
| 90% | **238.2 °C** |

Nine degrees apart. An additive model returns the same temperature for both, and
that is exactly what it gets wrong: **wetter dough wants a hotter oven.**

**Stage 4.** The baker table is where the trap is visible without any modelling:

| baker | mean hydration | mean rise |
|---|---|---|
| A | 62.5 | 16.7 |
| D | 75.5 | **33.4** |
| G | 86.5 | 29.5 |
| H | **88.9** | 27.2 |

Baker H works closest to the true optimum and gets a *worse* rise than baker D.
The generator gives the bakers who prefer wet dough lower skill, so hydration and
skill move in opposite directions. The regression cannot separate them unless you
tell it bakers exist.

| model | $R^2$ | AIC | hydration\* | error |
|---|---|---|---|---|
| linear + baker FE | 0.490 | 1819.4 | 95.00 | +8.38 pp |
| quadratics, no interaction | 0.919 | 1380.6 | 84.86 | −1.75 pp |
| interaction, **no baker FE** | 0.904 | 1411.1 | 82.41 | **−4.20 pp** |
| everything | **0.965** | **1185.5** | **86.69** | **+0.07 pp** |

The third row is the lesson. **$R^2 = 0.904$, every coefficient significant, and
the recommendation is 4.2 percentage points wrong.** Nothing in that model's own
output reveals it. This is omitted variable bias where the omitted variable was
not unobservable — it was sitting in a column called `baker`.

**Stage 5.** The confidence band on predicted rise is flat near the top. Quoting
86.69% implies a precision the data does not support.

## The distractors

`starter_age_days` and `flour` are **pure noise**. Adding them raises $R^2$ by a
hair, lowers adjusted $R^2$, and raises AIC. Anyone who watched $R^2$ rise and
concluded the variables mattered has met the reason adjusted $R^2$ exists — and
the reason it is not much of a defence either.

## What full marks look like

- A recipe stated as a **range** (roughly 82–91% hydration, 233–240 °C), not a
  point.
- The two conditional temperatures, with the sentence "wetter dough wants a
  hotter oven".
- The optimum with and without baker fixed effects, **and** the $R^2$ of the
  model without them, with the observation that the fit statistic gave no
  warning.
- One sentence on the cost of fixed effects: seven parameters, no ability to ask
  whether wetter bakers do better, and nothing identified that is constant
  within a baker.

## Common wrong answers

| Reported | What happened |
|---|---|
| 95% / 260 °C | fitted a plane and believed the corner |
| ~81% | quadratic in hydration only |
| ~82.4% | had the quadratics and the interaction, forgot the bakers |
| ~84.9% | had the quadratics and the bakers, forgot the interaction |
| "86.69% / 236.9 °C" | right answer, false precision |
| "hydration has a positive effect" | read a main effect as an effect |
