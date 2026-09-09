# 6 · Causal Inference

> Every planted effect in this dataset is positive and growing. The estimator
> most people reach for first returns **−2.09 with a *p*-value of 0.00001**. Not
> a noisy zero — a confident, significant, backwards answer.

## 6.1 The counterfactual you never see

For each unit, imagine two numbers: $Y_i(1)$, the outcome if treated, and
$Y_i(0)$, the outcome if not. The causal effect is $Y_i(1) - Y_i(0)$.

You never observe both. A café either has a cat or it does not, so **half of
every unit's causal information is missing by construction**. This is the
fundamental problem of causal inference, and no dataset solves it.

So we give up on individual effects and estimate an average:

$$ \text{ATT} = \mathbb{E}\left[Y_i(1) - Y_i(0) \mid \text{treated}\right] .$$

To get it we need a stand-in for $\mathbb{E}[Y_i(0) \mid \text{treated}]$ — what
the treated units *would* have done. Every method in this chapter is an argument
about that stand-in.

> A causal claim is always a claim about a counterfactual, and a counterfactual is
> never in your data. **The design is the argument for the substitute you used**,
> which is why the design section matters more than the estimator.

## 6.2 The 2×2, and the interaction that is the estimate

Two groups, two periods:

| | before | after | change |
|---|---|---|---|
| treated | 50 | 85 | +35 |
| control | 35 | 55 | +20 |
| | | | **DiD = +15** |

The treated group rose by 35, but things were changing anyway; the control's +20
is our estimate of what would have happened without treatment. The difference of
the differences is 15.

As a regression:

$$ Y = \beta_0 + \beta_1 \text{Treated} + \beta_2 \text{Post} + \beta_3 (\text{Treated}\times\text{Post}) + \varepsilon $$

with $\beta_0 = 35$ (control, before), $\beta_1 = 15$ (initial gap),
$\beta_2 = 20$ (common time change) and $\beta_3 = 15$ — **the interaction is the
estimate.** Everything in this chapter is that idea under complications.

### Parallel trends

The assumption doing all the work: *in the absence of treatment, the two groups
would have followed the same trend.*

It does **not** require the groups to be similar — levels may differ freely,
because $\beta_1$ absorbs that. It requires their *changes* to have matched. And
it is **untestable**, being a statement about a counterfactual. What you can do is
check whether trends matched before treatment and treat that as evidence about
after. That is the event study, and it is evidence rather than proof.

## 6.3 Two-way fixed effects, and where it breaks

The natural generalisation to many units and periods:

$$ Y_{it} = \alpha_i + \gamma_t + \beta D_{it} + \varepsilon_{it} $$

Unit effects, time effects, one treatment dummy.

```python
import essentials as ess
import statsmodels.formula.api as smf

d = ess.load("cat_cafe")
twfe = smf.ols("dwell_min ~ has_cat + C(cafe_id) + C(quarter)", data=d).fit()
```

```
has_cat = -2.0877   se 0.4725   t -4.42   p 1.0e-05
```

The planted effects range from +0.50 to +36.00 with a mean of **+11.92**. The
estimator returned the wrong sign with a small standard error.

This is not a contrived failure. It is the central finding of the
difference-in-differences literature since about 2018.

## 6.4 Goodman-Bacon: where that number came from

Goodman-Bacon (2021) showed that with staggered adoption, a two-way fixed effects
estimate is **exactly** a weighted average of every 2×2 comparison the data
allows.

```python
import diff_diff as dd

bacon = dd.BaconDecomposition().fit(
    d, outcome="dwell_min", unit="cafe_id", time="quarter",
    first_treat="cat_arrival_quarter")
bacon.to_dataframe()
```

| treated | control | type | estimate | weight |
|---|---|---|---|---|
| 4 | 7 | earlier_vs_later | +7.99 | 0.103 |
| **7** | **4** | **later_vs_earlier** | **−12.71** | **0.207** |
| 4 | 10 | earlier_vs_later | +14.07 | 0.200 |
| **10** | **4** | **later_vs_earlier** | **−16.76** | **0.200** |
| 7 | 10 | earlier_vs_later | +3.09 | 0.194 |
| **10** | **7** | **later_vs_earlier** | **−3.60** | **0.097** |

By type:

| type | weight | average | contribution |
|---|---|---|---|
| earlier_vs_later | 0.4968 | +8.525 | +4.235 |
| **later_vs_earlier** | **0.5032** | **−12.564** | **−6.323** |
| total | 1.0000 | | **−2.088** |

The arithmetic closes to fifteen decimal places. Half the weight averages +8.5,
the other half −12.6, they nearly cancel, and what survives is negative.

### The forbidden comparisons

In the `later_vs_earlier` rows, the **later** cohort is treated and the
**earlier** cohort is the control — a control group that already has a cat.

Take cohort 7 against cohort 4:

1. Cohort 7's dwell time rises, because its cat just arrived.
2. Cohort 4's dwell time **also** rises, because its own effect is still growing
   at +4 minutes a quarter.
3. Difference-in-differences subtracts the control's change from the treated
   group's change.
4. So it subtracts cohort 4's ongoing treatment effect — and since that effect
   grows faster than cohort 7's, the subtraction overshoots and books a negative
   number.

> **The control group's own treatment effect is being subtracted from the treated
> group's.** That is the whole bug, and one treatment dummy cannot see it.

Two conditions are both required:

- **Staggered adoption.** With one common date there are no already-treated
  controls and TWFE is fine.
- **Dynamic or heterogeneous effects.** With one constant effect everywhere the
  same number is subtracted from both sides and it cancels.

Both hold here, and both hold in most real policy rollouts. **Dynamic effects are
the normal case** — almost nothing works fully on its first day.

## 6.5 Group-time effects

Callaway and Sant'Anna (2021) refuse to pool. Instead of one $\beta$, estimate
one effect per cohort per period:

$$ \text{ATT}(g,t) = \mathbb{E}\left[Y_t(g) - Y_t(0) \mid G = g\right] $$

Each is a clean 2×2 whose controls are only units **not yet treated** at time
$t$. The forbidden comparisons are not down-weighted — they are never formed.

```python
cs = dd.CallawaySantAnna(control_group="not_yet_treated").fit(
    d, outcome="dwell_min", unit="cafe_id", time="quarter",
    first_treat="cat_arrival_quarter", aggregate="group")

cs.overall_att          # +10.0566
cs.group_time_effects   # dict keyed by (cohort, quarter)
cs.group_effects
```

Every identified post-treatment cell lands **within two standard errors** of the
planted value — 9 of 9, mean absolute error 0.382 on effects up to 24. Per
cohort:

| cohort | CS estimate | planted |
|---|---|---|
| 4 | **+13.470** (se 0.396) | +14.000 |
| 7 | **+3.009** (se 0.447) | +3.000 |

## 6.6 The event study

Averaging the cells by time since adoption:

| event time | estimate | se | planted |
|---|---|---|---|
| −8 | +0.666 | 0.465 | 0 |
| −7 | −1.181 | 0.481 | 0 |
| −3 | −0.777 | 0.515 | 0 |
| −1 | +0.421 | 0.345 | 0 |
| **0** | **+2.339** | 0.368 | +2.75 |
| 1 | +5.265 | 0.364 | +5.50 |
| 2 | +8.033 | 0.376 | +8.25 |
| 3 | +15.473 | 0.578 | +16.00 |
| 5 | +23.879 | 0.518 | +24.00 |

Flat and indistinguishable from zero before adoption; rising steadily after, and
tracking the planted line. That is what a credible design looks like.

### Reading the pre-period honestly

Twelve placebo cells, mean **−0.0016**, largest absolute estimate 1.18 — and
**one of the twelve is individually significant at 5%**.

At a 5% level on twelve tests you expect about 0.6 false rejections, so one is
exactly what a design with genuinely parallel trends looks like. And parallel
trends *does* hold here, by construction.

> **Pre-trend tests are a multiple-testing problem.** Reading them one star at a
> time will make you reject good designs; and because they are typically
> under-powered, it will also let you accept bad ones. A joint test, or a
> sensitivity bound, is the better instrument.

## 6.7 What is not identified

Nine post-treatment cells come back **NA**: everything from quarter 10 onwards,
and cohort 10 entirely.

The reason is countable. **In quarter 11, zero cafés are untreated.** By quarter
10 every café has a cat, so no not-yet-treated unit remains and nothing identifies
those effects.

Two-way fixed effects reported a number for exactly those quarters. It got it by
using already-treated cafés as controls, which is where much of the −2.09 came
from.

> **An estimator that returns NA is doing you a favour.** The nine NAs are not a
> shortcoming of the method; they are a correct description of what this research
> design can support. A method that hands you three stars for a quantity your data
> cannot identify is worse than one that refuses.

And this is a lesson about design, not software. **If every unit is treated by the
end of your panel, you have designed a study that cannot evaluate its own last
phase**, and no estimator invented later will fix it. Holding back twenty of the
186 cafés would have identified every cell — a decision made before any data
exists.

## 6.8 Sensitivity instead of assertion

Parallel trends is untestable, so Rambachan and Roth (2023) propose a better
question: **how large would a violation have to be before the conclusion
changes?**

A differential trend of $\delta$ per quarter contaminates $\text{ATT}(g,t)$ by
roughly $\delta \times (t-g+1)$. Solve for the $\delta$ that drives the estimate
to zero — the **breakdown value**:

| cohort | ATT | mean exposure | breakdown $\delta$ |
|---|---|---|---|
| 4 | +13.470 | 3.5 quarters | **+3.849 min/quarter** |
| 7 | +3.009 | 2.0 quarters | +1.504 min/quarter |

The measurable pre-treatment trend is **+0.0443 min/quarter** — about **87 times
smaller**. Which yields a defensible sentence:

> *"To explain away the cohort-4 effect, the treated cafés would have to have been
> on a differential trend of 3.85 minutes per quarter. The trend measurable before
> adoption is 0.04 minutes per quarter, roughly ninety times smaller. A violation
> of the required size would have been unmissable in the pre-period."*

`diff_diff` implements the full Rambachan–Roth procedure as `dd.HonestDiD`; the
arithmetic above is the same idea done by hand, so you can see what it rests on.

## 6.9 The rest of the toolkit

The same problem produced several estimators. They mostly agree, and knowing they
exist matters more than knowing which to prefer:

| | |
|---|---|
| **Callaway & Sant'Anna** (2021) | group-time ATTs, not-yet-treated controls |
| **Sun & Abraham** (2021) | interaction-weighted event study |
| **Borusyak, Jaravel & Spiess** (2024) | impute $Y(0)$ from untreated observations |
| **de Chaisemartin & D'Haultfœuille** (2020) | robust to treatment switching off |
| **Gardner** (2022) | two-stage: remove fixed effects, then estimate |
| **Arkhangelsky et al.** (2021) | synthetic DiD — weights units *and* periods |

All are in `diff_diff`. Report two, not one.

## 6.10 What to report

- The **design** first: who is treated, when, and what serves as control.
- A **decomposition** if you use TWFE at all, showing the forbidden weight.
- **Group-time effects**, or a stated reason for pooling.
- An **event study**, with the pre-period read as a set of tests.
- What is **not identified**, explicitly.
- A **breakdown value**, not an assertion that trends look parallel.

> A DiD result with no event study, no decomposition and no named control group is
> an interaction coefficient wearing the word "causal".

## 6.11 Summary

1. The interaction in a 2×2 is the DiD estimate.
2. Under staggered adoption, TWFE is a weighted average of 2×2s, and **some
   weights are negative**.
3. The bug is already-treated units used as controls.
4. It needs staggered adoption **and** dynamic or heterogeneous effects — both
   the normal case.
5. Group-time estimators recover the truth by never forming forbidden
   comparisons.
6. Read pre-trends as multiple tests.
7. An estimator that returns NA is doing you a favour.
8. Report a breakdown value.

## 6.12 Where the six sessions leave you

The through-line has been the same throughout: **the fit statistic is not the
diagnostic, and the confident answer is the one to check.**

Session 1's regression fitted at $R^2 = 0.96$ and reported a meaningless slope.
Session 2's model explained 90% of the variance and gave advice four points
wrong. Session 3's most accurate rule poisoned the forager. Session 4's lasso
silently dropped an ingredient. Session 5's accuracy concealed a whole species.
And this session's estimator returned the wrong sign at $p = 10^{-5}$.

In every case the output looked excellent. In every case the way to find out was
to ask what the method was actually comparing.

## Further reading

- Goodman-Bacon (2021), "Difference-in-differences with variation in treatment
  timing", *Journal of Econometrics* 225(2), 254–277.
- Callaway & Sant'Anna (2021), "Difference-in-differences with multiple time
  periods", *Journal of Econometrics* 225(2), 200–230.
- Sun & Abraham (2021), *Journal of Econometrics* 225(2), 175–199.
- de Chaisemartin & D'Haultfœuille (2020), *American Economic Review* 110(9),
  2964–2996.
- Rambachan & Roth (2023), "A more credible approach to parallel trends",
  *Review of Economic Studies* 90(5), 2555–2591.
- Borusyak, Jaravel & Spiess (2024), "Revisiting event-study designs",
  *Review of Economic Studies* 91(6), 3253–3285.
- Roth, Sant'Anna, Bilinski & Poe (2023), "What's trending in
  difference-in-differences?", *Journal of Econometrics* 235(2), 2218–2244.
