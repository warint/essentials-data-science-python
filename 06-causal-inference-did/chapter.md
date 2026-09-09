# 6 · Causal Inference

> Every planted effect in this dataset is positive and growing. The estimator
> most people reach for first returns **−2.09 with a *p*-value of 0.00001**. Not
> a noisy zero — a confident, significant, backwards answer.

## 6.1 The question the first five chapters could not answer

Every method so far has described an association and then, carefully, declined
to call it an effect. Chapter 1 insisted on "associated with". Chapter 2 showed
the same association taking values spanning a factor of four depending on which
variation identified it. Chapter 3 ended by noting that its mushrooms were a
sample somebody chose to collect.

The reason for all that caution is that the question — *did the cat do anything?*
— is not a question about the data you have. It is a question about data you do
not have and cannot get.

David Hume asked what we actually observe when we say one thing causes another,
and answered: contiguity, succession, and constant conjunction. The billiard ball
moves; another moves after it; this has always happened. What we never observe is
the *necessary connection*.

His second formulation is the one that founded modern causal inference, though it
took two centuries to be taken up. A cause, he wrote, is an object followed by
another, *and where, if the first object had not been, the second never had
existed*. That subordinate clause is a **counterfactual**: a statement about a
world that did not happen. It cannot be observed, in principle, for the same
reason a road not taken has no destination.

> Hume treated this as a sceptical problem about knowledge. Modern statistics
> treats it as a **missing data problem**, which turns out to be the more
> productive framing — because missing data can sometimes be recovered, under
> assumptions you can state, defend and occasionally test.

## 6.2 Potential outcomes

The framework that formalises Hume's clause is due to Neyman and, in its modern
form, to Rubin (1974). For each unit $i$ define two quantities:

$$
Y_i(1) = \text{the outcome if } i \text{ is treated},
\qquad
Y_i(0) = \text{the outcome if it is not}.
$$

Both are properties of the unit, defined whether or not the treatment happens.
The causal effect *for that unit* is $\tau_i = Y_i(1) - Y_i(0)$, and it is
exactly what Hume said we cannot observe: whichever treatment the unit receives,
the other potential outcome is missing, permanently and by construction.

This is the **fundamental problem of causal inference** (Holland, 1986), and it
means **no individual causal effect is ever identified.** What we can hope for is
an average.

**ATE** — average treatment effect, $\operatorname{E}[Y(1) - Y(0)]$ over the
whole population: what would happen if everyone were treated versus no one.

**ATT** — average treatment effect on the treated,
$\operatorname{E}[Y(1) - Y(0) \mid T = 1]$: what the treatment did for those who
actually received it. Usually the policy-relevant quantity, and usually not equal
to the ATE.

### Why naive comparisons fail, exactly

What you can compute is the difference in observed means. Add and subtract the
same term:

$$
\underbrace{\operatorname{E}[Y \mid T{=}1] - \operatorname{E}[Y \mid T{=}0]}_{\text{what you can compute}}
=
\underbrace{\operatorname{E}[Y(1) - Y(0) \mid T{=}1]}_{\text{ATT}}
+
\underbrace{\operatorname{E}[Y(0) \mid T{=}1] - \operatorname{E}[Y(0) \mid T{=}0]}_{\text{selection bias}} .
$$

The second term is the difference between the treated and untreated groups **in
the absence of treatment**. If cafés that adopt a cat would have differed from
those that did not even without one, the observed difference is the effect plus
that gap, and **nothing in the data separates them.**

On the café data, the simplest naive comparison — mean dwell time when a cat is
present against when it is absent — gives:

| | |
|---|---|
| mean dwell, cat absent | 43.984 |
| mean dwell, cat present | 58.959 |
| **naive difference** | **+14.975** |
| planted mean ATT | **+11.917** |

Overstated by 26%, and here the selection term is not even about *which* cafés
adopt — every café eventually does. It is about *when*: the "cat present"
observations are later quarters, and everyone's dwell time drifts up by 0.55
minutes a quarter regardless. **A before-and-after comparison attributes the
trend to the treatment.** This is also chapter 1's regression to the mean, in the
form that matters most: a single-group before-and-after comparison is not
evidence.

## 6.3 Randomisation, and what it buys

Randomised assignment sets the selection term to zero, and it is worth being
precise about why. If treatment is assigned by a coin flip, $T$ is independent of
everything about the unit — including $Y(0)$ and $Y(1)$ — so
$\operatorname{E}[Y(0) \mid T{=}1] = \operatorname{E}[Y(0) \mid T{=}0]$ and the
bias term vanishes.

Note what randomisation does **not** do. It does not make the groups identical;
in any finite sample they will differ, sometimes substantially. It makes them
differ *in a way unrelated to the outcome*, so the difference is noise rather than
bias — and noise is what standard errors are for. It also balances variables
nobody measured or thought of, which is the property **no observational method
can replicate.**

## 6.4 When you cannot randomise

**Conditional independence** (unconfoundedness, ignorability) — given covariates
$X$, treatment is as good as random: $\{Y(0), Y(1)\} \perp T \mid X$. Every
determinant of both treatment and outcome is in $X$. **This assumption is not
testable**, because it concerns the missing potential outcome.

**Overlap** (common support) — for every value of $X$, the probability of
treatment is strictly between 0 and 1. If some units would never be treated under
any circumstances, there is no comparison to be made for them, and an estimate
that includes them is extrapolating rather than comparing. Unlike conditional
independence, overlap **is** checkable.

Under both, matching or regression adjustment recovers the ATT. Rosenbaum and
Rubin (1983) proved the result that made this tractable: if treatment is
unconfounded given a covariate vector, it is unconfounded given the scalar
**propensity score** $\Pr(T = 1 \mid X)$ alone, so a many-dimensional matching
problem collapses to one dimension.

### Balance is the diagnostic, not fit

The most important shift here is what counts as evidence that an estimate is any
good. Chapter 1 judged a model by its residuals. **A matching estimator is judged
by whether the groups it compares are actually comparable** — standardised
differences on every covariate, conventionally below 0.1.

That is the result to report: not the $R^2$ of the propensity model, which is
irrelevant, and not its coefficients, which are a means and not an end.

One caution about the method. Propensity-score matching is the most popular
approach and not the best one: matching on a scalar score discards information
and can *increase* imbalance as the caliper tightens, a pathology documented at
length by King and Nielsen (2019). Matching directly on the covariates, or
weighting rather than matching, generally does better. The score remains a good
diagnostic for overlap even where it is a poor matching device.

> And the deeper limitation: **matching and regression adjustment are the same
> assumption in different clothing.** Both require that conditioning on observed
> covariates suffices. Neither creates identification; they only implement it.
> When the assumption fails, both fail, in the same direction.

## 6.5 Bad controls

The instinct that adding covariates is prudent is wrong in two specific cases,
and both are about causal structure rather than statistics.

A **mediator** sits on the causal path from treatment to outcome. Conditioning on
it blocks part of that path, so the coefficient on treatment no longer measures
the total effect — it measures whatever is left after the mediated portion has
been removed.

A **collider** is a common consequence of treatment and outcome. Conditioning on
it *creates* an association that does not exist (Elwert & Winship, 2014).

> **"Closer to the right answer" is not the same as "the right method."** In real
> work the truth is unknown, so the only evidence available is whether the
> procedure is justified. A procedure that is wrong for a reason you can state
> will be wrong by an unknown amount in the next dataset, and no diagnostic will
> announce it.

Deciding what belongs in the regression is a claim about causal structure, not a
statistical choice. Cinelli, Forney & Pearl (2022) is the short guide worth
keeping beside the keyboard.

## 6.6 The 2×2, and the interaction that is the estimate

Difference-in-differences takes the other route. Rather than assuming the
confounders are all observed, it uses **timing** to difference the unobserved
ones away.

Two groups, two periods:

| | before | after | change |
|---|---|---|---|
| treated | 50 | 85 | +35 |
| control | 35 | 55 | +20 |
| | | | **DiD = +15** |

The treated group rose by 35, but things were changing anyway; the control's +20
is our estimate of what would have happened without treatment. As a regression:

$$ Y = \beta_0 + \beta_1 \text{Treated} + \beta_2 \text{Post} + \beta_3 (\text{Treated}\times\text{Post}) + \varepsilon $$

with $\beta_0 = 35$ (control, before), $\beta_1 = 15$ (initial gap), $\beta_2 =
20$ (common time change) and $\beta_3 = 15$ — **the interaction is the estimate.**

The design is older than its name: Wing, Simon & Bello-Gomez (2018) trace it to
John Snow's 1855 study of London water companies, whose customers had not chosen
their supplier.

### On the café data, a clean 2×2 works perfectly

Take cohort 4 (treated in quarter 4) against cohort 10 (not yet treated), in
quarter 6:

| | quarter 1 | quarter 6 |
|---|---|---|
| cohort 4 | 42.785 | 57.506 |
| cohort 10 | 42.761 | 45.427 |
| **gap** | **+0.024** | **+12.079** |

$$ \text{DiD} = 12.079 - 0.024 = \mathbf{+12.05} $$

The planted $\text{ATT}(4,6)$ is **+12.00**. A single clean 2×2 recovers it to
within six hundredths of a minute. (Computed from the unrounded means the figure
is +12.054; the table above is rounded to three decimals.)

> **Hold on to that.** The design is not the problem. Everything that goes wrong
> in the rest of this chapter goes wrong in the *pooling* of many such
> comparisons into one number.

### Parallel trends

The assumption doing all the work: *in the absence of treatment, the two groups
would have followed the same trend.*

It does **not** require the groups to be similar — levels may differ freely,
because $\beta_1$ absorbs that. It requires their *changes* to have matched. And
it is **untestable**, being a statement about a counterfactual.

It is also **scale-dependent**, which is under-appreciated. Parallel trends in
levels is not parallel trends in logs, and the choice is part of the assumption
rather than a presentational detail. On this data:

| specification | TWFE estimate | $t$ |
|---|---|---|
| levels | **−2.0877** | −4.42 |
| logs | −0.0038 | −0.46 |

Both are wrong — the truth is positive — but they are wrong *differently*: levels
gives a confidently wrong sign, logs gives a null. **Say which scale you chose,
and why.**

## 6.7 Two-way fixed effects, and where it breaks

The natural generalisation to many units and periods:

$$ Y_{it} = \alpha_i + \gamma_t + \beta D_{it} + \varepsilon_{it} $$

```python
twfe = smf.ols("dwell_min ~ has_cat + C(cafe_id) + C(quarter)", data=d).fit()
```

```
has_cat = -2.0877   se 0.4725   t -4.42   p 1.0e-05
```

The planted effects range from +0.50 to +36.00 with a mean of **+11.92**. The
estimator returned the wrong sign with a small standard error and three stars.

This is not a contrived failure. It is the central finding of the
difference-in-differences literature since about 2018, and between them
Goodman-Bacon (2021) and de Chaisemartin & D'Haultfœuille (2020) called into
question a large body of published work.

## 6.8 Goodman-Bacon: where that number came from

With staggered adoption, a two-way fixed effects estimate is **exactly** a
weighted average of every 2×2 comparison the data allows.

```python
import diff_diff as dd
bacon = dd.BaconDecomposition().fit(
    d, outcome="dwell_min", unit="cafe_id", time="quarter",
    first_treat="cat_arrival_quarter")
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

De Chaisemartin & D'Haultfœuille (2020) put it more sharply still: the implied
weights can be **negative**, so the estimand may lie outside the range of every
underlying effect. It is not a biased average of the true effects. It is not an
average of them at all.

Two conditions are both required:

- **Staggered adoption.** With one common date there are no already-treated
  controls and TWFE is fine.
- **Dynamic or heterogeneous effects.** With one constant effect everywhere the
  same number is subtracted from both sides and it cancels.

Both hold here, and both hold in most real policy rollouts. **Dynamic effects are
the normal case** — almost nothing works fully on its first day.

## 6.9 Group-time effects

Callaway and Sant'Anna (2021) refuse to pool. Instead of one $\beta$, estimate
one effect per cohort per period:

$$ \text{ATT}(g,t) = \operatorname{E}\left[Y_t(g) - Y_t(0) \mid G = g\right] $$

Each is a clean 2×2 — the kind that worked perfectly in §6.6 — whose controls are
only units **not yet treated** at time $t$. The forbidden comparisons are not
down-weighted; they are never formed.

```python
cs = dd.CallawaySantAnna(control_group="not_yet_treated").fit(
    d, outcome="dwell_min", unit="cafe_id", time="quarter",
    first_treat="cat_arrival_quarter", aggregate="group")

cs.overall_att          # +10.0566
```

Every identified post-treatment cell lands **within two standard errors** of the
planted value — 9 of 9, mean absolute error 0.382 on effects up to 24. Per
cohort:

| cohort | CS estimate | planted |
|---|---|---|
| 4 | **+13.470** (se 0.396) | +14.000 |
| 7 | **+3.009** (se 0.447) | +3.000 |

## 6.10 The event study

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
tracking the planted line.

Note that the *raw* event-time means do not show this. Averaging `dwell_min` by
event time gives 44.01, 43.91, 44.62, 45.10 at event times −4 to −1 — a gentle
rise that looks like anticipation and is in fact the common time trend, because
later event times fall in later calendar quarters. **The estimator is what
removes it; raw means by event time are not an event study.**

### Reading the pre-period honestly

Twelve placebo cells, mean **−0.0016**, largest absolute estimate 1.18 — and
**one of the twelve is individually significant at 5%**.

At a 5% level on twelve tests you expect about 0.6 false rejections, so one is
exactly what a design with genuinely parallel trends looks like. And parallel
trends *does* hold here, by construction.

> **Pre-trend tests are a multiple-testing problem.** Reading them one star at a
> time will make you reject good designs; and because they are typically
> under-powered, it will also let you accept bad ones.

There is a worse version of this, and it is common. If your design has only one
pre-period, it *is* the reference period and its estimate is **mechanically
zero**. A flat pre-trend consisting of one arithmetically-zero point is not
evidence of anything, and presenting it as though it were is the most widespread
abuse of the event-study plot.

## 6.11 What is not identified

Nine post-treatment cells come back **NA**: everything from quarter 10 onwards,
and cohort 10 entirely.

The reason is countable. **In quarter 11, zero cafés are untreated.** By quarter
10 every café has a cat, so no not-yet-treated unit remains and nothing
identifies those effects.

Two-way fixed effects reported a number for exactly those quarters. It got it by
using already-treated cafés as controls, which is where much of the −2.09 came
from.

> **An estimator that returns NA is doing you a favour.** The nine NAs are not a
> shortcoming of the method; they are a correct description of what this research
> design can support. A method that hands you three stars for a quantity your data
> cannot identify is worse than one that refuses.

And this is a lesson about design, not software. **If every unit is treated by
the end of your panel, you have designed a study that cannot evaluate its own
last phase**, and no estimator invented later will fix it. Holding back twenty of
the 186 cafés would have identified every cell — a decision made before any data
exists.

## 6.12 Sensitivity instead of assertion

Parallel trends is untestable, so Rambachan and Roth (2023) change the question:
**how large would a violation have to be before the conclusion changes?**

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

> **This is not a test, and nothing here has been tested.** A sensitivity analysis
> converts an untestable assumption into a statement about **how much of it the
> result needs.** That is real progress over asserting the assumption, and it is
> not the same thing as evidence for it.

## 6.13 What difference-in-differences cannot fix

**Anything that changes at the same time as treatment.** Fixed effects remove
what is constant; the second difference removes what is common. A shock hitting
only the treated group in the treatment period is indistinguishable from the
treatment, and no amount of data separates them.

**Anticipation.** If units change behaviour before adoption because they know it
is coming, the pre-period is already contaminated and the "before" is not a clean
baseline. This shows as a non-zero pre-trend — when there are enough pre-periods
to see one.

**Composition change.** If the units in the panel change — entering or leaving
the sample — the first difference compares different populations rather than the
same one over time.

**Spillovers.** The control group must be unaffected. If a resident cat in one
café shifts behaviour at the café next door, the controls are partially treated
and the estimate is attenuated toward zero.

## 6.14 The rest of the toolkit

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

## 6.15 What to report

1. **The estimand, named.** ATE or ATT — they answer different questions.
2. **The adoption structure.** How many units in each cohort, how many periods
   before and after, and which cohorts are unusable. A table of this is worth
   more than a paragraph asserting the design is sound.
3. **The identifying assumption, as an argument.** Why should the gap have stayed
   constant? Institutional reasoning, not a citation to the assumption.
4. **The naive estimate alongside the adjusted one.** The distance between them
   is the size of the selection problem, and hiding it hides the reason for the
   analysis.
5. **A decomposition** if you use TWFE at all, showing the forbidden weight.
6. **A staggered-robust estimator alongside TWFE.** If they agree, the weights
   were benign and you have shown it. If they disagree, **the decomposition is
   the finding.**
7. **An event study**, with the reference period marked as such — and if there
   are too few pre-periods to see a trend, say that rather than implying the flat
   line is evidence.
8. **Standard errors clustered by unit**, with the number of clusters.
9. **The scale.** Levels or logs is part of the assumption.
10. **What is not identified**, explicitly.
11. **What could have moved at the same time.** The most credible papers name the
    confounding events they considered and explain why each is implausible —
    the modern version of Snow pointing out that nobody chose their water company.
12. **A breakdown value.** Worth more than a pre-trend $p$-value, and unlike the
    $p$-value it can be reported even when there is nothing to test.

## 6.16 Summary

1. No individual causal effect is ever identified. The observed difference is
   **ATT + selection bias**, and nothing in the data separates them.
2. Randomisation zeroes the selection term and balances what nobody measured. No
   observational method replicates that.
3. Conditional independence is untestable; overlap is checkable. **Balance, not
   fit, is the diagnostic.**
4. Matching and regression adjustment are the same assumption in different
   clothing.
5. Mediators and colliders are bad controls. "Closer to the right answer" is not
   "the right method".
6. **A clean 2×2 works.** On this data it recovers +12.05 against a planted
   +12.00. The failure is in the pooling.
7. Under staggered adoption, TWFE is a weighted average of 2×2s and **some
   weights are negative** — so the estimand may lie outside the range of every
   true effect.
8. The bug is already-treated units used as controls, and it needs staggered
   adoption **and** dynamic effects. Both are the normal case.
9. Read pre-trends as multiple tests. A single mechanically-zero pre-period is
   not evidence.
10. An estimator that returns NA is doing you a favour.
11. Parallel trends is scale-dependent. Say which scale.
12. Report a breakdown value.

## 6.17 Where the six chapters leave you

The through-line has been the same throughout: **the fit statistic is not the
diagnostic, and the confident answer is the one to check.**

| chapter | the output that looked excellent | what was wrong |
|---|---|---|
| 1 | $R^2 = 0.96$ | the slope meant nothing; six rows set the exponent |
| 2 | $R^2 = 0.90$, all coefficients significant | the recommendation was four points out |
| 3 | 87.7% accurate | it poisoned the forager one meal in four |
| 4 | 25 ingredients "identified" | 19 were noise, and one real one was invisible |
| 5 | 87.3% accurate | a whole species was ignored for 1.7 points |
| 6 | $p = 10^{-5}$ | the wrong sign |

In every case the way to find out was the same: **ask what the method was
actually comparing.**

That question is the one thing worth carrying out of all six chapters. It is what
a residual plot answers, what a variance decomposition answers, what a confusion
matrix answers, what a selection frequency answers, and what a Goodman-Bacon
decomposition answers. The methods differ; the question does not.

## Further reading

The written treatment is
[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/),
chapters [10 · Causal Inference I](https://warint.github.io/quantitative-methods/book/session-10.html)
and [11 · Causal Inference II: Difference-in-Differences](https://warint.github.io/quantitative-methods/book/session-11.html).

The potential-outcomes notation is **Rubin (1974)**, though the idea traces to
Neyman's 1923 work on agricultural experiments. **Holland (1986)** named the
fundamental problem and drew the boundary the discipline still uses — "no
causation without manipulation" — which remains contested and clarifying.

**Rosenbaum & Rubin (1983)** proved the propensity-score result that made
observational adjustment tractable. **Imbens (2004)** surveys the estimators that
followed and **Stuart (2010)** is the practical review. The corrective literature
is as important as the constructive one: **King & Nielsen (2019)** argue that
propensity-score matching specifically often *increases* imbalance;
**Elwert & Winship (2014)** set out endogenous selection bias, the family of
errors created by conditioning on colliders; and **Cinelli, Forney & Pearl
(2022)** give the compact treatment of which controls help and which do damage.

For difference-in-differences, **Wing, Simon & Bello-Gomez (2018)** traces the
design to Snow's 1855 water study and sets out modern practice.

The last decade has been dominated by a single discovery about staggered
adoption. **Goodman-Bacon (2021)** decomposed the two-way fixed effects estimator
into its underlying 2×2 comparisons and showed that some use already-treated
units as controls; **de Chaisemartin & D'Haultfœuille (2020)** showed the implied
weights can be negative, so the estimand may lie outside the range of every
underlying effect.

The constructive replacements estimate cohort-specific effects against clean
comparison groups and aggregate them explicitly: **Callaway & Sant'Anna (2021)**
for group-time average treatment effects, **Sun & Abraham (2021)** for event
studies, **Borusyak, Jaravel & Spiess (2024)** for the imputation approach.
**Rambachan & Roth (2023)** replaces the pre-trend ritual with a sensitivity
analysis. **Roth, Sant'Anna, Bilinski & Poe (2023)** is the survey that ties the
whole literature together.

On inference, the reference remains **Bertrand, Duflo & Mullainathan (2004)**,
discussed in chapter 1: DiD data are serially correlated panels, and standard
errors that ignore it reject true nulls at many times the nominal rate.
