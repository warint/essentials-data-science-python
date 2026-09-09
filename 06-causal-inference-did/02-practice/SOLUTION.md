# Session 6 · The Cat Café Effect — answer key

*Read this after you have tried. Every number is asserted by
`scripts/verify_puzzles.py`.*

## What was planted

186 cafés, 12 quarters, three adoption cohorts and **no never-treated group**.
The true effect for a café treated in quarter $g$ and observed in quarter $t$ is

$$ \text{ATT}(g,t) = \delta_g \times (t - g + 1), \qquad
\delta_4 = 4.0,\; \delta_7 = 1.5,\; \delta_{10} = 0.5 $$

Every effect is **positive** and **grows linearly** with time since adoption.
They range from +0.50 to +36.00, mean **+11.92**. Parallel trends holds by
construction. There is nothing subtle about the truth.

| cohort | cafés | effect per quarter |
|---|---|---|
| 4 | 64 | +4.0 |
| 7 | 62 | +1.5 |
| 10 | 60 | +0.5 |

## Stage 1 · The wrong answer

```
has_cat = -2.0877   se 0.4725   t -4.42   p 1.0e-05
```

**The wrong sign, with three stars.** Not a noisy zero. A careless analyst would
publish "resident cats reduce customer dwell time by two minutes per visit
($p < 0.001$)" — and that specification, one treatment dummy with unit and time
fixed effects, is the one used in a large share of published
difference-in-differences work.

The raw cohort plot looks entirely reasonable. Nothing in it warns you.

## Stage 2 · Why — the heart of the session

TWFE averages **six** 2×2 comparisons, and the weighted sum reproduces the
estimate to 15 decimal places (decomposition error $5.8 \times 10^{-15}$):

| treated | control | type | estimate | weight |
|---|---|---|---|---|
| 4 | 7 | earlier_vs_later | +7.99 | 0.103 |
| **7** | **4** | **later_vs_earlier** | **−12.71** | **0.207** |
| 4 | 10 | earlier_vs_later | +14.07 | 0.200 |
| **10** | **4** | **later_vs_earlier** | **−16.76** | **0.200** |
| 7 | 10 | earlier_vs_later | +3.09 | 0.194 |
| **10** | **7** | **later_vs_earlier** | **−3.60** | **0.097** |

By type:

| type | weight | average estimate | contribution |
|---|---|---|---|
| earlier_vs_later | 0.4968 | +8.525 | **+4.235** |
| **later_vs_earlier (forbidden)** | **0.5032** | **−12.564** | **−6.323** |
| total | 1.0000 | | **−2.088** |

**50.3% of the weight sits on forbidden comparisons**, and every one of them is
negative.

### The explanation, in the words the write-up needs

In the `later_vs_earlier` rows, the **later** cohort is treated and the
**earlier** cohort is the control group — a control group that already has a cat.

Take cohort 7 against cohort 4 in quarters 4–12. Cohort 7's dwell time rises when
its cat arrives. But cohort 4's dwell time is *also* rising, because its own
effect is still growing at +4 minutes a quarter. Difference-in-differences
subtracts the control group's change from the treated group's change — so it
subtracts cohort 4's ongoing treatment effect. Since cohort 4's effect grows
faster than cohort 7's (+4.0 against +1.5), the subtraction overshoots and the
comparison books a **negative** number.

> The control group's own treatment effect is being subtracted from the treated
> group's. That is the entire bug, and a single treatment dummy cannot see it.

**Two conditions are both required**, and both are the normal case in real
rollouts:

1. **Staggered adoption.** With one common date there are no already-treated
   controls and TWFE is fine.
2. **Dynamic or heterogeneous effects.** With one constant effect everywhere, the
   same number is subtracted from both sides and it cancels.

## Stage 3 · The right answer

```
Callaway-Sant'Anna, control_group="not_yet_treated"
overall ATT = +10.0566        TWFE = -2.0877
```

Every identified post-treatment cell lands **within two standard errors** of the
planted value — 9 of 9, mean absolute error **0.382** on effects up to 24:

| cohort | quarter | estimate | se | planted | error |
|---|---|---|---|---|---|
| 4 | 4 | 3.245 | 0.462 | 4.0 | −0.755 |
| 4 | 5 | 7.577 | 0.500 | 8.0 | −0.423 |
| 4 | 6 | 11.424 | 0.527 | 12.0 | −0.576 |
| 4 | 7 | 15.473 | 0.578 | 16.0 | −0.527 |
| 4 | 8 | 19.223 | 0.580 | 20.0 | −0.777 |
| 4 | 9 | 23.879 | 0.518 | 24.0 | −0.121 |
| 7 | 7 | 1.432 | 0.572 | 1.5 | −0.068 |
| 7 | 8 | 2.952 | 0.528 | 3.0 | −0.048 |
| 7 | 9 | 4.642 | 0.536 | 4.5 | +0.142 |

Per cohort:

| cohort | CS estimate | planted (same quarters) |
|---|---|---|
| 4 | **+13.470** (se 0.396) | +14.000 |
| 7 | **+3.009** (se 0.447) | +3.000 |

Cohort 7 is recovered to three decimal places.

**What `control_group="not_yet_treated"` changes mechanically:** for each
$\text{ATT}(g,t)$, the control pool is restricted to units whose adoption quarter
is later than $t$. Already-treated units are never used. The forbidden
comparisons are not down-weighted — they are **never formed**.

## Stage 4 · The event study

| event time | n cohorts | estimate | se | planted |
|---|---|---|---|---|
| −8 | 1 | +0.666 | 0.465 | 0 |
| −7 | 1 | −1.181 | 0.481 | 0 |
| −6 | 1 | +0.777 | 0.515 | 0 |
| −5 | 2 | −0.466 | 0.357 | 0 |
| −4 | 2 | +0.153 | 0.351 | 0 |
| −3 | 1 | −0.777 | 0.515 | 0 |
| −2 | 2 | +0.140 | 0.352 | 0 |
| −1 | 2 | +0.421 | 0.345 | 0 |
| **0** | 2 | **+2.339** | 0.368 | +2.75 |
| 1 | 2 | +5.265 | 0.364 | +5.50 |
| 2 | 2 | +8.033 | 0.376 | +8.25 |
| 3 | 1 | +15.473 | 0.578 | +16.00 |
| 4 | 1 | +19.223 | 0.580 | +20.00 |
| 5 | 1 | +23.879 | 0.518 | +24.00 |

Pre-treatment: **12 placebo cells, mean −0.0016**, largest absolute estimate
1.1807.

**And one of the twelve is individually significant at 5%.** That is the point of
step 4. At a 5% level on twelve tests you *expect* about 0.6 false rejections, so
one is exactly what a design with genuinely parallel trends looks like — and
parallel trends does hold here, by construction.

> **Pre-trend tests are a multiple-testing problem.** Reading them one star at a
> time will make you reject good designs; and because they are typically
> under-powered, it will also let you accept bad ones. A joint test, or a
> sensitivity bound, is the better instrument.

## Stage 5 · What you cannot know

**Nine post-treatment cells come back NA**: everything from quarter 10 onwards,
and cohort 10 entirely.

The reason is countable: **in quarter 11, zero cafés are untreated.** By quarter
10 every café in the sample has a cat, so there is no not-yet-treated unit left
and nothing identifies those effects.

TWFE reported a number for exactly those quarters. It got it by using
already-treated cafés as controls — which is where a large part of the −2.09 came
from.

> **An estimator that returns NA is doing you a favour.** The nine NAs are not a
> shortcoming of Callaway–Sant'Anna; they are a correct description of what this
> research design can support. A method that hands you three stars for a quantity
> your data cannot identify is worse than one that refuses.

### The breakdown value

A differential trend of $\delta$ per quarter contaminates $\text{ATT}(g,t)$ by
roughly $\delta \times (t-g+1)$. Solving for the $\delta$ that drives each cohort
estimate to zero:

| cohort | ATT | mean exposure | **breakdown $\delta$** |
|---|---|---|---|
| 4 | +13.470 | 3.5 quarters | **+3.849 min/quarter** |
| 7 | +3.009 | 2.0 quarters | **+1.504 min/quarter** |

The pre-treatment trend we can actually measure is **+0.0443 min/quarter** —
about **87 times smaller** than the cohort-4 breakdown value.

The sentence to write:

> *"To explain away the cohort-4 effect, the treated cafés would have to have been
> on a differential trend of 3.85 minutes per quarter. The trend measurable before
> adoption is 0.04 minutes per quarter, roughly ninety times smaller. A violation
> of the required size would have been unmissable in the pre-period."*

That is a far stronger claim than "pre-trends look parallel", and it is the form
the current literature expects.

### The design recommendation

**Keep a control group untreated.** If every unit is treated by the end of your
panel, you have designed a study that cannot evaluate its own last phase — and no
estimator invented later will fix it. Holding back even 20 of the 186 cafés would
have identified every cell. This is a decision made before any data exists, which
is why the design section of a paper matters more than its estimator.

## The optional extras

`dd.SunAbraham`, `dd.ImputationDiD` (Borusyak–Jaravel–Spiess) and
`dd.TwoStageDiD` (Gardner) should broadly agree with Callaway–Sant'Anna, because
they solve the same problem by different routes. Reporting two estimators rather
than one is now standard practice.

`control_group="never_treated"` fails or returns almost nothing: there are no
never-treated cafés. Discovering that is instructive.

Adding `covariates=["seats", "wifi_mbps"]` should change little — they are
time-invariant and absorbed by the design — and confirming that a covariate does
nothing is a useful check that you understand what is being differenced away.

## What full marks look like

- The design stated first: three cohorts, no never-treated group
- Both numbers: −2.09 and +10.06, with the reason for the gap
- The **50.3% forbidden weight**, and their **−12.56** average
- The mechanism written out properly: the control group's own effect is being
  subtracted
- Group-time estimates checked against the planted truth
- The event study, with the pre-period read as **twelve tests, one significant**
- The nine NA cells explained by counting untreated cafés in quarter 11
- A breakdown value with the pre-trend comparison
- A design recommendation

## Common wrong answers

| Reported | What happened |
|---|---|
| "cats reduce dwell time by 2 minutes ($p<0.001$)" | reported TWFE as the answer |
| "+10.06" with no explanation of −2.09 | got the right answer, learnt nothing |
| "parallel trends holds" | it is untestable; bound it instead |
| "the pre-trends fail, one period is significant" | read twelve tests one at a time |
| "the NAs are a bug in the package" | they are the most informative output |
| "+11.92, matching the planted mean" | quoted the answer key, not an estimate |
