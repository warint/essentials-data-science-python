# Session 3 · The Mushroom Forager — answer key

*Read this after you have tried. Every number is asserted by
`scripts/verify_puzzles.py`.*

## The planted model

Log-odds of poisonous, with `spore_print = brown`, `gill_spacing = close` and
`habitat = grassland` as reference levels:

| term | planted | recovered |
|---|---|---|
| intercept | −9.00 | −9.24 |
| `volva_present` | **+4.20** | **+4.29** |
| `spore_print = white` | +3.20 | +3.40 |
| `ring_present` | +2.10 | +2.01 |
| `spore_print = pink` | +1.50 | +1.81 |
| `gill_spacing = crowded` | +1.30 | +1.18 |
| `habitat = woodland` | +1.10 | +0.97 |
| `habitat = mulch` | +0.65 | +0.68 |
| `odor_index` | +0.60 | +0.63 |
| `stem_height_cm` | +0.16 | +0.15 |
| `cap_diameter_cm` | −0.08 | −0.06 |
| `gill_spacing = distant` | −0.80 | −0.60 |
| `spore_print = black` | −1.90 | −1.79 |

Estimated on 1,400 rows, so a coefficient landing 0.3 from its planted value is
sampling error rather than a problem. The ordering is recovered exactly.

Base rate **28.7% poisonous**. AUC **0.9382**.

## The dominant feature

`volva_present`: coefficient **+4.29**, odds ratio **×73**. A volva is the
membranous sac at the base of the stem that marks *Amanita*, the genus
responsible for most fatal mushroom poisonings. 61.1% of mushrooms with one are
poisonous against 21.6% without.

The follow-up question matters more than the number. An odds ratio of 73 does
**not** multiply a probability by 73:

| starting $p$ | after applying OR = 73 |
|---|---|
| 0.01 | 0.4245 |
| 0.10 | 0.8903 |
| 0.50 | 0.9865 |
| 0.90 | 0.9985 |

Same odds ratio, four completely different probability changes. Odds ratios are
constant across the range; probability effects are not.

## The threshold — the whole point of the session

The baseline: **"always edible" is 71.3% accurate** and eats all 402 poisonous
mushrooms.

With $c_{FN} = 50$ (hospital visit) and $c_{FP} = 1$ (discarded mushroom),
expected loss is minimised near

$$ t^\star = \frac{c_{FP}}{c_{FP} + c_{FN}} = \frac{1}{51} = 0.0196 .$$

| threshold | accuracy | poisonous missed | basket discarded | expected loss |
|---|---|---|---|---|
| always edible | 0.713 | **402 of 402** | 0% | 20,100 |
| $t = 0.50$ | **0.877** | **105 (26.1%)** | 26% | 5,317 |
| $t = 0.10$ | 0.764 | 17 (4.2%) | 50% | 1,164 |
| $t = 0.0196$ | 0.585 | **1 (0.25%)** | 70% | **630** |
| $t = 0.005$ | 0.449 | 0 | 84% | 772 |

**The accuracy-maximising rule misses one poisonous mushroom in four.** The
cost-optimal rule has *worse* accuracy — 0.585 against 0.877 — and cuts expected
loss by a factor of eight.

The empirical minimum is at $t \approx 0.031$ rather than exactly 0.0196, and it
should be: with 1,400 discrete cases the loss is a step function, so its minimum
is a flat region. Everything from 0.021 to 0.031 is within 5% of the floor, and
the loss at the theoretical value is within 3% of the best achievable. **The
formula names the right neighbourhood, not a value to quote to four decimals.**

Full marks for any defensible pair of costs, provided the threshold follows from
them and the write-up says which error was chosen deliberately. A group that
argued for $c_{FN}/c_{FP} = 500$ and discarded 84% of the basket has done the
exercise correctly.

## The ordinal part

`toxicity` splits as: edible 998, upset stomach 267, hospital 71, fatal 64.

The proportional-odds model claims a volva has the *same* effect on the odds of
"at least upset stomach" as on the odds of "fatal". In this generator that is
true by construction — the grades are cuts through one latent risk — so the
separate binary logits should give similar coefficients, and finding that they do
is the correct answer.

The lesson is the check, not the result. In real graded outcomes the assumption
frequently fails, and when it does the model reports an average of two different
effects that is true of neither.

## What full marks look like

- A threshold with **stated costs**, and the acknowledgement that it scores worse
  on accuracy.
- Accuracy always reported next to the 71.3% baseline.
- The confusion matrix at both thresholds, with the four cells named in words.
- "My rule lets through 0.25 poisonous mushrooms per hundred; the accurate rule
  lets through 26."
- The volva identified, with its odds ratio, and the observation that it is not a
  probability ratio.

## Common wrong answers

| Reported | What happened |
|---|---|
| "88% accurate — good model" | never compared with the 71.3% baseline |
| $t = 0.5$, unremarked | treated a default as a fact |
| "a volva makes it 73× more likely to be poisonous" | odds ratio read as a probability ratio |
| "AUC 0.94, so the rule is safe" | AUC is the whole curve; a decision is one point on it |
| a threshold with no costs | the arithmetic has nothing to minimise |
