# Session 3 · Logistic Regression

**The model gives you a probability. The decision is still yours.**

Binary and ordinal outcomes, odds ratios, and the part that most treatments skip:
that turning a probability into an action requires a threshold, and the threshold
belongs to the costs rather than to the model.

## The puzzle

**The Mushroom Forager.** 1,400 mushrooms, nine features, and a verdict. Hand in
a decision rule you would actually eat from.

The model is easy and the decision is not. A logistic regression reaches
AUC 0.938 and finds — without being told — that the single strongest predictor is
`volva_present`, the sac at the stem base that marks *Amanita*: an odds ratio of
73.

Then the trap. On data that is 71.3% edible:

| rule | accuracy | poisonous mushrooms eaten |
|---|---|---|
| "always edible" | 71.3% | **all 402** |
| threshold 0.5 | **87.7%** | **105 — one meal in four** |
| cost-optimal, 50:1 | 58.5% | **1 in 402** |

The rule with the best accuracy poisons you regularly. The rule you would carry
into the woods has *worse* accuracy and one eighth the expected loss. Once you
write the two costs down, the threshold follows —
$t^\star = c_{FP}/(c_{FP}+c_{FN})$ — and it is nowhere near 0.5.

## What you should be able to do afterwards

- Fit a logistic regression and read a coefficient as an odds ratio
- Explain why an odds ratio is not a probability ratio
- Build a confusion matrix and quote accuracy against the majority-class baseline
- Choose a threshold from costs, and say which error you chose to make more of
- Run a likelihood-ratio test for a group of coefficients
- Fit an ordinal model, and check the proportional-odds assumption it rests on

## The written treatment

[*Quantitative Methods in International Business with Python*](https://warint.github.io/quantitative-methods/book/) (Warin),
chapter [4 · Logistic Regression: Binary, Ordinal, and Multinomial](https://warint.github.io/quantitative-methods/book/session-04.html).
Assigned in the pre-session; [`chapter.md`](chapter.md) follows its argument on
this session's data.

## Contents

| | |
|---|---|
| [`00-pre-session/`](00-pre-session/) | Odds, log-odds, the logistic function. **45 min.** |
| [`01-lecture/`](01-lecture/) | The deck. Executes its own Python. |
| [`chapter.md`](chapter.md) | The lecture notes, as prose. |
| [`02-practice/`](02-practice/) | The workshop, a starter script, and `SOLUTION.md`. |

## Data

```python
import essentials as ess
d = ess.load("mushrooms")     # 1,400 rows; `poisonous` binary, `toxicity` ordinal
```
