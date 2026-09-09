# Session 1 · The Roller Coaster Problem — answer key

*Read this after you have tried. Every number below was produced by
`scripts/verify_puzzles.py`, which asserts them on every run.*

## The planted answer

The relationship is **energy conservation**. A car that falls a height $h$
converts potential energy to kinetic energy:

$$ mgh = \tfrac{1}{2}mv^2 \quad\Longrightarrow\quad v = \sqrt{2gh} $$

So $v = c\,h^{k}$ with

| | |
|---|---|
| **exponent $k$** | **exactly 0.5** — speed goes as the square root of the drop |
| **constant $c$** | about **15.2** with $h$ in metres and $v$ in km/h |

The frictionless constant is $3.6\sqrt{2g} = 15.95$. The generator gives each
coaster an energy efficiency drawn uniformly on $[0.86, 0.94]$, so the fitted
constant lands near $15.95\sqrt{0.90} \approx 15.1$. **Friction costs speed; it
does not change the exponent.** A group that reports $k \approx 0.5$ and a
constant a little below 15.95, and explains the gap that way, has the whole
thing.

## What the stages should have produced

**Stage 1.** On all 84 rows, `speed_kmh ~ drop_m` gives $R^2 = 0.4595$ —
mediocre, and uninformative about *why*. On the 78 gravity coasters alone it
gives $R^2 = 0.9635$, which looks excellent and is still the wrong shape. Either
starting point is fine; what matters is that $R^2$ never diagnosed anything.

**Stage 2.** Residuals versus fitted show an **arch**: negative at small drops,
positive in the middle, negative again at large drops. Binned mean residuals on
the gravity subset:

| drop band (m) | 0–25 | 25–45 | 45–65 | 65–85 | 85–105 | 105+ |
|---|---|---|---|---|---|---|
| mean residual (km/h) | −7.33 | +1.45 | +3.82 | +6.44 | +0.61 | −4.52 |

The sign changes twice. That is a curved relationship being fitted by a straight
line. Adding `I(drop_m**2)` gives $t = -8.57$, $p \approx 10^{-12}$ — the
curvature is not a matter of taste.

**Stage 3.** Regressing $\log v$ on $\log h$ over the 78 gravity coasters:

```
exponent  k = 0.4971    95% CI [0.4835, 0.5106]
constant  c = 15.17
R2 = 0.9837
```

The interval contains 0.5 and excludes 1 comfortably. The sqrt form also beats
the quadratic ($R^2$ 0.9837 against 0.9816) on one fewer parameter, which is the
tidier argument.

**Stage 4.** Leverages sum to exactly $p = 2$ — the trace of the hat matrix,
always, with no tolerance. The six largest Cook's distances are dominated by the
**launched** coasters: 5 of the top 6, and all 6 sit in the top 8.

Reading those rows is the point. They are propelled by hydraulics or linear
induction motors rather than dropped, so they reach 150–240 km/h from
structures 12–46 m high. **Energy conservation does not constrain them** — a
motor put the energy in. They are not bad measurements; they are a different
physical system.

Keeping them costs:

| | exponent |
|---|---|
| gravity only, $n = 78$ | **0.4971** |
| all 84 coasters | 0.3722 |

Six rows — 7% of the data — move the exponent from a law of physics to a number
that means nothing, with a narrow interval around it.

## The distinction being taught

The right sequence is **diagnostic → rows → mechanism → decision**. Cook's
distance did not tell anyone to delete anything; it said *go and look at these
six*. The `propulsion` column then explained them, and the physics justified
excluding them.

Full marks for excluding them **with that argument**. Full marks, too, for
keeping them in and fitting `propulsion` as a separate group, which is arguably
better practice — it uses all the data and estimates the difference rather than
assuming it. **No marks** for deleting them because they were far from the line.

## Distractor columns

`queue_min` is pure noise. `length_m` and `inversions` carry a little real
signal because both were generated from `drop_m`. Anyone who added `queue_min`
and watched $R^2$ rise has met the reason adjusted $R^2$ exists — and the reason
it is not much of a defence either.

## Common wrong answers

| Reported | What happened |
|---|---|
| $k \approx 0.37$ | kept the launched coasters |
| $k \approx 1$ | fitted the raw units and reported the slope |
| "$R^2 = 0.96$, model is good" | never plotted the residuals |
| $k = 0.5$ with no interval | got the right answer without evidence for it |
