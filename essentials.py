"""One call to load any dataset used in this course.

    import essentials as ess

    coasters = ess.load("coasters")     # session 1
    ess.catalog()                       # what else is there

Why this exists
---------------
Every dataset in this course is *generated*, from a fixed seed, by the code in
this file. Nothing is downloaded. That is a deliberate choice with three
consequences worth understanding, because they are the reason the practice
sessions work the way they do.

**It runs with the wifi off.** A ninety-minute workshop that begins with two
hundred people hitting the same server begins badly. `load()` builds the frame
locally in well under a second and caches it as parquet, so the second call is
instant and the room's network is never in the critical path.

**Everyone has the same bytes.** The seeds are fixed, so your Cook's distance
and your neighbour's Cook's distance are the same number. When two groups
disagree about a result, the disagreement is about the analysis, which is the
only kind worth having in class.

**There is a right answer.** This is the important one. Each dataset has a
*planted* truth — a physical constant, an optimal recipe, a set of five
ingredients that matter, a treatment effect. You are not being asked to produce
a defensible-sounding number. You are being asked to recover a specific one,
and you can be scored against it. Every dataset also contains a **trap**: a
naive method that looks like it worked, reports a healthy fit statistic, and
gets the wrong answer. Finding the trap is the exercise.

The planted answers are in each session's `02-practice/SOLUTION.md`. Read them
after you have tried, not before — a puzzle you have been told the answer to
teaches nothing, and these are built to be solvable.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

__all__ = ["load", "catalog", "generate", "path", "ROOT", "DATA"]

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"

# Physical constant, used by the session 1 generator and quoted in its solution.
GRAVITY = 9.81  # m/s^2


# --------------------------------------------------------------------------
# Session 1 · The Roller Coaster Problem
# --------------------------------------------------------------------------
def _make_coasters(seed: int = 20260101) -> pd.DataFrame:
    """84 roller coasters. A law of physics is hiding in two of the columns.

    Energy conservation: a car that falls a height h arrives at the bottom with
    speed v = sqrt(2*g*h), so v is proportional to the SQUARE ROOT of the drop.
    On a log-log scale that is a straight line of slope exactly 0.5, and that
    exponent is the planted answer.

    The trap: a straight line in the raw units fits with R^2 near 0.93 and
    looks entirely respectable. Its residuals are curved, because a square root
    is not a straight line, and a student who reports the slope of that line
    has reported a number that means nothing.

    The influential points: six coasters are launched by hydraulic or magnetic
    propulsion rather than dropped. Energy conservation does not constrain them
    — they reach high speed from a low structure — so they sit far off the law
    and carry most of the influence. Cook's distance finds them; the
    `propulsion` column explains them. That is the intended sequence: a
    diagnostic points at rows, and going to look at the rows reveals a
    subgroup the model should never have been fitted on.
    """
    rng = np.random.default_rng(seed)
    n_gravity, n_launch = 78, 6

    # Gravity coasters: drop heights spread over the range actually built.
    drop = np.concatenate([
        rng.uniform(8, 40, 26),      # family coasters
        rng.uniform(40, 90, 34),     # the bulk of the steel fleet
        rng.uniform(90, 140, 18),    # hypercoasters
    ])
    # About 90% of the potential energy survives friction, so the constant sits
    # a little below the frictionless 3.6*sqrt(2g) = 15.94. The EXPONENT is
    # untouched: friction costs speed, it does not change the power law.
    efficiency = rng.uniform(0.86, 0.94, n_gravity)
    speed = 3.6 * np.sqrt(2 * GRAVITY * efficiency * drop)
    speed *= np.exp(rng.normal(0, 0.035, n_gravity))   # measurement noise, in logs

    # Launched coasters: small structure, very high speed. Off the law entirely.
    l_drop = rng.uniform(12, 46, n_launch)
    l_speed = rng.uniform(150, 240, n_launch)

    drop = np.concatenate([drop, l_drop])
    speed = np.concatenate([speed, l_speed])
    propulsion = np.array(["gravity"] * n_gravity + ["launch"] * n_launch)

    # Plausible-looking companions. `length_m` and `inversions` carry a little
    # real signal; `queue_min` is noise, and is here so that somebody discovers
    # that adding a column can raise R^2 while explaining nothing.
    length = 220 + 11.5 * drop + rng.normal(0, 190, len(drop))
    inversions = rng.poisson(np.clip(drop / 45, 0.2, 3.2))
    queue = rng.integers(5, 95, len(drop))

    adjectives = ["Iron", "Storm", "Comet", "Vortex", "Banshee", "Cobra", "Titan",
                  "Phoenix", "Kraken", "Zephyr", "Hydra", "Falcon", "Nomad",
                  "Basilisk", "Wyvern", "Tempest", "Onyx", "Grizzly", "Mantis",
                  "Serpent", "Rampart", "Halcyon"]
    nouns = ["Run", "Drop", "Chase", "Dive", "Rush", "Coil", "Plunge", "Loop"]
    names = [f"{adjectives[i % len(adjectives)]} {nouns[(i * 3) % len(nouns)]}"
             for i in range(len(drop))]

    df = pd.DataFrame({
        "coaster": names,
        "park": [f"Park {chr(65 + i % 12)}" for i in range(len(drop))],
        "year_opened": rng.integers(1979, 2024, len(drop)),
        "drop_m": np.round(drop, 1),
        "speed_kmh": np.round(speed, 1),
        "length_m": np.round(np.clip(length, 120, None), 0),
        "inversions": inversions,
        "queue_min": queue,
        "propulsion": propulsion,
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------
# Session 2 · The Sourdough Optimum
# --------------------------------------------------------------------------
# The response surface, stated once so the generator and the solution cannot
# drift apart. u = hydration - 75, v = oven_c - 225.
#
#   rise = 40 + 1.10u - 0.055u^2 + 0.30v - 0.020v^2 + 0.015uv + baker + e
#
# The Hessian is negative definite, so this is a genuine peak rather than a
# saddle. Setting both partial derivatives to zero gives the planted optimum:
#   hydration* = 86.6 %, oven* = 236.9 C
#
# The interaction is what makes the answer a *surface* rather than two separate
# recommendations: the best oven temperature is 228.8 C for a 65% dough and
# 238.1 C for a 90% one. An additive model gives the same temperature for both,
# which is the specific thing it gets wrong.
_SD = dict(b0=40.0, bh=1.10, bhh=-0.055, bt=0.30, btt=-0.020, bht=0.015)


def _make_sourdough(seed: int = 20260202) -> pd.DataFrame:
    """240 loaves, 8 bakers. Find the recipe that maximises rise.

    The planted optimum is a hydration and an oven temperature that can only be
    recovered by fitting BOTH quadratic terms AND the interaction: the
    interaction tilts the surface, so the best temperature depends on the
    hydration you chose. Fit the two variables additively and linearly and you
    get a surface with no peak at all, and a confident answer at the edge of
    the design.

    The trap is the bakers. Each has a house style — one works dry, one works
    wet — so hydration is confounded with baker skill, and the eight of them
    differ by more than the noise. Omit the baker fixed effects and the
    hydration coefficient absorbs the skill difference, which moves the
    estimated optimum by several points of hydration. This is what fixed
    effects are FOR, on data small enough to see it happen.
    """
    rng = np.random.default_rng(seed)
    bakers = list("ABCDEFGH")
    n_each = 30

    # House styles: each baker's preferred hydration band. Deliberately
    # confounded with skill below.
    centre = dict(zip(bakers, [62, 66, 71, 75, 78, 82, 86, 90], strict=True))
    # Skill runs the OTHER way to the wet-dough preference, so a model without
    # baker effects reads "wetter is worse" and shifts the optimum down.
    skill = dict(zip(bakers, [9.0, 6.5, 4.0, 1.5, -1.0, -3.5, -6.0, -8.5],
                        strict=True))

    rows = []
    for b in bakers:
        h = np.clip(rng.normal(centre[b], 5.5, n_each), 55, 95)
        t = rng.uniform(190, 260, n_each)
        u, v = h - 75, t - 225
        rise = (_SD["b0"] + _SD["bh"] * u + _SD["bhh"] * u**2
                + _SD["bt"] * v + _SD["btt"] * v**2
                + _SD["bht"] * u * v
                + skill[b] + rng.normal(0, 3.0, n_each))
        rows.append(pd.DataFrame({
            "baker": b,
            "hydration_pct": np.round(h, 1),
            "oven_c": np.round(t, 0),
            "starter_age_days": rng.integers(2, 15, n_each),
            "flour": rng.choice(["white", "wholemeal", "spelt"], n_each,
                                p=[0.6, 0.25, 0.15]),
            "rise_mm": np.round(rise, 1),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["loaf_id"] = np.arange(1, len(df) + 1)
    cols = ["loaf_id", "baker", "hydration_pct", "oven_c", "starter_age_days",
            "flour", "rise_mm"]
    return df[cols].sample(frac=1, random_state=seed).reset_index(drop=True)


def sourdough_optimum() -> tuple[float, float]:
    """The planted optimum, solved from the coefficients above rather than typed.

    Kept here so the lecture, the generator and the solution quote one number.
    """
    # Gradient of the quadratic surface is linear in (u, v); solve the 2x2 system.
    a = np.array([[2 * _SD["bhh"], _SD["bht"]],
                  [_SD["bht"], 2 * _SD["btt"]]])
    b = np.array([-_SD["bh"], -_SD["bt"]])
    u, v = np.linalg.solve(a, b)
    return 75 + u, 225 + v


# --------------------------------------------------------------------------
# Session 3 · The Mushroom Forager
# --------------------------------------------------------------------------
_MUSH = dict(
    intercept=-9.00, cap_diameter_cm=-0.08, stem_height_cm=0.16,
    odor_index=0.60, ring_present=2.10, volva_present=4.20,
    spore_white=3.20, spore_pink=1.50, spore_black=-1.90,   # brown = reference
    gill_crowded=1.30, gill_distant=-0.80,                  # close = reference
    hab_woodland=1.10, hab_mulch=0.65,                      # grassland = reference
)


def _make_mushrooms(seed: int = 20260303) -> pd.DataFrame:
    """1,400 mushrooms from one forest. Which ones can you eat?

    A logistic regression recovers the planted coefficients, and one of them is
    much larger than the rest: `volva_present`, the membranous sac at the base
    of the stem that marks the genus responsible for most fatal poisonings. Part
    of the puzzle is discovering that a single feature carries that much of the
    signal.

    The trap is the threshold, and it is not a modelling error — it is a
    decision error. Accuracy is maximised near p = 0.5, and a forager who uses
    0.5 will eat a poisonous mushroom roughly once every forty meals. The costs
    are not symmetric: a false negative is a hospital visit, a false positive is
    a discarded mushroom. Once you write the two costs down, the threshold that
    minimises expected loss is c_FP / (c_FP + c_FN), which is nowhere near 0.5,
    and the model that "scores worse" on accuracy is the one you would actually
    carry into the woods.

    `toxicity` is the same latent risk cut into four ordered grades, for the
    ordinal part of the session.
    """
    rng = np.random.default_rng(seed)
    n = 1400

    cap = np.round(rng.gamma(6.0, 1.5, n) + 2, 1)
    stem = np.round(rng.gamma(5.0, 1.6, n) + 1, 1)
    odor = np.round(rng.uniform(0, 10, n), 2)
    ring = rng.binomial(1, 0.34, n)
    volva = rng.binomial(1, 0.18, n)
    spore = rng.choice(["brown", "white", "pink", "black"], n,
                       p=[0.40, 0.28, 0.17, 0.15])
    gill = rng.choice(["close", "crowded", "distant"], n, p=[0.45, 0.33, 0.22])
    hab = rng.choice(["grassland", "woodland", "mulch"], n, p=[0.34, 0.46, 0.20])

    eta = (_MUSH["intercept"]
           + _MUSH["cap_diameter_cm"] * cap
           + _MUSH["stem_height_cm"] * stem
           + _MUSH["odor_index"] * odor
           + _MUSH["ring_present"] * ring
           + _MUSH["volva_present"] * volva
           + _MUSH["spore_white"] * (spore == "white")
           + _MUSH["spore_pink"] * (spore == "pink")
           + _MUSH["spore_black"] * (spore == "black")
           + _MUSH["gill_crowded"] * (gill == "crowded")
           + _MUSH["gill_distant"] * (gill == "distant")
           + _MUSH["hab_woodland"] * (hab == "woodland")
           + _MUSH["hab_mulch"] * (hab == "mulch"))

    p = 1 / (1 + np.exp(-eta))
    poisonous = rng.binomial(1, p, n)

    # Ordinal grade: the same latent risk, cut. Only genuinely poisonous
    # mushrooms can land above grade 0.
    cuts = [1.2, 2.6, 4.2]
    latent = eta + rng.logistic(0, 1, n)
    grade = np.digitize(latent, cuts)
    grade = np.where(poisonous == 0, 0, np.clip(grade, 1, 3))
    labels = np.array(["edible", "upset stomach", "hospital", "fatal"])

    return pd.DataFrame({
        "specimen": np.arange(1, n + 1),
        "cap_diameter_cm": cap,
        "stem_height_cm": stem,
        "odor_index": odor,
        "spore_print": spore,
        "gill_spacing": gill,
        "habitat": hab,
        "ring_present": ring,
        "volva_present": volva,
        "poisonous": poisonous,
        "toxicity": pd.Categorical(labels[grade], categories=list(labels),
                                   ordered=True),
    })


# --------------------------------------------------------------------------
# Session 4 · The Secret Recipe
# --------------------------------------------------------------------------
_RECIPE_TRUTH = {
    "vanilla_extract": 4.0,
    "brown_sugar": 2.5,
    "sea_salt": -3.2,
    "browned_butter": 1.8,
    "dutch_cocoa": 1.1,
    "baking_soda": 1.5,     # correlated pair, r = 0.95, equal effects
    "baking_powder": 1.5,
}


def _make_secret_recipe(seed: int = 20260404) -> pd.DataFrame:
    """60 bakes, 200 candidate ingredients, 7 that matter. Find the recipe.

    p is more than three times n, so ordinary least squares has no unique
    solution at all — there is no "just fit it and look at the p-values" here,
    which is the point of opening the session with this shape of data.

    The lasso recovers the sparse truth. Ridge does not: it shrinks all 200
    towards zero and sets none of them there, so it predicts respectably and
    names nothing.

    The planted subtlety is the leaveners. `baking_soda` and `baking_powder`
    are correlated above 0.95 and have IDENTICAL true effects. The lasso cannot
    keep both — the geometry of its penalty forces a choice — so it zeroes one
    outright and hands the survivor most of the pair's combined effect. It does
    this in every cross-validation split, with no sign of distress, which is
    worse than doing it at random: nothing in the output tells you a second
    ingredient is sitting behind the one you were given.

    The elastic net retains both, because the ridge half of its penalty has no
    reason to prefer either. But it does not identify the SPLIT: change the
    cross-validation seed and the two coefficients trade places while their sum
    stays roughly constant. Two collinear ingredients carry one identifiable
    quantity between them, and no penalty can conjure a second. Knowing which of
    those two facts you have is the point.

    Two further lessons come free, and both are about honesty.

    At the cross-validated penalty the lasso selects around 25 ingredients, not
    7. Cross-validation tunes for PREDICTION, and the penalty that predicts best
    is smaller than the penalty that selects best. A lasso's non-zero set is not
    a list of true effects, and reporting it as one is the commonest misuse of
    the method.

    And `dutch_cocoa`, whose true effect of 1.1 is the weakest planted signal, is
    essentially lost by every estimator here — it comes back somewhere between
    0.00 and 0.25 depending on the split. At n = 60 it sits below the detection
    floor. No amount of method choice recovers a signal the sample size cannot
    support, and a method that appears to has usually found noise.
    """
    rng = np.random.default_rng(seed)
    n, p = 60, 200

    base = ["flour", "butter", "granulated_sugar", "brown_sugar", "egg_yolk",
            "egg_white", "vanilla_extract", "almond_extract", "sea_salt",
            "baking_soda", "baking_powder", "dutch_cocoa", "browned_butter",
            "buttermilk", "sour_cream", "honey", "molasses", "maple_syrup",
            "cinnamon", "nutmeg", "cardamom", "cloves", "ginger", "allspice",
            "orange_zest", "lemon_zest", "espresso_powder", "malted_milk",
            "cream_cheese", "mascarpone", "creme_fraiche", "yoghurt",
            "dark_chocolate", "milk_chocolate", "white_chocolate", "cacao_nibs",
            "hazelnut", "pecan", "walnut", "pistachio", "macadamia", "almond_flour",
            "oat_flour", "rye_flour", "semolina", "cornstarch", "tapioca",
            "xanthan_gum", "gelatin", "agar"]
    names = list(base)
    suffix = ["_toasted", "_ground", "_flaked", "_syrup", "_powder", "_paste",
              "_infused", "_smoked"]
    i = 0
    while len(names) < p:
        names.append(f"{base[i % len(base)]}{suffix[(i // len(base)) % len(suffix)]}")
        i += 1
    names = names[:p]

    z = rng.standard_normal((n, p))
    # Force the leavener pair to correlate at 0.95.
    i_soda, i_powder = names.index("baking_soda"), names.index("baking_powder")
    rho = 0.97
    z[:, i_powder] = rho * z[:, i_soda] + np.sqrt(1 - rho**2) * z[:, i_powder]

    grams = np.round(10 + 3 * z, 2)          # every ingredient in grams
    y = 50.0 + sum(_RECIPE_TRUTH[k] * z[:, names.index(k)] for k in _RECIPE_TRUTH)
    y = y + rng.normal(0, 2.5, n)

    df = pd.DataFrame(grams, columns=names)
    df.insert(0, "bake_id", np.arange(1, n + 1))
    df["taste_score"] = np.round(y, 2)
    return df


# --------------------------------------------------------------------------
# Session 5 · The Bestiary
# --------------------------------------------------------------------------
# Four species, each a 2-D Gaussian. Unequal covariances, so the optimal
# boundary is curved and no straight-line classifier can reach it.
_SPECIES = {
    "glimmerwing": dict(prior=0.34, mean=[6.5, 3.0],  cov=[[1.30, 0.35], [0.35, 0.55]]),
    "thornback":   dict(prior=0.34, mean=[10.5, 5.5], cov=[[2.60, -1.15], [-1.15, 1.40]]),
    "dusk_moth":   dict(prior=0.30, mean=[8.0, 8.5],  cov=[[0.70, 0.20], [0.20, 2.40]]),
    "ember_drake": dict(prior=0.02, mean=[13.5, 9.5], cov=[[0.90, 0.30], [0.30, 0.90]]),
}
_LABEL_NOISE = 0.07   # a recorded species is wrong this often: the Bayes floor


def _make_bestiary(seed: int = 20260505) -> pd.DataFrame:
    """1,600 creatures, four species, and a boundary you can check against.

    Because the generator is a mixture of known Gaussians, the Bayes-optimal
    decision boundary can be computed exactly, and so can the error rate that
    no classifier can beat. That turns the bias-variance trade-off from a
    picture in a textbook into something you can measure: overlay your k = 1
    boundary on the true one and it is shattered into islands; overlay k = 200
    and it is smooth and in the wrong place. Somewhere between them is a k that
    gets close to a floor you can name in advance.

    Two traps. `ember_drake` is 2% of the data, so a classifier that never
    predicts it is 98% accurate on that class and completely useless for the
    only species anyone cares about — the confusion matrix shows this and
    accuracy hides it. And `mass_g`, `antenna_mm` and `scale_count` are pure
    noise: they carry no information about species at all. Feed them to a
    k-nearest-neighbour classifier and watch the accuracy fall, because
    distance in an irrelevant dimension is still distance.
    """
    rng = np.random.default_rng(seed)
    n = 1600
    names = list(_SPECIES)
    priors = np.array([_SPECIES[s]["prior"] for s in names])
    priors = priors / priors.sum()

    true_idx = rng.choice(len(names), n, p=priors)
    xy = np.zeros((n, 2))
    for j, s in enumerate(names):
        m = true_idx == j
        xy[m] = rng.multivariate_normal(_SPECIES[s]["mean"], _SPECIES[s]["cov"],
                                        int(m.sum()))

    # Field identification is imperfect: some records carry the wrong species.
    # This is what puts a floor under every classifier's error rate.
    #
    # The confusion is only ever among the three COMMON species. An ember drake
    # is unmistakable once you have seen one, so its labels are clean — which
    # also means the 2% class stays a genuine 2% rather than being padded out by
    # other species' mistakes. A rare class whose labels were mostly wrong would
    # be unlearnable, and that is a different lesson from the one intended.
    common = [j for j, sp in enumerate(names) if sp != "ember_drake"]
    recorded = true_idx.copy()
    flip = (rng.random(n) < _LABEL_NOISE) & np.isin(true_idx, common)
    for i in np.flatnonzero(flip):
        recorded[i] = rng.choice([k for k in common if k != true_idx[i]])

    return pd.DataFrame({
        "creature_id": np.arange(1, n + 1),
        "wing_span_cm": np.round(xy[:, 0], 2),
        "tail_length_cm": np.round(xy[:, 1], 2),
        "mass_g": np.round(rng.gamma(4, 22, n), 1),        # noise
        "antenna_mm": np.round(rng.uniform(2, 19, n), 2),  # noise
        "scale_count": rng.integers(40, 260, n),           # noise
        "species": pd.Categorical([names[i] for i in recorded], categories=names),
    })


def bestiary_posterior(wing_span, tail_length):
    """The true posterior over species at any point — the Bayes rule itself.

    Returned as an (n, 4) array in the column order of `_SPECIES`. Take the
    argmax for the optimal prediction, or contour the difference between the top
    two for the optimal boundary. Use it to draw the truth underneath whatever
    your classifier learned.
    """
    from scipy.stats import multivariate_normal

    x = np.column_stack([np.ravel(wing_span), np.ravel(tail_length)])
    names = list(_SPECIES)
    dens = np.column_stack([
        _SPECIES[s]["prior"] * multivariate_normal(
            _SPECIES[s]["mean"], _SPECIES[s]["cov"]).pdf(x)
        for s in names
    ])
    # Label noise is symmetric across the other classes, so it mixes each
    # posterior towards the others without moving the argmax.
    k = len(names)
    dens = dens / dens.sum(axis=1, keepdims=True)
    return (1 - _LABEL_NOISE) * dens + _LABEL_NOISE * (1 - dens) / (k - 1)


def bestiary_species() -> list[str]:
    """Species names, in the column order `bestiary_posterior` returns."""
    return list(_SPECIES)


# --------------------------------------------------------------------------
# Session 6 · The Cat Cafe Effect
# --------------------------------------------------------------------------
_CAT_DELTA = {4: 4.0, 7: 1.5, 10: 0.5}   # effect per quarter since adoption
_CAT_T = 12


def _make_cat_cafe(seed: int = 20260606) -> pd.DataFrame:
    """186 cafes adopt a resident cat on different dates. Did it work?

    Every planted treatment effect in this dataset is POSITIVE and every one of
    them GROWS with time since the cat arrived: the true ATT for a cafe treated
    in quarter g, observed in quarter t, is delta_g * (t - g + 1). Parallel
    trends holds by construction. There is nothing subtle about the truth.

    The trap is the estimator everyone reaches for first. Two-way fixed effects
    with a single treatment dummy — unit effects, quarter effects, one
    coefficient — returns a NEGATIVE number on this data. Not a noisy zero: the
    wrong sign, with a small standard error and three stars.

    The reason is that with staggered adoption, two-way fixed effects is a
    weighted average of every available 2x2 comparison, and some of those
    comparisons use the ALREADY-TREATED early cohort as the control group for
    the later cohort. When effects grow over time, the early cohort's outcome is
    still rising while it serves as a control, so the comparison attributes that
    rise to the absence of treatment and books it as a negative effect. Some of
    the weights in that average are negative. Goodman-Bacon's decomposition
    shows you exactly which comparisons they are, and the group-time estimator
    of Callaway and Sant'Anna refuses to make them, which is why it recovers the
    planted effects and two-way fixed effects does not.

    This is the whole argument for the modern difference-in-differences
    literature, on data where you know what the answer was supposed to be.
    """
    rng = np.random.default_rng(seed)
    cohorts = [4, 7, 10]
    sizes = [64, 62, 60]

    rows = []
    cid = 0
    for g, k in zip(cohorts, sizes, strict=True):
        for _ in range(k):
            cid += 1
            unit_fe = rng.normal(42, 7.0)         # cafes differ a lot
            seats = int(rng.integers(14, 70))
            wifi = round(rng.uniform(15, 240), 1)
            city = rng.choice(["Northgate", "Riverside", "Old Mill", "Harbour"])
            for t in range(1, _CAT_T + 1):
                time_fe = 0.55 * t               # everyone's dwell time drifts up
                exposure = max(0, t - g + 1)
                effect = _CAT_DELTA[g] * exposure
                y = unit_fe + time_fe + effect + rng.normal(0, 2.2)
                rows.append((cid, city, seats, wifi, t, g, int(t >= g),
                             exposure, round(y, 2)))

    df = pd.DataFrame(rows, columns=[
        "cafe_id", "city", "seats", "wifi_mbps", "quarter", "cat_arrival_quarter",
        "has_cat", "quarters_with_cat", "dwell_min"])
    # `quarters_with_cat` is the honest exposure clock and is kept, because the
    # event study needs it. It is NOT the treatment dummy; `has_cat` is.
    return df


def cat_cafe_truth() -> pd.DataFrame:
    """Every planted ATT(g, t) — the answer key for session 6's event study."""
    rows = [
        {"cohort": g, "quarter": t, "quarters_with_cat": t - g + 1,
         "true_att": _CAT_DELTA[g] * (t - g + 1)}
        for g in _CAT_DELTA for t in range(g, _CAT_T + 1)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The loader
# --------------------------------------------------------------------------
_GENERATORS = {
    "coasters": _make_coasters,
    "sourdough": _make_sourdough,
    "mushrooms": _make_mushrooms,
    "secret_recipe": _make_secret_recipe,
    "bestiary": _make_bestiary,
    "cat_cafe": _make_cat_cafe,
}

_DESCRIPTIONS = {
    "coasters":      (1, "84 roller coasters. Recover the exponent of a physical law."),
    "sourdough":     (2, "240 loaves, 8 bakers. Find the recipe that maximises rise."),
    "mushrooms":     (3, "1,400 mushrooms. Decide which ones you would eat."),
    "secret_recipe": (4, "60 bakes, 200 ingredients, 7 that matter. Name them."),
    "bestiary":      (5, "1,600 creatures, 4 species, and a boundary you can check."),
    "cat_cafe":      (6, "186 cafes, staggered adoption. Did the cat work?"),
}


def path(name: str) -> pathlib.Path:
    """Where `name` caches. The file does not have to exist yet."""
    return DATA / f"{name}.parquet"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Give text columns plain `object` dtype, on every pandas version.

    pandas 3 stores strings as an Arrow-backed extension type by default. It is
    a better dtype in every way except one that matters here: `df[col].values`
    hands scikit-learn an Arrow array, which it cannot index, and the traceback
    is six frames deep in pyarrow and mentions nothing a student would
    recognise. Course data is small, so the memory that `object` costs is worth
    the afternoon it saves.
    """
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            cats = df[col].cat.categories
            if cats.dtype != object:
                df[col] = df[col].cat.rename_categories(cats.astype(object))
        elif pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
            df[col] = df[col].astype(object)
    return df


def generate(name: str) -> pd.DataFrame:
    """Build a dataset from its seed, ignoring any cache."""
    if name not in _GENERATORS:
        raise KeyError(f"unknown dataset {name!r}. Try essentials.catalog().")
    return _normalise(_GENERATORS[name]())


def load(name: str, *, refresh: bool = False) -> pd.DataFrame:
    """Return a course dataset, building and caching it on first call.

    Nothing is downloaded, so this works offline and returns the same bytes on
    every machine. Pass `refresh=True` to rebuild from the generator.
    """
    cache = path(name)
    if cache.exists() and not refresh:
        try:
            return _normalise(pd.read_parquet(cache))
        except Exception:
            pass                      # unreadable cache: fall through and rebuild
    df = generate(name)
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    except Exception:
        pass                          # no pyarrow, or a read-only clone: fine
    return df


def catalog() -> pd.DataFrame:
    """Every dataset, the session it belongs to, and what it asks of you."""
    return pd.DataFrame(
        [{"dataset": k, "session": v[0], "puzzle": v[1]}
         for k, v in sorted(_DESCRIPTIONS.items(), key=lambda kv: kv[1][0])]
    )
