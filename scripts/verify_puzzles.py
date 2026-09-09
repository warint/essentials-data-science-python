#!/usr/bin/env python3
"""Assert that every planted answer is still in the data.

    python scripts/verify_puzzles.py

Each dataset in this course contains a specific right answer and a specific
trap. Both are properties of the generator, and both are easy to destroy by
adjusting a coefficient — raise the noise a little and the lasso stops
recovering the recipe; add a never-treated group to the cafes and two-way fixed
effects stops getting the sign wrong, which is the entire point of session 6.

So the answers are tested. If you change `essentials.py`, run this. A session
whose puzzle no longer has its answer in it is worse than no session.

Every number asserted below was measured, not assumed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import essentials as ess  # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{PASS if condition else FAIL}] {label}" + (f"   {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


# --------------------------------------------------------------------------
def session_1_coasters() -> None:
    import statsmodels.formula.api as smf

    print("\nSession 1 · coasters — the exponent of a physical law")
    d = ess.load("coasters", refresh=True)
    g = d[d.propulsion == "gravity"].copy()
    g["lv"], g["lh"] = np.log(g.speed_kmh), np.log(g.drop_m)

    m = smf.ols("lv ~ lh", data=g).fit()
    lo, hi = m.conf_int().loc["lh"]
    check("log-log slope recovers 0.5", lo < 0.5 < hi,
          f"slope {m.params['lh']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")

    lin = smf.ols("speed_kmh ~ drop_m", data=g).fit()
    check("the trap looks respectable (linear R2 > 0.90)", lin.rsquared > 0.90,
          f"R2 = {lin.rsquared:.4f}")

    quad = smf.ols("speed_kmh ~ drop_m + I(drop_m**2)", data=g).fit()
    check("but its curvature is unmissable", quad.pvalues["I(drop_m ** 2)"] < 1e-6,
          f"quadratic term t = {quad.tvalues['I(drop_m ** 2)']:.2f}")

    a = d.copy()
    a["lv"], a["lh"] = np.log(a.speed_kmh), np.log(a.drop_m)
    ma = smf.ols("lv ~ lh", data=a).fit()
    check("launched coasters wreck the exponent if kept", ma.params["lh"] < 0.42,
          f"slope with all 84 = {ma.params['lh']:.4f}")

    cooks = ma.get_influence().cooks_distance[0]
    launched_in_top6 = (a.propulsion.values[np.argsort(cooks)[-6:]] == "launch").sum()
    check("Cook's distance points at them", launched_in_top6 >= 5,
          f"{launched_in_top6}/6 of the top-6 most influential are launched")


# --------------------------------------------------------------------------
def session_2_sourdough() -> None:
    import statsmodels.formula.api as smf
    from scipy.optimize import minimize

    print("\nSession 2 · sourdough — the recipe that maximises rise")
    d = ess.load("sourdough", refresh=True)
    h_true, t_true = ess.sourdough_optimum()

    def peak(fit: object) -> tuple[float, float]:
        def neg(z: np.ndarray) -> float:
            row = pd.DataFrame([{"hydration_pct": z[0], "oven_c": z[1], "baker": "D"}])
            return -fit.predict(row).iloc[0]
        return tuple(minimize(neg, [75, 225],
                              bounds=[(55, 95), (190, 260)]).x)

    full = ("rise_mm ~ hydration_pct + I(hydration_pct**2) + oven_c"
            " + I(oven_c**2) + hydration_pct:oven_c + C(baker)")
    no_fe = full.replace(" + C(baker)", "")

    fit = smf.ols(full, data=d).fit()
    h, t = peak(fit)
    check("full model recovers the planted optimum", abs(h - h_true) < 0.4,
          f"found {h:.2f}% / {t:.1f} C   planted {h_true:.2f}% / {t_true:.1f} C")

    h_bad, _ = peak(smf.ols(no_fe, data=d).fit())
    check("omitting baker effects moves it several points", abs(h_bad - h_true) > 2.5,
          f"without baker FE: {h_bad:.2f}%  (off by {h_bad - h_true:+.2f} pp)")
    check("and the broken model still looks fine",
          smf.ols(no_fe, data=d).fit().rsquared > 0.85,
          f"R2 without baker FE = {smf.ols(no_fe, data=d).fit().rsquared:.3f}")

    check("the interaction is real", abs(fit.tvalues["hydration_pct:oven_c"]) > 8,
          f"t = {fit.tvalues['hydration_pct:oven_c']:.2f}")


# --------------------------------------------------------------------------
def session_3_mushrooms() -> None:
    import statsmodels.formula.api as smf
    from sklearn.metrics import confusion_matrix, roc_auc_score

    print("\nSession 3 · mushrooms — the threshold is the decision")
    d = ess.load("mushrooms", refresh=True)
    check("base rate is roughly a third", 0.24 < d.poisonous.mean() < 0.35,
          f"poisonous = {d.poisonous.mean():.3f}")

    f = ("poisonous ~ cap_diameter_cm + stem_height_cm + odor_index + ring_present"
         " + volva_present + C(spore_print, Treatment('brown'))"
         " + C(gill_spacing, Treatment('close')) + C(habitat, Treatment('grassland'))")
    m = smf.logit(f, data=d).fit(disp=0)
    p = m.predict(d)

    check("the model discriminates well", roc_auc_score(d.poisonous, p) > 0.92,
          f"AUC = {roc_auc_score(d.poisonous, p):.4f}")
    check("volva is recovered as the dominant feature",
          abs(m.params["volva_present"] - ess._MUSH["volva_present"]) < 0.5,
          f"est {m.params['volva_present']:+.2f} vs planted "
          f"{ess._MUSH['volva_present']:+.2f}  (odds ratio "
          f"x{np.exp(m.params['volva_present']):.0f})")

    def loss(t: float, c_fn: int = 50, c_fp: int = 1) -> tuple[int, int, float]:
        _, fp, fn, tp = confusion_matrix(d.poisonous, (p > t).astype(int)).ravel()
        return fn, fp, c_fn * fn + c_fp * fp

    fn_half, _, loss_half = loss(0.5)
    t_star = 1 / 51
    fn_star, _, loss_star = loss(t_star)
    grid = {t: loss(t)[2] for t in np.concatenate([[0.001, 0.005, 0.01],
                                                   np.linspace(0.02, 0.9, 45)])}
    argmin = min(grid, key=grid.get)

    check("accuracy-maximising threshold poisons the forager", fn_half > 60,
          f"t=0.5 misses {fn_half} of {int(d.poisonous.sum())} poisonous "
          f"({fn_half / d.poisonous.sum():.1%})")
    check("the cost-optimal threshold is near c_FP/(c_FP+c_FN)",
          abs(argmin - t_star) < 0.05,
          f"empirical loss minimiser {argmin:.4f}, theory says {t_star:.4f}")
    check("and it cuts expected loss by an order of magnitude",
          loss_star < loss_half / 5,
          f"loss {loss_half:.0f} at t=0.5 -> {loss_star:.0f} at t={t_star:.4f}, "
          f"misses {fn_star}")


# --------------------------------------------------------------------------
def session_4_recipe() -> None:
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.preprocessing import StandardScaler

    print("\nSession 4 · secret recipe — sparse recovery and the grouping effect")
    r = ess.load("secret_recipe", refresh=True)
    x = r.drop(columns=["bake_id", "taste_score"])
    truth_effects = ess._RECIPE_TRUTH

    check("p exceeds n, so OLS has no unique solution", x.shape[1] > r.shape[0],
          f"n = {r.shape[0]}, p = {x.shape[1]}")
    rho = x.baking_soda.corr(x.baking_powder)
    check("the leavener pair is near-collinear", rho > 0.93,
          f"corr(baking_soda, baking_powder) = {rho:.4f}")

    xs = StandardScaler().fit_transform(x.values)
    y = r.taste_score.to_numpy()

    def picked(model: object) -> set[str]:
        model.fit(xs, y)
        return set(x.columns[np.flatnonzero(np.abs(model.coef_) > 1e-6)])

    lasso_m = LassoCV(cv=10, random_state=0, max_iter=100_000)
    enet_m = ElasticNetCV(l1_ratio=0.95, cv=10, random_state=0, max_iter=100_000)
    ridge_m = RidgeCV(alphas=np.logspace(-2, 4, 25))

    lasso, _enet, ridge = picked(lasso_m), picked(enet_m), picked(ridge_m)
    lasso_c = pd.Series(lasso_m.coef_, index=x.columns)
    enet_c = pd.Series(enet_m.coef_, index=x.columns)

    check("ridge names nothing", len(ridge) == x.shape[1],
          f"{len(ridge)} non-zero coefficients out of {x.shape[1]}")
    check("the lasso zeroes one of the two identical leaveners",
          (lasso_c[["baking_soda", "baking_powder"]].abs() < 1e-6).sum() == 1,
          f"soda {lasso_c.baking_soda:+.3f}, powder {lasso_c.baking_powder:+.3f}")
    check("the elastic net retains both",
          (enet_c[["baking_soda", "baking_powder"]].abs() > 0.05).all(),
          f"soda {enet_c.baking_soda:+.3f}, powder {enet_c.baking_powder:+.3f}")
    check("but does not identify the split — only the pair's total",
          abs(enet_c[["baking_soda", "baking_powder"]].sum()
              - lasso_c[["baking_soda", "baking_powder"]].sum()) < 0.6,
          f"totals: enet {enet_c[['baking_soda', 'baking_powder']].sum():+.3f}, "
          f"lasso {lasso_c[['baking_soda', 'baking_powder']].sum():+.3f}")
    strong = ("vanilla_extract", "brown_sugar", "sea_salt", "browned_butter")
    check("the four strong signals come back with the right signs",
          all(np.sign(enet_c[k]) == np.sign(truth_effects[k]) for k in strong),
          ", ".join(f"{k.split('_')[0]} {enet_c[k]:+.2f}" for k in strong))
    check("the weakest planted signal is NOT recovered at n=60",
          abs(enet_c["dutch_cocoa"]) < 0.4,
          f"dutch_cocoa true +1.10, estimated {enet_c.dutch_cocoa:+.3f} "
          f"— below the detection floor, and that is the lesson")
    check("the CV-optimal lasso over-selects (it tunes for prediction)",
          len(lasso) > 12,
          f"{len(lasso)} selected, only 7 are real")


# --------------------------------------------------------------------------
def session_5_bestiary() -> None:
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    print("\nSession 5 · bestiary — a boundary you can check against")
    d = ess.load("bestiary", refresh=True)
    share = (d.species == "ember_drake").mean()
    check("the rare species stays rare", 0.012 < share < 0.035,
          f"ember_drake = {share:.3%} of records")

    names = ess.bestiary_species()
    bayes = np.array(names)[
        ess.bestiary_posterior(d.wing_span_cm, d.tail_length_cm).argmax(1)]
    floor = accuracy_score(d.species.astype(str).to_numpy(), bayes)
    check("the Bayes rule gives a nameable ceiling", 0.80 < floor < 0.92,
          f"Bayes accuracy = {floor:.4f}")

    y = d.species.astype(str).to_numpy()
    real = d[["wing_span_cm", "tail_length_cm"]].to_numpy()
    noisy = d[["wing_span_cm", "tail_length_cm", "mass_g",
               "antenna_mm", "scale_count"]].to_numpy()

    def curve(x: np.ndarray) -> dict[int, tuple[float, float]]:
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3,
                                              random_state=1, stratify=y)
        sc = StandardScaler().fit(xtr)
        xtr, xte = sc.transform(xtr), sc.transform(xte)
        out = {}
        for k in (1, 5, 15, 45, 125, 400):
            m = KNeighborsClassifier(k).fit(xtr, ytr)
            out[k] = (m.score(xtr, ytr), m.score(xte, yte))
        return out

    c = curve(real)
    check("k=1 memorises the training set", c[1][0] > 0.99 and c[1][1] < 0.82,
          f"train {c[1][0]:.3f}, test {c[1][1]:.3f}")
    best_k = max(c, key=lambda k: c[k][1])
    check("test accuracy peaks in the middle and lands near the floor",
          best_k not in (1, 400) and abs(c[best_k][1] - floor) < 0.05,
          f"best k = {best_k}, test {c[best_k][1]:.3f} against a floor of {floor:.3f}")
    check("very large k underfits", c[400][1] < c[best_k][1],
          f"k=400 test {c[400][1]:.3f} vs best {c[best_k][1]:.3f}")

    n = curve(noisy)
    check("three irrelevant columns cost real accuracy",
          n[best_k][1] < c[best_k][1] - 0.01,
          f"k={best_k}: {c[best_k][1]:.3f} on 2 features -> "
          f"{n[best_k][1]:.3f} with 3 noise columns added")


# --------------------------------------------------------------------------
def session_6_cat_cafe() -> None:
    import statsmodels.formula.api as smf

    print("\nSession 6 · cat cafes — where two-way fixed effects gets it wrong")
    d = ess.load("cat_cafe", refresh=True)
    truth = ess.cat_cafe_truth()

    check("every planted effect is positive", truth.true_att.min() > 0,
          f"ATT ranges +{truth.true_att.min():.2f} to +{truth.true_att.max():.2f}, "
          f"mean +{truth.true_att.mean():.2f}")

    twfe = smf.ols("dwell_min ~ has_cat + C(cafe_id) + C(quarter)", data=d).fit()
    b, t, pv = (twfe.params["has_cat"], twfe.tvalues["has_cat"],
                twfe.pvalues["has_cat"])
    check("two-way fixed effects returns the WRONG SIGN", b < 0,
          f"has_cat = {b:+.3f}")
    check("and it is confident about it", pv < 0.01,
          f"t = {t:+.2f}, p = {pv:.1e}")

    try:
        import diff_diff as dd
    except ImportError:
        print("[ skip ] diff_diff not installed — run: pip install -r requirements.txt")
        return

    bacon = dd.BaconDecomposition().fit(
        d, outcome="dwell_min", unit="cafe_id", time="quarter",
        first_treat="cat_arrival_quarter")
    forbidden = bacon.total_weight_later_vs_earlier
    check("half the weight sits on forbidden comparisons", forbidden > 0.4,
          f"{forbidden:.1%} of weight uses already-treated units as controls, "
          f"averaging {bacon.weighted_avg_later_vs_earlier:+.2f}")
    check("the decomposition reproduces the estimate exactly",
          abs(bacon.decomposition_error) < 1e-6,
          f"TWFE {bacon.twfe_estimate:+.4f} = weighted sum of its 2x2s")

    cs = dd.CallawaySantAnna(control_group="not_yet_treated").fit(
        d, outcome="dwell_min", unit="cafe_id", time="quarter",
        first_treat="cat_arrival_quarter")

    rows = [{"cohort": g, "quarter": q, "est": v["effect"], "se": v["se"]}
            for (g, q), v in cs.group_time_effects.items()]
    got = pd.DataFrame(rows).merge(truth, on=["cohort", "quarter"], how="left")
    post = got[(got.quarter >= got.cohort) & got.est.notna()]
    within = (post.est - post.true_att).abs() <= 2 * post.se

    check("Callaway-Sant'Anna gets the sign right", cs.overall_att > 0,
          f"overall ATT = {cs.overall_att:+.3f} "
          f"(planted mean +{truth.true_att.mean():.2f})")
    check("and recovers every identified cell within 2 standard errors",
          within.all(),
          f"{int(within.sum())}/{len(post)} cells, mean absolute error "
          f"{(post.est - post.true_att).abs().mean():.3f}")

    pre = got[got.quarter < got.cohort]
    check("pre-treatment placebos are flat, as planted",
          abs(pre.est.mean()) < 0.15,
          f"mean pre-treatment estimate {pre.est.mean():+.4f} over {len(pre)} cells")

    unidentified = got[(got.quarter >= got.cohort) & got.est.isna()]
    check("and it refuses the cells with no counterfactual left",
          len(unidentified) > 0,
          f"{len(unidentified)} post-treatment cells returned as NA — after "
          f"quarter 10 every cafe has a cat, so nothing identifies them")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(__doc__.strip().splitlines()[0])
    for fn in (session_1_coasters, session_2_sourdough, session_3_mushrooms,
               session_4_recipe, session_5_bestiary, session_6_cat_cafe):
        fn()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All planted answers are present and all traps still spring.")
