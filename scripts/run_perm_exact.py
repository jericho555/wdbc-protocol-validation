"""Exact permutation importance on RFE-12 for the four model families.

Reproduces the section-5 experiment from _p1_interp.py, but computes the
importance with a plain numpy loop instead of sklearn.inspection.permutation_importance.
That entry point went through joblib/loky and terminated a worker on this Windows
host (_p1_interp_b.log, section 5); the loop below is mathematically identical
(each column shuffled in isolation, 50 repeats, RandomState(0)) and needs no
parallel backend.

Everything else is unchanged: same data, same 75/25 split with random_state=42,
same RFE-12 filter (DecisionTree, random_state=0) fitted on the 426 training
rows, same four families on the same feature matrices.
"""
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
# . (sklearn 1.7.2) must win, to match _p1_interp_b.log's TW ordering.

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.simplefilter("ignore")

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')

N_REPEATS = 50
SEED = 0  # matches _p1_interp.py, section 5

print("=" * 100)
print("EXACT PERMUTATION IMPORTANCE | RFE-12 | 4 families | %d repeats/column" % N_REPEATS)
print("=" * 100)
import sklearn as _sk, numpy as _np, xgboost as _xgb
print("stack: sklearn %s | numpy %s | xgboost %s" % (_sk.__version__, _np.__version__, _xgb.__version__))

# ----------------------------------------------------------------- reproduce the setup
raw = pd.read_csv(DATA_PATH)
data = raw.drop([c for c in ("id", "Unnamed: 32", "diagnosis") if c in raw.columns], axis=1)
labels = raw["diagnosis"].map({"M": 1, "B": 0})
COLS = list(data.columns)
trX, teX, trY, teY = train_test_split(data, labels, random_state=42, shuffle=True, test_size=0.25)
ytr, yte = np.asarray(trY), np.asarray(teY)
sc = StandardScaler().fit(trX)
Ztr, Zte = sc.transform(trX), sc.transform(teX)

rfe = RFE(estimator=DecisionTreeClassifier(random_state=0), n_features_to_select=12).fit(trX, trY)
TW = sorted(c for c, v in zip(COLS, rfe.support_) if v)
t_pos = [COLS.index(c) for c in TW]
print("the retained twelve: %s" % ", ".join(TW))
LOG_TW = ['area_se', 'area_worst', 'compactness_se', 'compactness_worst',
          'concave points_mean', 'concave points_se', 'fractal_dimension_se',
          'perimeter_worst', 'radius_worst', 'smoothness_se',
          'smoothness_worst', 'texture_worst']
if TW != LOG_TW:
    print("WARNING: TW differs from _p1_interp_b.log line 15.")
    print("         logged: %s" % ", ".join(LOG_TW))
    print("         rerun:  %s" % ", ".join(TW))
else:
    print("         TW matches the logged run.")

# raw-scale matrices for tree families, standardised for LogReg
X12_tr = np.asarray(trX)[:, t_pos]
X12_te = np.asarray(teX)[:, t_pos]
Z12_tr = Ztr[:, t_pos]
Z12_te = Zte[:, t_pos]

FAMILIES = [
    ("XGBoost",        XGBClassifier(objective="binary:logistic", nthread=4, seed=42), X12_tr, X12_te),
    ("Random forest",  RandomForestClassifier(n_estimators=100, random_state=0),       X12_tr, X12_te),
    ("Extra trees",    ExtraTreesClassifier(n_estimators=500, random_state=0),         X12_tr, X12_te),
    ("LogReg L2 (std)", LogisticRegression(solver="lbfgs", max_iter=2000),              Z12_tr, Z12_te),
]

# ----------------------------------------------------------------- permutation loop
def exact_permutation_importance(fitted, X, y, n_repeats=N_REPEATS, seed=SEED):
    """Mean and per-repeat accuracy drop from shuffling each column of X.

    Matches sklearn.inspection.permutation_importance(scoring='accuracy',
    n_repeats=..., random_state=...) column-by-column, using a single
    RandomState(0) stream and no joblib worker.
    """
    X = np.asarray(X, dtype=float).copy()
    y = np.asarray(y)
    n, p = X.shape
    baseline = float(fitted.score(X, y))
    rng = np.random.RandomState(seed)
    per_repeat = np.zeros((p, n_repeats))
    for j in range(p):
        original = X[:, j].copy()
        for r in range(n_repeats):
            X[:, j] = original[rng.permutation(n)]
            per_repeat[j, r] = baseline - float(fitted.score(X, y))
        X[:, j] = original
    return baseline, per_repeat


def shap_ranking_from_log():
    """mean |phi| for M_REF from _p1_interp_b.log section 3, in TW order."""
    # logged descending: concave points_mean, texture_worst, area_worst, perimeter_worst,
    # compactness_se, smoothness_worst, area_se, compactness_worst, radius_worst,
    # concave points_se, smoothness_se, fractal_dimension_se
    order = ['concave points_mean', 'texture_worst', 'area_worst', 'perimeter_worst',
             'compactness_se', 'smoothness_worst', 'area_se', 'compactness_worst',
             'radius_worst', 'concave points_se', 'smoothness_se', 'fractal_dimension_se']
    vals = [2.1956, 1.3975, 1.2835, 0.8264, 0.7832, 0.7622,
            0.7335, 0.5550, 0.3787, 0.3579, 0.1870, 0.1379]
    m = dict(zip(order, vals))
    return np.array([m[c] for c in TW])


phi_shap = shap_ranking_from_log()

results = {}
print("\n%-16s %10s %10s %10s" % ("family", "baseline", "largest", "smallest"))
for nm, model, Xt, Xe in FAMILIES:
    t0 = time.time()
    fitted = model.fit(Xt, ytr)
    baseline, per_repeat = exact_permutation_importance(fitted, Xe, yte)
    mean = per_repeat.mean(1)
    sd = per_repeat.std(1, ddof=1)
    rho = stats.spearmanr(phi_shap, mean).statistic
    results[nm] = {
        "baseline_accuracy": baseline,
        "baseline_correct": int(round(baseline * len(yte))),
        "total": len(yte),
        "mean_drop": [float(v) for v in mean],
        "sd_drop": [float(v) for v in sd],
        "per_repeat": [[float(v) for v in row] for row in per_repeat],
        "spearman_with_shap": float(rho),
        "seconds": time.time() - t0,
    }
    hi = int(np.argmax(mean))
    lo = int(np.argmin(mean))
    print("%-16s %10.3f %10.4f   %10.4f   (%.1fs)" % (
        nm, baseline, mean[hi], mean[lo], results[nm]["seconds"]))
    results[nm]["top_feature"] = TW[hi]
    results[nm]["bottom_feature"] = TW[lo]

# full table (per-feature, four families)
print("\n%-24s %s" % ("feature", " ".join("%-14s" % n for n, *_ in FAMILIES)))
print("-" * (24 + 15 * len(FAMILIES)))
for j, feat in enumerate(TW):
    row = " ".join("%-14s" % ("%.4f" % results[n]["mean_drop"][j]) for n, *_ in FAMILIES)
    print("%-24s %s" % (feat, row))
print("-" * (24 + 15 * len(FAMILIES)))
row_shap = " ".join("%-14s" % ("%.4f" % phi_shap[j]) for j in range(12))
print("%-24s %s" % ("[SHAP mean |phi|]", row_shap))

print("\nRanking per family (best -> weakest):")
for nm, *_ in FAMILIES:
    order = np.argsort(-np.array(results[nm]["mean_drop"]))
    labels = ["%s" % TW[j] for j in order]
    print("  %-16s %s" % (nm, " > ".join(labels)))

print("\nSpearman of each family's permutation ranking with M_REF's SHAP ranking:")
for nm, *_ in FAMILIES:
    print("  %-16s rho %+.3f" % (nm, results[nm]["spearman_with_shap"]))

# cross-family mean of the four normalised ranks; used by consensus figure
mean_drop_matrix = np.array([results[n]["mean_drop"] for n, *_ in FAMILIES])
cross_family_mean = mean_drop_matrix.mean(0)
print("\nCross-family mean accuracy drop (per column), descending:")
for j in np.argsort(-cross_family_mean):
    print("  %-24s %.4f" % (TW[j], cross_family_mean[j]))

# save JSON for _p1_visualize.py
out_path = os.path.join(os.path.dirname(HERE), 'data', 'perm_exact.json')
payload = {
    "TW": TW,
    "n_repeats": N_REPEATS,
    "seed": SEED,
    "families": [n for n, *_ in FAMILIES],
    "per_family": {n: {
        "baseline_accuracy": results[n]["baseline_accuracy"],
        "baseline_correct": results[n]["baseline_correct"],
        "mean_drop": results[n]["mean_drop"],
        "sd_drop": results[n]["sd_drop"],
        "spearman_with_shap": results[n]["spearman_with_shap"],
        "top_feature": results[n]["top_feature"],
        "bottom_feature": results[n]["bottom_feature"],
    } for n in [nm for nm, *_ in FAMILIES]},
    "cross_family_mean_drop": [float(v) for v in cross_family_mean],
    "shap_mean_abs_in_TW_order": [float(v) for v in phi_shap],
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
print("\nSaved: %s" % out_path)
