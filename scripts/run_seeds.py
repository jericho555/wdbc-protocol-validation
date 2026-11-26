"""
Follow-up to _p1_rerun.py: is the RFE result a property of the method or of seed 42?

A. 30 paired splits (seed 0..29, everything else as in the notebook). On each split
   the full 30-feature XGBoost and the 12-feature RFE pipeline are both fitted on
   trainX only, and compared on the same 143-case test partition.

B. 200 unseeded RandomForest fits on the seed-42 split, to see whether the published
   140/143 is reachable at all when random_state is not set.

C. How stable the 12 selected features are across the 30 splits.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

CSV = os.path.join(HERE, "paper 2", "data.csv")
if not os.path.exists(CSV):
    CSV = os.path.join(HERE, "paper 1", "data.csv")
if not os.path.exists(CSV):
    print("data.csv not found")
    sys.exit(2)

raw = pd.read_csv(CSV)
data = raw.drop([c for c in ("id", "Unnamed: 32", "diagnosis") if c in raw.columns], axis=1)
labels = raw["diagnosis"].replace({"M": 1, "B": 0}).astype(int)
print("data: %d x %d, %d/%d" % (data.shape[0], data.shape[1],
                                int((labels == 0).sum()), int((labels == 1).sum())))


def xgb():
    return XGBClassifier(objective="binary:logistic", nthread=4, seed=42)


# ---------------------------------------------------------------- section A
print("\n=== A. THIRTY PAIRED SPLITS, FULL MODEL vs RFE MODEL ===")
print("%4s %8s %8s %8s   %s" % ("seed", "full", "rfe12", "diff", "selected"))
diffs, full_acc, rfe_acc = [], [], []
counts = {}
for seed in range(30):
    trX, teX, trY, teY = train_test_split(
        data, labels, random_state=seed, shuffle=True, test_size=0.25)
    m_full = xgb()
    m_full.fit(trX, trY)
    c_full = int((m_full.predict(teX) == teY).sum())

    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=12)
    pipe = Pipeline(steps=[("fs", rfe), ("model", xgb())])
    pipe.fit(trX, trY)
    c_rfe = int((pipe.predict(teX) == teY).sum())
    for name in data.columns[rfe.get_support()]:
        counts[name] = counts.get(name, 0) + 1

    diffs.append(c_rfe - c_full)
    full_acc.append(c_full)
    rfe_acc.append(c_rfe)
    mark = "" if seed != 42 else "   <- the paper's seed"
    print("%4d %6d/143 %6d/143 %+8d   %d/%d%s" % (
        seed, c_full, c_rfe, c_rfe - c_full, int(rfe.get_support().sum()),
        len(teY), mark))

d = np.array(diffs, dtype=float)
print("\nfull  : mean %.2f cases correct, sd %.2f, range %d-%d" % (
    np.mean(full_acc), np.std(full_acc, ddof=1), min(full_acc), max(full_acc)))
print("rfe12 : mean %.2f cases correct, sd %.2f, range %d-%d" % (
    np.mean(rfe_acc), np.std(rfe_acc, ddof=1), min(rfe_acc), max(rfe_acc)))
print("paired difference (rfe - full): mean %+.2f cases, sd %.2f, "
      "rfe better %d/30, worse %d/30, tied %d/30" % (
          d.mean(), d.std(ddof=1), int((d > 0).sum()), int((d < 0).sum()),
          int((d == 0).sum())))
print("95%% CI on the mean difference: %+.2f to %+.2f cases" % (
    d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
    d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d))))

# ---------------------------------------------------------------- section B
print("\n=== B. 200 UNSEEDED RANDOM FOREST FITS ON THE SEED-42 SPLIT ===")
trX, teX, trY, teY = train_test_split(
    data, labels, random_state=42, shuffle=True, test_size=0.25)
correct = []
for _ in range(200):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(trX, trY)
    correct.append(int((rf.predict(teX) == teY).sum()))
arr = np.array(correct)
print("correct cases: mean %.2f, sd %.2f, min %d, max %d" % (
    arr.mean(), arr.std(ddof=1), arr.min(), arr.max()))
for target in (138, 139, 140, 141):
    print("  %d/143 (%.4f) occurred %d/200" % (
        target, target / 143.0, int((arr == target).sum())))

# ---------------------------------------------------------------- section C
print("\n=== C. FEATURE STABILITY ACROSS THE 30 SPLITS ===")
for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
    print("  %-26s %2d/30" % (name, n))
print("features selected in all 30 splits: %d of 30" % (
    sum(1 for v in counts.values() if v == 30)))
