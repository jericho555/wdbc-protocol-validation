"""Consolidated final run for paper 1.

Usage:  python _p1_final.py b      (notebook stack: xgboost 1.7.2 / numpy 1.26.4)
        python _p1_final.py a      (current stack: xgboost 3.2.0 / numpy 2.2.6)

Sections
  1  reference partition (seed 42): the seven published configurations
  2  reference partition: the reduced model with the wrapper tree seeded
  3  thirty partitions (seeds 0-29), both models, paired
  4  thirty refits of the same partition with the wrapper tree UNSEEDED
     -- measures how far the published call actually moves
"""
import math
import os
import sys
import warnings


HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')

print("=" * 78)
print("xgboost %s | numpy %s | sklearn %s" % (
    xgboost.__version__, np.__version__, sklearn.__version__))
print("=" * 78)

path = next(p for p in CANDIDATES if os.path.exists(p))
raw = pd.read_csv(path)
data = raw.drop([c for c in ("id", "Unnamed: 32", "diagnosis") if c in raw.columns], axis=1)
labels = raw["diagnosis"].map({"M": 1, "B": 0})
print("file %s | %d x %d | benign %d / malignant %d" % (
    os.path.basename(path), data.shape[0], data.shape[1],
    int((labels == 0).sum()), int((labels == 1).sum())))


def wilson(k, n, z=1.959963984540054):
    p = k / float(n)
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def row(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    k = tn + tp
    sens = tp / float(tp + fn)
    spec = tn / float(tn + fp)
    prec = tp / float(tp + fp)
    f1 = 2 * prec * sens / (prec + sens)
    lo, hi = wilson(k, n)
    return dict(name=name, k=k, n=n, tp=tp, fn=fn, tn=tn, fp=fp, sens=sens,
                spec=spec, prec=prec, f1=f1, lo=lo, hi=hi)


def show(r):
    print("%-20s %3d/%-3d  %5.1f-%-5.1f  sens %.3f  spec %.3f  FN %d  FP %d  "
          "prec %.3f  F1 %.3f" % (r["name"], r["k"], r["n"], r["lo"], r["hi"],
                                  r["sens"], r["spec"], r["fn"], r["fp"],
                                  r["prec"], r["f1"]))


def xgb(threads=4):
    return XGBClassifier(objective="binary:logistic", nthread=threads, seed=42)


trX, teX, trY, teY = train_test_split(
    data, labels, random_state=42, shuffle=True, test_size=0.25)
print("\ntest partition: %d cases (%d benign / %d malignant)" % (
    len(teY), int((teY == 0).sum()), int((teY == 1).sum())))

# ---------------------------------------------------------------- section 1
print("\n=== 1. TABLE I, REFERENCE PARTITION (notebook calls, unchanged) ===")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    configs = [
        ("KNN", KNeighborsClassifier(n_neighbors=20)),
        ("SVM", SVC(kernel="linear")),
        ("Logistic Regression", LogisticRegression(solver="lbfgs")),
        ("MLP", MLPClassifier()),
        ("Random Forest", RandomForestClassifier(n_estimators=100)),
        ("XGBoost", xgb()),
    ]
    table1 = []
    for name, clf in configs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(trX, trY)
            table1.append(row(name, teY, clf.predict(teX)))
        show(table1[-1])
conv = sorted({"%s: %s" % (w.category.__name__, str(w.message)[:60])
               for w in caught if issubclass(w.category, ConvergenceWarning)})
print("convergence warnings in this run: %s" % ("; ".join(conv) if conv else "none"))

# ---------------------------------------------------------------- section 2
print("\n=== 2. REDUCED MODEL, WRAPPER TREE SEEDED (random_state=0) ===")
rfe = RFE(estimator=DecisionTreeClassifier(random_state=0), n_features_to_select=12)
pipe = Pipeline(steps=[("fs", rfe), ("model", xgb())])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipe.fit(trX, trY)
sub_sel = list(data.columns[rfe.get_support()])
r = row("XGBoost+RFE", teY, pipe.predict(teX))
show(r)
print("selected: %s" % ", ".join(sub_sel))

print("\n=== 2b. THE SAME MODEL WITH THE NOTEBOOK'S UNSEEDED WRAPPER, 30 REFITS ===")
from collections import Counter
cnt, subs = Counter(), Counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(30):
        f = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=12)
        p = Pipeline(steps=[("fs", f), ("model", xgb())])
        p.fit(trX, trY)
        cnt[int((p.predict(teX) == teY.values).sum())] += 1
        subs[tuple(sorted(data.columns[f.get_support()]))] += 1
print("correct counts : %s" % dict(sorted(cnt.items())))
print("distinct subsets: %d ; most common seen %d/30" % (len(subs), subs.most_common(1)[0][1]))

# ---------------------------------------------------------------- section 3
for variant in (None, 0):
    tag = "the notebook's unseeded wrapper" if variant is None else "wrapper seeded at 0"
    print("\n=== 3. THIRTY PAIRED PARTITIONS, seeds 0-29, %s ===" % tag)
    print("%4s %10s %10s %6s" % ("seed", "full", "rfe12", "diff"))
    diffs, full_c, rfe_c, feat = [], [], [], Counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(30):
            a, b_, c_, d_ = train_test_split(data, labels, random_state=seed,
                                             shuffle=True, test_size=0.25)
            m = xgb().fit(a, c_)
            cf = int((m.predict(b_) == d_).sum())
            f = RFE(estimator=DecisionTreeClassifier(random_state=variant),
                    n_features_to_select=12)
            p = Pipeline(steps=[("fs", f), ("model", xgb())]).fit(a, c_)
            cr = int((p.predict(b_) == d_).sum())
            for name in data.columns[f.get_support()]:
                feat[name] += 1
            diffs.append(cr - cf)
            full_c.append(cf)
            rfe_c.append(cr)
            print("%4d %8d/143 %8d/143 %+6d" % (seed, cf, cr, cr - cf))

    d = np.array(diffs, dtype=float)
    half = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
    print("\nfull   : %.2f +- %.2f  range %d-%d" % (
        np.mean(full_c), np.std(full_c, ddof=1), min(full_c), max(full_c)))
    print("reduced: %.2f +- %.2f  range %d-%d" % (
        np.mean(rfe_c), np.std(rfe_c, ddof=1), min(rfe_c), max(rfe_c)))
    print("paired : %+.2f cases, 95%% CI %+.2f to %+.2f, better %d / worse %d / tied %d" % (
        d.mean(), d.mean() - half, d.mean() + half,
        int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())))

    print("\nfeature stability across the 30 partitions (%s):" % tag)
    for name, n in sorted(feat.items(), key=lambda kv: (-kv[1], kv[0])):
        print("  %-26s %2d/30" % (name, n))
    print("selected in all 30: %d" % sum(1 for v in feat.values() if v == 30))

# ---------------------------------------------------------------- section 4
print("\n=== 4. THE TWO UNSEEDED CONFIGURATIONS, 30 FITS ON THE REFERENCE PARTITION ===")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, factory in (("Random Forest", lambda: RandomForestClassifier(n_estimators=100)),
                          ("MLP", lambda: MLPClassifier())):
        hits = []
        for _ in range(30):
            m = factory().fit(trX, trY)
            hits.append(int((m.predict(teX) == teY.values).sum()))
        a = np.array(hits)
        print("%-14s min %d  median %.0f  max %d  sd %.2f  distinct %s" % (
            name, a.min(), np.median(a), a.max(), a.std(ddof=1),
            dict(sorted(Counter(hits).items()))))

print("\n=== 5. THE BOOSTED MODEL ITSELF, TEN REFITS ON FIXED INPUT ===")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fixed = [c for c in sub_sel]
    prev, same = None, 0
    for _ in range(10):
        pr = xgb().fit(trX[fixed], trY).predict_proba(teX[fixed])[:, 1]
        if prev is not None and np.array_equal(prev, pr):
            same += 1
        prev = pr
    print("identical to the first fit: %d/9 ; correct %d/143" % (
        same, int(((prev > 0.5).astype(int) == teY.values).sum())))
