"""Protocol experiments that close the gaps the manuscript itself names.

Section V of main.tex lists four things that were not done:
  "A stratified k-fold protocol with the selector fitted inside each fold ... has not been run."
  "whether 8 or 16 would do the same job was not tested."
  "A comparison among defaults measures the defaults as much as the algorithms."
  "no calibration analysis was performed."
This script does the first three; calibration is in _p1_clin.py. Nothing here replaces the
reference partition result - it puts an interval around it that does not depend on 143 rows.

Folds: repeated stratified K-fold over the full 569 rows, with the RFE->classifier pipeline
fitted inside each training fold, so the held-out fold never influences the subset. The
random_state=42 / 426/143 partition is left untouched and reported alongside for comparison.
"""
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')
warnings.simplefilter("ignore")

print("=" * 100)
print("PROTOCOL  | sklearn %s | xgboost %s" % (
    sklearn.__version__, xgboost.__version__))
print("=" * 100)

raw = pd.read_csv(DATA_PATH)
X = raw.drop([c for c in ("id", "Unnamed: 32", "diagnosis") if c in raw.columns], axis=1)
y = raw["diagnosis"].map({"M": 1, "B": 0}).values
COLS = list(X.columns)
XA = np.asarray(X, dtype=float)


def xgb12():
    return XGBClassifier(objective="binary:logistic", nthread=4, seed=42)


def pipe(clf, k=None, scale=False):
    steps = []
    if scale:
        steps.append(("sc", StandardScaler()))
    if k is not None:
        steps.append(("fs", RFE(estimator=DecisionTreeClassifier(random_state=0),
                                n_features_to_select=k)))
    steps.append(("m", clf))
    return Pipeline(steps)


def rbf_cv(model, label, n_splits=5, n_repeats=4, seed=0):
    """One pass of repeated stratified CV. Returns the per-fold counts."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    correct, aucs = [], []
    tnfnp = np.zeros(4, dtype=int)
    for f, (tr, te) in enumerate(cv.split(XA, y)):
        m = model()
        m.fit(XA[tr], y[tr])
        p = m.predict(XA[te])
        correct.append(int((p == y[te]).sum()))
        tnfnp += np.array(confusion_matrix(y[te], p, labels=[0, 1]).ravel())
        if hasattr(m, "predict_proba"):
            aucs.append(roc_auc_score(y[te], m.predict_proba(XA[te])[:, 1]))
    tn, fp, fn, tp = tnfnp
    acc = np.mean(correct) / len(y)
    sd = np.std(correct) / len(y)
    print("  %-30s mean %.4f of 569 (%.2f cases per 143-equivalent), sd across %2d folds "
          "%.4f" % (label, acc, acc * 143, len(correct), sd))
    print("      pooled: %d/%d correct  sens %.3f  spec %.3f  FN %4d  FP %4d   AUC(mean of "
          "folds) %s" % (
              int(tn + tp), len(y) * n_repeats, tp / float(tp + fn), tn / float(tn + fp), fn, fp,
              "%.4f" % np.mean(aucs) if aucs else "n/a"))
    return np.array(correct)


print("""
=== 1. STRATIFIED CV WITH THE SELECTOR INSIDE EACH FOLD ===
20 folds: 5-fold stratified, repeated 4 times with different fold assignments. The held-out
fold never contributes to the RFE ranking, so the subset is chosen blind to it every time.
This is the protocol the manuscript says has not been run.""")
base = rbf_cv(lambda: pipe(xgb12()), "XGBoost, all 30, native scale")
red = rbf_cv(lambda: pipe(xgb12(), k=12), "XGBoost + seeded RFE-12, native")
lin = rbf_cv(lambda: pipe(LogisticRegression(max_iter=2000), k=12, scale=True),
             "LogReg L2 + RFE-12, standardised")
lins = rbf_cv(lambda: pipe(LogisticRegression(max_iter=2000), scale=True),
              "LogReg L2, all 30, standardised")
linn = rbf_cv(lambda: pipe(LogisticRegression(max_iter=2000)),
              "LogReg L2, all 30, native scale")
rf = rbf_cv(lambda: pipe(RandomForestClassifier(n_estimators=100, random_state=0)),
            "Random Forest, all 30, native")
svmr = rbf_cv(lambda: pipe(SVC(kernel="rbf"), k=12, scale=True),
              "SVM rbf + RFE-12, standardised")
svml = rbf_cv(lambda: pipe(SVC(kernel="linear"), scale=True),
              "SVM linear, all 30, standardised")
mlp = rbf_cv(lambda: pipe(MLPClassifier(random_state=0, max_iter=4000), scale=True),
             "MLP (seed 0, 4000 it), std, all 30")

print("""
=== 2. PAIRED ACROSS FOLDS: DOES THE REDUCTION MOVE ANYTHING? ===
Same fold assignments for both models, so the comparison is paired case by case.""")
for nm, a, b in [("reduced-12 minus all-30 (XGB)", red, base),
                 ("std-logistic-12 minus native-logistic-30", lin, linn)]:
    d = (a - b) / float(len(y)) * 100
    t = stats.ttest_1samp(d, 0.0)
    lo, hi = np.mean(d) - 1.96 * stats.sem(d), np.mean(d) + 1.96 * stats.sem(d)
    print("  %-44s mean %+.2f pp, 95%% CI %+.2f to %+.2f, paired t p %.3f (wins %d, losses %d, "
          "ties %d)" % (nm, np.mean(d), lo, hi, t.pvalue, int((d > 0).sum()),
                        int((d < 0).sum()), int((d == 0).sum())))
    print("     folds are not independent (4 repeats share rows), so the interval is optimistic")

# --------------------------------------------------------------------- 3
print("""
=== 3. THE STOPPING RULE, TESTED (RFE target size, 20 stratified folds each) ===
"whether 8 or 16 would do the same job was not tested" - this is that test.""")
print("  %-6s %-14s %-14s %-14s %s" % ("k", "XGB corr/569", "LogReg(std)", "RF",
                                        "subset overlap with the reference twelve"))
tr42, _te, y42, _y2 = train_test_split(XA, y, random_state=42, shuffle=True, test_size=0.25)
ref12 = set(c for c, v in zip(
    COLS, RFE(estimator=DecisionTreeClassifier(random_state=0),
              n_features_to_select=12).fit(tr42, y42).support_) if v)
print("  (reference twelve, fitted once on the 426: %s)" % ", ".join(sorted(ref12)))
for k in (4, 6, 8, 10, 12, 16, 20, 24, 28, 30):
    row = []
    for mk in [lambda: pipe(xgb12(), k=k),
               lambda: pipe(LogisticRegression(max_iter=2000), k=k, scale=True),
               lambda: pipe(RandomForestClassifier(n_estimators=100, random_state=0), k=k)]:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
        c = 0
        for tr, te in cv.split(XA, y):
            c += int((mk().fit(XA[tr], y[tr]).predict(XA[te]) == y[te]).sum())
        row.append(c / 5.0)
    r = RFE(estimator=DecisionTreeClassifier(random_state=0), n_features_to_select=k).fit(
        tr42, y42)
    sel = set(c for c, v in zip(COLS, r.support_) if v)
    ov = "%d of 12" % len(sel & ref12)
    print("  %-6d %-14.1f %-14.1f %-14.1f %s" % (k, row[0], row[1], row[2], ov))

# --------------------------------------------------------------------- 4
print("""
=== 4. A LITTLE TUNING, IN FOLDS ONLY (the "nothing was tuned" gap) ===
Grid search nested inside the cross-validation: for each outer training fold, a 3-fold inner
search chooses the hyper-parameter, and the outer fold scores the result. Nothing is picked on
the outer test fold.""")
from sklearn.model_selection import GridSearchCV


def nested(est, params, label, outer_seed=0):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=outer_seed)
    c = 0
    picks = []
    for tr, te in cv.split(XA, y):
        g = GridSearchCV(est, params, cv=3, scoring="accuracy", n_jobs=1, refit=True)
        g.fit(XA[tr], y[tr])
        picks.append(g.best_params_)
        c += int((g.predict(XA[te]) == y[te]).sum())
    print("  %-34s %6.1f / 569 correct  (%.3f)" % (label, c / 5.0, (c / 5.0) / len(y)))
    uniq = {}
    for p in picks:
        uniq[str(sorted(p.items()))] = uniq.get(str(sorted(p.items())), 0) + 1
    for kk, vv in sorted(uniq.items(), key=lambda t: -t[1]):
        print("       chosen in %d of 5 folds: %s" % (vv, kk))


nested(SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
       "SVM, native scale (no scaler)")
print("  (this row is deliberately the unscaled fit, to be read against the scaled one below; "
      "the scaler is not part of this pipeline)")
sc_all = StandardScaler().fit(XA)
XS = sc_all.transform(XA)   # reported only as a diagnostic; the CV below scales inside folds


def nested_scaled(est, params, label):
    """Grid search nested in 5-fold CV, scaler refitted inside every outer training fold."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
    c = 0
    picks = []
    for tr, te in cv.split(XA, y):
        g = GridSearchCV(Pipeline([("sc", StandardScaler()), ("m", est)]),
                         {"m__" + k: v for k, v in params.items()},
                         cv=3, scoring="accuracy", n_jobs=1, refit=True)
        g.fit(XA[tr], y[tr])
        picks.append(g.best_params_)
        c += int((g.predict(XA[te]) == y[te]).sum())
    print("  %-34s %6.1f / 569 correct  (%.3f)" % (label, c / 5.0, (c / 5.0) / len(y)))
    uniq = {}
    for p in picks:
        uniq[str(sorted(p.items()))] = uniq.get(str(sorted(p.items())), 0) + 1
    for kk, vv in sorted(uniq.items(), key=lambda t: -t[1]):
        print("       chosen in %d of 5 folds: %s" % (vv, kk))


nested_scaled(SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
              "SVM, properly scaled")
nested_scaled(LogisticRegression(solver="liblinear", max_iter=5000),
              {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]},
              "Logistic, scaled, l1/l2 x C")
nested(MLPClassifier(random_state=0, max_iter=4000), {"hidden_layer_sizes": [(100,), (50, 25)]},
       "MLP, native scale (for contrast)")
print("\ndone at %s" % time.strftime("%H:%M:%S"))
