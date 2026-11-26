"""Ensembles, run separately because the panel's voting row failed on a library mismatch.

_p1_panel.py recorded:  Vote RF+XGB+LDA  FAILED ValueError: The estimator XGBc should be a
classifier.  That is a version mismatch, not a modelling result: xgboost 1.7.2 sets
_estimator_type on the instance, sklearn 1.7.2 asks the class. This script establishes
(a) whether declaring the tag is enough, and (b) what the ensemble actually scores once it
is allowed to fit. Everything else is the same protocol: 426/143 at seed 42, seeded RFE-12,
scaler fitted on the training rows only.
"""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import is_classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')

print("=" * 96)
print("ENSEMBLES  | sklearn %s | xgboost %s | numpy %s" % (
    sklearn.__version__, xgboost.__version__, np.__version__))
print("=" * 96)


class XGBc(XGBClassifier):
    """XGBClassifier that answers sklearn 1.7's tag question truthfully."""

    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        t = super().__sklearn_tags__()
        t.estimator_type = "classifier"
        return t


probe = XGBc(objective="binary:logistic", nthread=4, seed=42)
print("plain XGBClassifier is_classifier: %s" % is_classifier(
    XGBClassifier(objective="binary:logistic", nthread=4, seed=42)))
print("shimmed   XGBc       is_classifier: %s" % is_classifier(probe))

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
tp = sorted(COLS.index(c) for c in TW)
print("twelve: %s" % ", ".join(TW))


def ece(p, y, bins=5):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum()
            err += m.sum() * abs(p[m].mean() - y[m].mean())
    return err / tot if tot else float("nan")


def report(nm, m, Xa, Xb, note=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xa, ytr)
        pred = np.asarray(m.predict(Xb)).astype(int)
        has_p = hasattr(m, "predict_proba")
        p = m.predict_proba(Xb)[:, 1] if has_p else m.decision_function(Xb)
    tn, fp, fn, tp_ = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    au = roc_auc_score(yte, p)
    line = ("  %-34s %3d/143  sens %.3f spec %.3f FN %2d FP %2d  AUC %.4f" % (
        nm, int(tn + tp_), tp_ / float(tp_ + fn), tn / float(tn + fp), int(fn), int(fp), au))
    if has_p:
        line += "  Brier %.4f logL %.4f ECE %.4f" % (
            float(np.mean((p - yte) ** 2)), log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6)),
            ece(p, yte.astype(float)))
    else:
        line += "  (margin only, no probability)"
    print(line + note)


def rf():
    return RandomForestClassifier(n_estimators=100, random_state=0)


def et():
    return ExtraTreesClassifier(n_estimators=500, random_state=0)


def lda():
    return LinearDiscriminantAnalysis()


def xg():
    return XGBc(objective="binary:logistic", nthread=4, seed=42)


for arm, (Xa, Xb) in [("native", (np.asarray(trX)[:, tp], np.asarray(teX)[:, tp])),
                      ("scaled", (Ztr[:, tp], Zte[:, tp]))]:
    print("\n--- %s, the twelve ---" % arm)
    report("single: XGBoost", xg(), Xa, Xb)
    report("single: Extra trees", et(), Xa, Xb)
    report("single: LogReg L2", LogisticRegression(max_iter=2000), Xa, Xb)
    try:
        report("vote soft RF+XGB+LDA", VotingClassifier(
            [("rf", rf()), ("xg", xg()), ("lda", lda())], voting="soft"), Xa, Xb)
    except Exception as exc:
        print("  vote soft RF+XGB+LDA FAILED %s: %s" % (type(exc).__name__, str(exc)[:70]))
    try:
        report("vote hard RF+XGB+LDA", VotingClassifier(
            [("rf", rf()), ("xg", xg()), ("lda", lda())], voting="hard"), Xa, Xb)
    except Exception as exc:
        print("  vote hard RF+XGB+LDA FAILED %s: %s" % (type(exc).__name__, str(exc)[:70]))
    try:
        report("vote soft RF+ET+XGB+LDA", VotingClassifier(
            [("rf", rf()), ("et", et()), ("xg", xg()), ("lda", lda())], voting="soft"), Xa, Xb)
    except Exception as exc:
        print("  vote soft RF+ET+XGB+LDA FAILED %s: %s" % (type(exc).__name__, str(exc)[:70]))
    try:
        report("stack RF+ET+XGB+LDA, LR meta", StackingClassifier(
            [("rf", rf()), ("et", et()), ("xg", xg()), ("lda", lda())],
            final_estimator=LogisticRegression(max_iter=2000), cv=5, stack_method="auto"), Xa, Xb)
    except Exception as exc:
        print("  stacking FAILED %s: %s" % (type(exc).__name__, str(exc)[:70]))
    try:
        report("bagged XGBoost (10 x, row subsample .8)",
               BaggingClassifier(estimator=xg(), n_estimators=10, max_samples=0.8,
                                 random_state=0), Xa, Xb)
    except Exception as exc:
        print("  bagging FAILED %s: %s" % (type(exc).__name__, str(exc)[:70]))

print("""
Note on interpretation: an ensemble of models fitted on the same twelve redundant columns
cannot be expected to disagree much, and the McNemar pairs in _p1_clin.py measure whether
any of them disagree on the cases that matter.""")
print("done")
