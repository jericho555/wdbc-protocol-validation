"""Extended model panel and interpretation for the conference paper.

Adds five classifiers not in the original 20-model panel:
  1. HistGradientBoostingClassifier - histogram-based gradient boosting (different algorithm)
  2. BaggingClassifier - bootstrap aggregation meta-estimator
  3. RidgeClassifier - linear classifier with L2 margin (not logistic)
  4. SGDClassifier - stochastic gradient linear classifier
  5. StackingClassifier - proper CV-based stacking ensemble

Also adds extended interpretation:
  - ROC curve points (TPR, FPR at multiple thresholds)
  - Precision-recall points
  - Cost-sensitive analysis (varying misclassification costs)
  - Decision curve analysis (net benefit across threshold probabilities)

Two arms: native and scaled, both on RFE-12 and full-30.
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
from sklearn.ensemble import (
    BaggingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Compatibility shim for xgboost 1.7.2 with sklearn 1.7
class XGBc(XGBClassifier):
    """XGBClassifier that sklearn 1.7 will accept inside ensemble meta-estimators."""
    _estimator_type = "classifier"
    def __sklearn_tags__(self):
        t = super().__sklearn_tags__()
        t.estimator_type = "classifier"
        return t

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')
warnings.simplefilter("ignore")

print("=" * 100)
print("EXTENDED PANEL  | sklearn %s | xgboost %s | numpy %s" % (
    sklearn.__version__, xgboost.__version__, np.__version__))
print("=" * 100)

raw = pd.read_csv(DATA_PATH)
X = raw.drop([c for c in ("id", "Unnamed: 32", "diagnosis") if c in raw.columns], axis=1)
y = raw["diagnosis"].map({"M": 1, "B": 0}).values
COLS = list(X.columns)
XA = np.asarray(X, dtype=float)

trX, teX, ytr, yte = train_test_split(XA, y, random_state=42, shuffle=True, test_size=0.25)
sc = StandardScaler().fit(trX)
Ztr = sc.transform(trX)
Zte = sc.transform(teX)

# RFE-12
rfe = RFE(estimator=DecisionTreeClassifier(random_state=0), n_features_to_select=12)
rfe.fit(trX, ytr)
TW = [c for c in COLS if rfe.support_[COLS.index(c)]]
t_pos = [COLS.index(c) for c in TW]

print("twelve: %s" % ", ".join(TW))
print("test partition: %d cases (%d benign / %d malignant)" % (
    len(yte), int((yte == 0).sum()), int((yte == 1).sum())))


def metrics(y_true, y_pred, proba=None):
    """Compute accuracy, sensitivity, specificity, AUC, Brier, log-loss."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = int(tp + tn)
    sens = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    out = "%d/%d  sens %.3f spec %.3f FN %2d FP %2d" % (acc, len(y_true), sens, spec, fn, fp)
    if proba is not None:
        try:
            auc = roc_auc_score(y_true, proba)
            brier = float(np.mean((proba - y_true) ** 2))
            eps = 1e-15
            logL = -float(np.mean(
                np.log(np.clip(proba, eps, 1 - eps)) * y_true +
                np.log(np.clip(1 - proba, eps, 1 - eps)) * (1 - y_true)))
            out += "  AUC %.4f  Brier %.4f logL %.4f" % (auc, brier, logL)
        except Exception:
            pass
    return out


def roc_points(y_true, proba, n_points=10):
    """Return TPR, FPR at n_points thresholds."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    idx = np.linspace(0, len(fpr) - 1, n_points, dtype=int)
    return [(fpr[i], tpr[i], thr[i]) for i in idx]


def pr_points(y_true, proba, n_points=10):
    """Return precision, recall at n_points thresholds."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    # thr has size n-1, prec and rec have size n
    idx = np.linspace(0, len(thr) - 1, n_points, dtype=int)
    return [(prec[i], rec[i], thr[i]) for i in idx]


def cost_sensitive(y_true, proba, cost_fn=10.0, cost_fp=1.0):
    """Compute total cost at threshold 0.5 with given misclassification costs."""
    pred = (proba > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    total = cost_fn * fn + cost_fp * fp
    return total, tn, fp, fn, tp


def decision_curve(y_true, proba, thresholds=None):
    """Net benefit at each threshold probability."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 20)
    n = len(y_true)
    prevalence = y_true.mean()
    results = []
    for t in thresholds:
        pred = (proba > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        tp_rate = tp / n
        fp_rate = fp / n
        odds = t / (1 - t)
        nb = tp_rate - fp_rate * odds
        results.append((t, nb, tp, fp))
    return results


# Define the five new models
NEW_MODELS = [
    ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, random_state=42)),
    ("Bagging (DT base)", lambda: BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=50, random_state=42, max_samples=0.8)),
    ("RidgeClassifier", lambda: RidgeClassifier(alpha=1.0, random_state=42)),
    ("SGDClassifier (log)", lambda: SGDClassifier(
        loss="log_loss", alpha=0.0001, max_iter=1000, random_state=42, tol=1e-3)),
    ("Stacking (RF+XGB+LR meta)", lambda: StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, random_state=0)),
            ("xgb", XGBc(objective="binary:logistic", nthread=4, seed=42)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5, stack_method="predict_proba")),
]


def run_model(name, model_fn, Xa, Xb, ya, yb, note=""):
    """Fit model, return (pred, proba, info_dict)."""
    m = model_fn()
    try:
        m.fit(Xa, ya)
        pred = m.predict(Xb)
        if hasattr(m, "predict_proba"):
            try:
                proba = m.predict_proba(Xb)[:, 1]
            except Exception:
                proba = m.decision_function(Xb) if hasattr(m, "decision_function") else None
        elif hasattr(m, "decision_function"):
            proba = m.decision_function(Xb)
        else:
            proba = None
        return pred, proba, {"status": "OK"}
    except Exception as exc:
        return None, None, {"status": "FAILED", "error": str(exc)}


# =============================================================================
# RUN THE EXTENDED PANEL
# =============================================================================
for arm_name, Xa, Xb in [("native", trX[:, t_pos], teX[:, t_pos]),
                          ("scaled", Ztr[:, t_pos], Zte[:, t_pos])]:
    print("\n" + "=" * 100)
    print("=== ARM: %s, RFE-12 ===" % arm_name)
    print("=" * 100)
    
    for name, model_fn in NEW_MODELS:
        pred, proba, info = run_model(name, model_fn, Xa, Xb, ytr, yte, note=arm_name)
        if info["status"] == "FAILED":
            print("  %-30s FAILED %s" % (name, info["error"]))
            continue
        
        line = "  %-30s %s" % (name, metrics(yte, pred, proba))
        print(line)
        
        # ROC points
        if proba is not None:
            roc = roc_points(yte, proba, n_points=5)
            print("     ROC (FPR, TPR at 5 thresholds): " + ", ".join(
                "(%.2f, %.2f @%.2f)" % (f, t, th) for f, t, th in roc))
            
            # Precision-recall points
            pr = pr_points(yte, proba, n_points=5)
            print("     P-R (prec, rec at 5 thresholds): " + ", ".join(
                "(%.2f, %.2f @%.2f)" % (p, r, th) for p, r, th in pr))
            
            # Cost-sensitive analysis
            for cost_fn, cost_fp in [(10.0, 1.0), (5.0, 1.0), (1.0, 5.0)]:
                total, tn, fp, fn, tp = cost_sensitive(yte, proba, cost_fn, cost_fp)
                print("     cost (FN=%.1f, FP=%.1f): total %.1f  (TP %d, FP %d, FN %d, TN %d)" % (
                    cost_fn, cost_fp, total, tp, fp, fn, tn))
            
            # Decision curve (5 representative thresholds)
            dc = decision_curve(yte, proba, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
            print("     decision curve (threshold, net_benefit, TP, FP): " + ", ".join(
                "(%.1f: %.3f, %d, %d)" % (t, nb, tp, fp) for t, nb, tp, fp in dc))


# =============================================================================
# EXTENDED INTERPRETATION: COMPARISON ACROSS ALL MODELS
# =============================================================================
print("\n" + "=" * 100)
print("=== EXTENDED INTERPRETATION ===")
print("=" * 100)

# Fit all models on scaled RFE-12, collect predictions
Xa = Ztr[:, t_pos]
Xb = Zte[:, t_pos]
all_preds = {}
all_probas = {}
for name, model_fn in NEW_MODELS:
    pred, proba, info = run_model(name, model_fn, Xa, Xb, ytr, yte)
    if info["status"] == "OK" and proba is not None:
        all_preds[name] = pred
        all_probas[name] = proba

# Agreement matrix: how often do pairs of models agree?
print("\n--- Model agreement matrix (fraction of 143 cases where both predict the same label) ---")
names = sorted(all_preds.keys())
print("  %-30s" % "", " ".join("%-12s" % n for n in names))
for n1 in names:
    row = []
    for n2 in names:
        agree = float(np.mean(all_preds[n1] == all_preds[n2]))
        row.append("%.3f" % agree)
    print("  %-30s %s" % (n1, " ".join("%-12s" % r for r in row)))

# Consensus predictions
print("\n--- Consensus analysis ---")
consensus_pred = np.mean([all_preds[n] for n in names], axis=0) > 0.5
consensus_acc = float(np.mean(consensus_pred == yte))
print("  majority-vote consensus accuracy: %.3f (%d/143)" % (consensus_acc, int(consensus_acc * 143)))

# Cases where all models agree vs disagree
all_agree = np.all([all_preds[n] == consensus_pred for n in names], axis=0)
print("  cases where all %d models agree: %d/143" % (len(names), int(all_agree.sum())))
print("  cases where at least one disagrees: %d/143" % (int((~all_agree).sum())))

# Disagreement cases: which ones, and what are their features?
disagree_idx = np.where(~all_agree)[0]
if len(disagree_idx) > 0:
    print("  disagreement cases (index, true_label, model_predictions):")
    for idx in disagree_idx[:10]:  # show first 10
        preds = [all_preds[n][idx] for n in names]
        print("     case %3d  true %d  predictions: %s" % (idx, yte[idx], ", ".join(str(p) for p in preds)))

# AUC ranking and comparison
print("\n--- AUC ranking (scaled RFE-12) ---")
aucs = []
for name in names:
    try:
        auc = roc_auc_score(yte, all_probas[name])
        aucs.append((name, auc))
    except Exception:
        pass
aucs.sort(key=lambda x: -x[1])
for rank, (name, auc) in enumerate(aucs, 1):
    print("  #%d  %-30s  AUC %.4f" % (rank, name, auc))

# Brier score ranking
print("\n--- Brier score ranking (lower is better) ---")
briers = []
for name in names:
    try:
        brier = float(np.mean((all_probas[name] - yte) ** 2))
        briers.append((name, brier))
    except Exception:
        pass
briers.sort(key=lambda x: x[1])
for rank, (name, brier) in enumerate(briers, 1):
    print("  #%d  %-30s  Brier %.4f" % (rank, name, brier))

# Calibration: expected calibration error (ECE)
print("\n--- Calibration (ECE, lower is better) ---")
def ece(y_true, proba, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        in_bin = (proba > bin_boundaries[i]) & (proba <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_pred = proba[in_bin].mean()
            avg_true = y_true[in_bin].mean()
            ece_val += np.abs(avg_pred - avg_true) * prop_in_bin
    return ece_val

eces = []
for name in names:
    try:
        e = ece(yte, all_probas[name])
        eces.append((name, e))
    except Exception:
        pass
eces.sort(key=lambda x: x[1])
for rank, (name, e) in enumerate(eces, 1):
    print("  #%d  %-30s  ECE %.4f" % (rank, name, e))


print("\ndone at %s" % time.strftime("%H:%M:%S"))
