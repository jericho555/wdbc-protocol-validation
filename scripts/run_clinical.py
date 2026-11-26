"""Clinical-statistical companion: thresholds, paired tests, error anatomy, calibration,
baselines, and a determinism check that the panel turned up by accident.

Protocol of record throughout: the 426/143 split at random_state=42, the seeded RFE wrapper
for the twelve, everything fitted on training rows only. Every threshold and every calibration
map here is chosen inside the 426 and applied to the 143 once.
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
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, learning_curve,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')
warnings.simplefilter("ignore")
rng = np.random.default_rng(0)

print("=" * 100)
print("CLINICAL STATISTICS  | sklearn %s | xgboost %s" % (
    sklearn.__version__, xgboost.__version__))
print("=" * 100)

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
tp = [COLS.index(c) for c in TW]
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


class XGBc(XGBClassifier):
    """XGBClassifier that sklearn 1.7 will accept inside CalibratedClassifierCV."""
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        t = super().__sklearn_tags__()
        t.estimator_type = "classifier"
        return t


def xgb12(seed=42):
    return XGBc(objective="binary:logistic", nthread=4, seed=seed)


# --------------------------------------------------------------------- 1
print("\n=== 1. DOES XGBOOST DEPEND ON COLUMN ORDER? (found while cross-checking two logs) ===")
print("The panel scored standardised RFE-12 + XGBoost at 140/143; a second script that asked for")
print("the same twelve columns in alphabetical order rather than file order scored 139/143.")
print("Same rows, same hyper-parameters, same seed. This is that check, on both scales.")
combos = [("native / file order", np.asarray(trX)[:, tp], np.asarray(teX)[:, tp]),
          ("native / alphabetical", np.asarray(trX)[:, sorted(tp)], np.asarray(teX)[:, sorted(tp)]),
          ("scaled / file order", Ztr[:, tp], Zte[:, tp]),
          ("scaled / alphabetical", Ztr[:, sorted(tp)], Zte[:, sorted(tp)])]
fits = []
for nm, Xa, Xb in combos:
    m = xgb12().fit(Xa, ytr)
    p = m.predict_proba(Xb)[:, 1]
    fits.append((nm, m, p))
    print("  %-24s %3d/143  AUC %.4f" % (nm, int(((p > 0.5).astype(int) == yte).sum()),
                                         roc_auc_score(yte, p)))
for i in range(len(fits)):
    for j in range(i + 1, len(fits)):
        a, b = fits[i], fits[j]
        lab_a = (a[2] > 0.5).astype(int)
        lab_b = (b[2] > 0.5).astype(int)
        if a[0].split("/")[0] != b[0].split("/")[0]:
            continue      # only compare within the same scale
        print("  %-24s vs %-24s max|dp| %.5f, labels differ on %d of 143, counts %d vs %d" % (
            a[0], b[0], float(np.max(np.abs(a[2] - b[2]))), int((lab_a != lab_b).sum()),
            int((lab_a == yte).sum()), int((lab_b == yte).sum())))
dumps = [fits[k][1].get_booster().get_dump() for k in range(len(fits))]
print("  trees whose text differs between the two column orders (same scale): "
      "%d of %d native, %d of %d scaled" % (
          sum(1 for x, z in zip(dumps[0], dumps[1]) if x != z), len(dumps[0]),
          sum(1 for x, z in zip(dumps[2], dumps[3]) if x != z), len(dumps[2])))
print("  seeds 0/1/2/42 under the alphabetical scaled order: %s" % ", ".join(
    "%d" % int(((xgb12(s).fit(Ztr[:, sorted(tp)], ytr)
                 .predict_proba(Zte[:, sorted(tp)])[:, 1] > 0.5).astype(int) == yte).sum())
    for s in (0, 1, 2, 42)))

# --------------------------------------------------------------------- 2
print("\n=== 2. A THRESHOLD CHOSEN IN THE TRAINING ROWS, APPLIED ONCE ===")
print("reference model M_REF (unscaled RFE-12 XGBoost) and M_LIN (standardised RFE-12 logistic)")
skf = StratifiedKFold(5, shuffle=True, random_state=0)
m_lin = LogisticRegression(solver="lbfgs", max_iter=2000).fit(Ztr[:, tp], ytr)
for nm, mk, Xa, Xb in [("M_REF", lambda: xgb12(), np.asarray(trX)[:, tp], np.asarray(teX)[:, tp]),
                       ("M_LIN", lambda: LogisticRegression(solver="lbfgs", max_iter=2000),
                        Ztr[:, tp], Zte[:, tp])]:
    oof = cross_val_predict(mk(), Xa, ytr, cv=skf, method="predict_proba")[:, 1]
    ths = np.unique(np.round(oof, 4))
    best_j, best_t = -1, 0.5
    best_sj, best_st = -1, 0.5
    for t in ths:
        pr = (oof >= t).astype(int)
        sens = pr[ytr == 1].mean()
        spec = (1 - pr)[ytr == 0].mean()
        if sens + spec - 1 > best_j:
            best_j, best_t = sens + spec - 1, t
        if spec >= 0.95 and sens > best_sj:
            best_sj, best_st = sens, t
    pt = mk().fit(Xa, ytr).predict_proba(Xb)[:, 1]
    print("  %s  out-of-fold Youden threshold %.3f (J %.3f); spec>=0.95 threshold %.3f" % (
        nm, best_t, best_j, best_st))
    for tag, t in [("0.50 default", 0.5), ("CV Youden", best_t), ("CV spec>=0.95", best_st)]:
        pr = (pt >= t).astype(int)
        tn, fp, fn, tpo = np.array([[((yte == 0) & (pr == 0)).sum(), ((yte == 0) & (pr == 1)).sum()],
                                    [((yte == 1) & (pr == 0)).sum(),
                                     ((yte == 1) & (pr == 1)).sum()]]).ravel()
        print("      %-16s corr %3d  sens %.3f  spec %.3f  FN %2d  FP %2d  PPV %.3f  NPV %.3f" % (
            tag, tn + tpo, tpo / float(tpo + fn), tn / float(tn + fp), fn, fp,
            tpo / float(tpo + fp) if tpo + fp else 0.0, tn / float(tn + fn)))
    print("      test AUC %.4f  Brier %.4f  ECE %.4f (threshold-free)" % (
        roc_auc_score(yte, pt), brier_score_loss(yte, pt), ece(pt, yte.astype(float))))

# --------------------------------------------------------------------- 3
print("\n=== 3. PAIRED COMPARISONS ON THE SAME 143 ROWS (exact McNemar) ===")
print("McNemar is the right test here: both models see the identical cases, so only the")
print("discordant pairs carry information. Reported as an exact binomial test on b+c.")
CFG = [("XGB rfe12 unscaled", lambda: xgb12(), np.asarray(trX)[:, tp], np.asarray(teX)[:, tp]),
       ("XGB all30 unscaled", lambda: xgb12(), np.asarray(trX), np.asarray(teX)),
       ("LogReg rfe12 scaled", lambda: LogisticRegression(max_iter=2000), Ztr[:, tp], Zte[:, tp]),
       ("SVMlin rfe12 scaled", lambda: SVC(kernel="linear"), Ztr[:, tp], Zte[:, tp]),
       ("ExtraTrees rfe12", lambda: ExtraTreesClassifier(n_estimators=500, random_state=0),
        np.asarray(trX)[:, tp], np.asarray(teX)[:, tp]),
       ("RF all30 unscaled", lambda: RandomForestClassifier(n_estimators=100, random_state=0),
        np.asarray(trX), np.asarray(teX)),
       ("KNN20 all30", lambda: KNeighborsClassifier(n_neighbors=20),
        np.asarray(trX), np.asarray(teX))]

pcs = PCA(n_components=12, random_state=0).fit(Ztr)
CFG.append(("PCA12 XGB scaled", lambda: xgb12(), pcs.transform(Ztr), pcs.transform(Zte)))


def preds_of(nm):
    for name, mk, Xa, Xb in CFG:
        if name == nm:
            m = mk().fit(Xa, ytr)
            if hasattr(m, "predict_proba"):
                return (m.predict_proba(Xb)[:, 1] > 0.5).astype(int)
            return np.asarray(m.predict(Xb)).astype(int)
    raise KeyError(nm)


lab = {}
for nm, _mk, _a, _b in CFG:
    lab[nm] = preds_of(nm)
for nm in list(lab):
    print("  %-22s %3d/143" % (nm, int((lab[nm] == yte).sum())))
print("  pairs:")
keys = list(lab)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        d = lab[a] - lab[b]
        n01 = int((d == 1).sum())   # a right, b wrong
        n10 = int((d == -1).sum())  # a wrong, b right
        if n01 + n10 == 0:
            print("     %-22s vs %-22s identical errors (b=c=0)" % (a, b))
            continue
        pv = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
        print("     %-22s vs %-22s  b %2d c %2d  exact p %.3f%s" % (
            a, b, n01, n10, pv, "  *" if pv < 0.05 else ""))

# --------------------------------------------------------------------- 4
print("\n=== 4. HOW WIDE IS THE AUC ESTIMATE? (2000 bootstrap resamples of the 143) ===")
for nm, p in [("M_REF", xgb12().fit(np.asarray(trX)[:, tp],
                                    ytr).predict_proba(np.asarray(teX)[:, tp])[:, 1]),
              ("M_LIN", m_lin.predict_proba(Zte[:, tp])[:, 1])]:
    obs = roc_auc_score(yte, p)
    boots = []
    for _ in range(2000):
        k = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[k])) < 2:
            continue
        boots.append(roc_auc_score(yte[k], p[k]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("  %-6s AUC %.4f  bootstrap 95%% [%.4f, %.4f]  (%d resamples, min spread %.4f)" % (
        nm, obs, lo, hi, len(boots), hi - lo))

# --------------------------------------------------------------------- 5
print("\n=== 5. ERROR ANATOMY: WHICH CASES SURVIVE EVERY MODEL? ===")
misses = {}
for nm in keys:
    for i in np.where(lab[nm] != yte)[0]:
        misses[i] = misses.get(i, 0) + 1
hard = sorted(misses, key=lambda i: -misses[i])
print("  %d distinct test cases are missed by at least one of the %d configurations;" % (
    len(hard), len(keys)))
print("  %d are missed by half or more of them:" % sum(1 for v in misses.values() if v >= len(keys) // 2))
Xall = np.asarray(teX)
Zall = Zte
for i in hard[:8]:
    truth = "malignant" if yte[i] == 1 else "benign"
    z = Zall[i]
    top = np.argsort(-np.abs(z))[:3]
    votes = int(sum(lab[nm][i] for nm in keys))
    print("     case %3d  true %-9s missed by %2d/%d  called malignant by %2d  | furthest "
          "from cohort mean: %s" % (
              i, truth, misses[i], len(keys), votes,
              ", ".join("%s %+.1f sd" % (COLS[top[kk]], z[top[kk]]) for kk in range(3))))
print("  (a malignant case missed by every model is a false negative the whole panel shares;")
print("   the sd column is the case's z-score on the standardised whole-30 representation)")
print("  errors by class across the panel: %s" % ", ".join(
    "%s FN %d FP %d" % (nm, int(((lab[nm] == 0) & (yte == 1)).sum()),
                        int(((lab[nm] == 1) & (yte == 0)).sum())) for nm in keys))

# --------------------------------------------------------------------- 6
print("\n=== 6. CALIBRATION, MAP FITTED INSIDE THE 426, SCORED ON THE 143 ===")
for nm, mk, Xa, Xb in [("M_REF", lambda: xgb12(), np.asarray(trX)[:, tp], np.asarray(teX)[:, tp]),
                       ("RF all30", lambda: RandomForestClassifier(n_estimators=100,
                                                                   random_state=0),
                        np.asarray(trX), np.asarray(teX)),
                       ("M_LIN", lambda: LogisticRegression(max_iter=2000),
                        Ztr[:, tp], Zte[:, tp])]:
    raw_p = mk().fit(Xa, ytr).predict_proba(Xb)[:, 1]
    line = "  %-9s raw     Brier %.4f  logL %.4f  ECE %.4f" % (
        nm, brier_score_loss(yte, raw_p), log_loss(yte, np.clip(raw_p, 1e-6, 1 - 1e-6)),
        ece(raw_p, yte.astype(float)))
    print(line)
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(mk(), method=method, cv=5).fit(Xa, ytr)
        cp = cal.predict_proba(Xb)[:, 1]
        print("  %-9s %-7s Brier %.4f  logL %.4f  ECE %.4f  (moves %d of 143 labels)" % (
            nm, method, brier_score_loss(yte, cp), log_loss(yte, np.clip(cp, 1e-6, 1 - 1e-6)),
            ece(cp, yte.astype(float)),
            int((((cp > 0.5).astype(int)) != ((raw_p > 0.5).astype(int))).sum())))

# --------------------------------------------------------------------- 7
print("\n=== 7. WHAT DOES THE PANEL BEAT? (baselines on the same 143 rows) ===")
dm = DummyClassifier(strategy="most_frequent").fit(ytr, ytr)
print("  majority class (%s of the 426)          %3d/143" % (
    "benign", int((dm.predict(np.zeros((len(yte), 1))).ravel() == yte).sum())))
np.random.seed(0)
dm2 = DummyClassifier(strategy="uniform").fit(ytr, ytr)
print("  uniform chance draw                      %3d/143" % int(
    (dm2.predict(np.zeros((len(yte), 1))).ravel() == yte).sum()))
for c in ["concave points_worst", "area_worst", "perimeter_worst", "radius_worst"]:
    v = np.asarray(trX[c])
    ths = np.unique(np.round(v, 4))
    # the cut point AND the direction are chosen on the 426 only, then read off the 143 once
    best, best_acc, best_dir = 0.0, -1, 1
    for t in ths:
        for dr in (1, -1):
            a = int(((dr * (v - t) >= 0).astype(int) == ytr).sum())
            if a > best_acc:
                best, best_acc, best_dir = t, a, dr
    acc = int(((best_dir * (np.asarray(teX[c]) - best) >= 0).astype(int) == yte).sum())
    print("  single rule: %s %s %.3f  train %3d/426  test %3d/143" % (
        c, ">=" if best_dir == 1 else "<=", best, best_acc, acc))
print("  (no model in between; the cut is fitted on the 426, never on the 143)")

# --------------------------------------------------------------------- 8
print("\n=== 8. IS 426 ROWS ENOUGH? (learning curve, stratified 5-fold, training rows only) ===")
for nm, mk, Xa in [("M_REF", lambda: xgb12(), np.asarray(trX)[:, tp]),
                   ("M_LIN", lambda: LogisticRegression(max_iter=2000), Ztr[:, tp]),
                   ("LogReg all30", lambda: LogisticRegression(max_iter=2000), Ztr)]:
    sizes, tr_s, va_s = learning_curve(mk(), Xa, ytr, cv=skf,
                                       train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
                                       scoring="accuracy", random_state=0, n_jobs=2)
    print("  %-13s" % nm, "  ".join("%4d:%.3f" % (int(n), m) for n, m in zip(sizes, va_s.mean(1))))
    print("   %-14s" % "", "  ".join("      +-%.3f" % m for m in va_s.std(1)))
print("\ndone")
