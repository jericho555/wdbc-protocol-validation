"""Interpretation experiments for the conference paper.

Two models are interpreted, not one:
  M_REF  the protocol of record - unscaled, seeded RFE-12, XGBoost            (140/143)
  M_LIN  the best scorer in the new panel - standardised RFE-12, logistic L2  (141/143)
The second is included because its explanation is a coefficient vector, which is the cleanest
clinical statement available on this data, and because it wins on the same held-out rows.

Everything is fitted on the 426 training rows. The 143 test rows are used for scoring only;
where a procedure needs a choice (threshold, ranking) it is made by cross-validation inside
the training rows and then applied once to the test rows.

SHAP is computed exactly by xgboost's own TreeSHAP (pred_contribs / pred_interactions), so no
new dependency is introduced and the numbers are the same computation shap.TreeExplainer runs.
"""
import math
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')
warnings.simplefilter("ignore")

print("=" * 100)
print("INTERPRETATION  | sklearn %s | xgboost %s | numpy %s" % (
    sklearn.__version__, xgboost.__version__, np.__version__))
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
t_pos = [COLS.index(c) for c in TW]
print("the retained twelve: %s" % ", ".join(TW))


def xgb12():
    return XGBClassifier(objective="binary:logistic", nthread=4, seed=42)


m_ref = xgb12().fit(np.asarray(trX)[:, t_pos], ytr)
p_ref = m_ref.predict_proba(np.asarray(teX)[:, t_pos])[:, 1]
m_lin = LogisticRegression(solver="lbfgs", max_iter=2000).fit(Ztr[:, t_pos], ytr)
p_lin = m_lin.predict_proba(Zte[:, t_pos])[:, 1]
print("M_REF  %d/143   M_LIN  %d/143   (reproduces the panel rows)" % (
    int(((p_ref > 0.5).astype(int) == yte).sum()), int(((p_lin > 0.5).astype(int) == yte).sum())))

# --------------------------------------------------------------------- 1
print("\n=== 1. THE LOGISTIC MODEL IS ITS OWN EXPLANATION (standardised, RFE-12) ===")
print("one line per feature: coefficient is the log-odds change per within-cohort sd;")
print("OR is exp(coefficient); the p-value is a Wald test on the fitted model, reported")
print("descriptively - 12 tests, no multiplicity correction applied.")
b = m_lin.coef_.ravel()
pr = m_lin.predict_proba(Ztr[:, t_pos])[:, 1]
W = pr * (1 - pr)
Xd = np.column_stack([np.ones(len(ytr)), Ztr[:, t_pos]])
try:
    cov = np.linalg.inv(Xd.T @ (Xd * W[:, None]))
    se = np.sqrt(np.diag(cov))[1:]
    has_se = se.size == len(TW)
except Exception as exc:
    se, has_se = None, False
    print("  (Fisher standard errors unavailable: %s)" % exc)
for rank, j in enumerate(np.argsort(-np.abs(b)), 1):
    z = b[j] / se[j] if has_se else float("nan")
    pv = 2 * (1 - stats.norm.cdf(abs(z))) if has_se else float("nan")
    print("  %2d %-24s beta %+.3f  OR %5.2f  [%.2f, %.2f]  z %+.2f  p %.1g" % (
        rank, TW[j], b[j], math.exp(b[j]),
        math.exp(b[j] - 1.96 * se[j]) if has_se else float("nan"),
        math.exp(b[j] + 1.96 * se[j]) if has_se else float("nan"), z, pv))
print("  intercept %.3f  -> P(malignant) at the cohort mean = %.3f" % (
    m_lin.intercept_[0], 1 / (1 + math.exp(-m_lin.intercept_[0]))))

# all thirty under the same penalty, to show what the wrapper threw away
full = LogisticRegression(solver="lbfgs", max_iter=5000).fit(Ztr, ytr)
bf = full.coef_.ravel()
print("  the 18 discarded features, same pipeline, |beta| descending:")
for j in np.argsort(-np.abs(bf)):
    if COLS[j] not in TW:
        print("     %-24s beta %+.3f  OR %5.2f" % (COLS[j], bf[j], math.exp(bf[j])))

# --------------------------------------------------------------------- 2
print("\n=== 2. EACH RETAINED FEATURE ON ITS OWN (training rows) ===")
print("univariate AUC and standardised mean difference; no model between the measurement and")
print("the label, so this is the clinical signal a single measurement carries.")
print("  %-24s %6s %6s %8s %8s" % ("feature", "AUC", "d", "mean B", "mean M"))
uni = {}
for c in TW + ["concave points_worst", "perimeter_se", "area_mean"]:
    v = np.asarray(trX[c])
    a, bb = v[ytr == 0], v[ytr == 1]
    pooled = math.sqrt((bb.var(ddof=1) * (len(bb) - 1) + a.var(ddof=1) * (len(a) - 1)) /
                       (len(a) + len(bb) - 2))
    au = stats.mannwhitneyu(bb, a).statistic / (len(bb) * len(a))
    uni[c] = au
    print("  %-24s %6.3f %6.2f %8.3f %8.3f%s" % (
        c, au, (bb.mean() - a.mean()) / pooled, a.mean(), bb.mean(),
        "" if c in TW else "   (not retained)"))

# --------------------------------------------------------------------- 3
print("\n=== 3. EXACT TREE SHAP, M_REF (unscaled RFE-12 XGBoost, test rows) ===")
dm = xgboost.DMatrix(np.asarray(teX)[:, t_pos], feature_names=TW)
contrib = m_ref.get_booster().predict(dm, pred_contribs=True)
phi = contrib[:, :-1]
print("additivity check: sum(phi)+bias reproduces the margin? max |gap| %.2e" % float(
    np.max(np.abs(phi.sum(1) + contrib[:, -1] -
                  np.log(p_ref / (1 - p_ref))))))
mean_abs = np.abs(phi).mean(0)
for rank, j in enumerate(np.argsort(-mean_abs), 1):
    signed = phi[yte == 1, j].mean() - phi[yte == 0, j].mean()
    print("  %2d %-24s mean|phi| %.4f   mean phi (M) %+.4f  (B) %+.4f" % (
        rank, TW[j], mean_abs[j], phi[yte == 1, j].mean(), phi[yte == 0, j].mean()))
print("  Spearman rank correlation of this ranking with:")
print("     the logistic |beta| ranking          rho %+.3f" % stats.spearmanr(
    np.argsort(-mean_abs), np.argsort(-np.abs(b))).statistic)
retained = [c for c in COLS if rfe.support_[COLS.index(c)]]
print("     the wrapper tree's own importances   rho %+.3f" % stats.spearmanr(
    np.argsort(-mean_abs),
    np.argsort(-np.asarray([np.abs(rfe.estimator_.feature_importances_)[retained.index(c)]
                            for c in TW]))).statistic)
print("     univariate AUC of the twelve         rho %+.3f" % stats.spearmanr(
    np.argsort(-mean_abs), np.argsort(-np.array([uni[c] for c in TW]))).statistic)
weak = min(TW, key=lambda c: uni[c])
print("  a model-need ranking is not a signal ranking: %s has the smallest univariate AUC"
      % weak)
print("  (%.3f) yet ranks %dth by mean |SHAP| in M_REF." % (
    uni[weak], int(np.where(np.argsort(-mean_abs) == TW.index(weak))[0][0]) + 1))

# how much attribution mass sits on the eighteen features the wrapper dropped
m_all = xgb12().fit(np.asarray(trX), ytr)
dm_all = xgboost.DMatrix(np.asarray(teX), feature_names=COLS)
ca = m_all.get_booster().predict(dm_all, pred_contribs=True)[:, :-1]
ma = np.abs(ca).mean(0)
keep = np.array([c in TW for c in COLS])
print("  the same classifier trained on all thirty: share of mean |phi| mass on the retained")
print("  twelve %.3f, on the discarded eighteen %.3f (mass totals %.4f)" % (
    ma[keep].sum() / ma.sum(), ma[~keep].sum() / ma.sum(), ma.sum()))
top_drop = np.argsort(-ma * ~keep)[:3]
print("  largest attribution among the discarded: %s" % ", ".join(
    "%s %.4f" % (COLS[j], ma[j]) for j in top_drop))

# --------------------------------------------------------------------- 4
print("\n=== 4. INTERACTIONS: HOW MUCH OF M_REF IS NOT ADDITIVE? ===")
inter = m_ref.get_booster().predict(dm, pred_interactions=True)[:, :-1, :-1]
main_share = float(np.abs(np.diagonal(inter, axis1=1, axis2=2)).sum() / np.abs(inter).sum())
print("array shape %s ; diagonal (main effects) %.3f of total |value|, off-diagonal (pairs)"
      " %.3f" % (str(inter.shape), main_share, 1 - main_share))
agg = np.abs(inter).mean(0)
pairs = [(TW[i], TW[j], agg[i, j] + agg[j, i]) for i in range(12) for j in range(i + 1, 12)]
pairs.sort(key=lambda t: -t[2])
print("  strongest pairs on the test rows:")
for a, bb, v in pairs[:8]:
    print("     %-24s %-24s %.4f" % (a, bb, v))
print("  weakest five of the 66:")
for a, bb, v in pairs[-5:]:
    print("     %-24s %-24s %.4f" % (a, bb, v))

# --------------------------------------------------------------------- 5
print("\n=== 5. PERMUTATION IMPORTANCE, FOUR FAMILIES, SAME TWELVE (test rows) ===")
print("accuracy drop when one column is shuffled, 50 repeats, mean +/- spread over repeats.")
MODELS = [("XGBoost", lambda: xgb12().fit(np.asarray(trX)[:, t_pos], ytr)),
          ("Random forest", lambda: RandomForestClassifier(
              n_estimators=100, random_state=0).fit(np.asarray(trX)[:, t_pos], ytr)),
          ("Extra trees", lambda: ExtraTreesClassifier(
              n_estimators=500, random_state=0).fit(np.asarray(trX)[:, t_pos], ytr)),
          ("LogReg L2 (std)", lambda: LogisticRegression(solver="lbfgs", max_iter=2000)
           .fit(Ztr[:, t_pos], ytr))]
Xi = {"XGBoost": np.asarray(teX)[:, t_pos], "Random forest": np.asarray(teX)[:, t_pos],
      "Extra trees": np.asarray(teX)[:, t_pos], "LogReg L2 (std)": Zte[:, t_pos]}
perm = {}
for nm, mk in MODELS:
    fit = mk()
    r = permutation_importance(fit, Xi[nm], yte, scoring="accuracy", n_repeats=50,
                              random_state=0, n_jobs=1)
    perm[nm] = r.importances_mean
    print("  %-16s baseline %.3f ; largest drop %.4f (%s) ; smallest %.4f (%s)" % (
        nm, fit.score(Xi[nm], yte), r.importances_mean.max(),
        TW[int(np.argmax(r.importances_mean))], r.importances_mean.min(),
        TW[int(np.argmin(r.importances_mean))]))
print("  %-24s %s" % ("feature", " ".join("%-16s" % nm for nm, _ in MODELS)))
for j in range(12):
    vals = []
    for nm, _ in MODELS:
        rk = int(np.where(np.argsort(-perm[nm]) == j)[0][0]) + 1
        vals.append("%-16s" % ("#%d" % rk))
    print("  %-24s %s" % (TW[j], " ".join(vals)))
print("  agreement of each family's permutation ranking with M_REF's SHAP ranking:")
for nm, _ in MODELS:
    print("     %-16s Spearman rho %+.3f" % (
        nm, stats.spearmanr(np.abs(phi).mean(0), perm[nm]).statistic))

# --------------------------------------------------------------------- 6
print("\n=== 6. DOES THE ATTRIBUTION PREDICT WHAT THE MODEL NEEDS? (occlusion, test rows) ===")
print("remove the top-k features by mean |SHAP|, compare with 20 random k-subsets, refit-free.")
order_shap = np.argsort(-mean_abs)
print("  k  %-22s %-22s %s" % ("SHAP order acc / AUC", "random order acc / AUC", "best random"))
rng = np.random.default_rng(0)
base_acc = float(((p_ref > 0.5).astype(int) == yte).mean())
base_auc = stats.mannwhitneyu(p_ref[yte == 1], p_ref[yte == 0]).statistic / (
    (yte == 1).sum() * (yte == 0).sum())
print("  0  %-22.3f %-22.3f %.3f   (baseline, AUC %.3f)" % (base_acc, base_acc, base_acc, base_auc))
Xraw = np.asarray(teX)[:, t_pos]
mu_tr = np.asarray(trX)[:, t_pos].mean(0)
for k in (1, 2, 3, 4, 6):
    def score(cols_off, vals=None):
        O = Xraw.copy()
        O[:, cols_off] = mu_tr[cols_off]      # replace with training-row means
        return m_ref.predict_proba(O)
    ps = score(order_shap[:k])
    acc_s = float(((ps > 0.5).astype(int) == yte).sum())
    auc_s = stats.mannwhitneyu(ps[yte == 1], ps[yte == 0]).statistic / (
        (yte == 1).sum() * (yte == 0).sum())
    accs, aucs = [], []
    for _ in range(20):
        pick = rng.choice(12, size=k, replace=False)
        pr = score(pick)
        accs.append(float((pr > 0.5).astype(int).sum()))
        aucs.append(stats.mannwhitneyu(pr[yte == 1], pr[yte == 0]).statistic /
                    ((yte == 1).sum() * (yte == 0).sum()))
    print("  %d  %-22s %-22s %.3f" % (
        k, "%d / %.3f" % (acc_s, auc_s), "%.1f / %.3f" % (np.mean(accs), np.mean(aucs)),
        max(accs)))

# --------------------------------------------------------------------- 7
print("\n=== 7. PARTIAL DEPENDENCE ON M_REF (test rows, grid over the training range) ===")
for j in np.argsort(-mean_abs)[:4]:
    pd_res = partial_dependence(m_ref, Xi["XGBoost"], [j], grid_resolution=11, kind="average")
    gv, av = np.ravel(pd_res["grid_values"]), np.ravel(pd_res["average"][0])
    diffs = np.diff(av)
    mono = bool(np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9))
    lo, hi = float(av.min()), float(av.max())
    print("  %-24s avg response %.3f -> %.3f over %.2f -> %.2f ; %s ; spread %.3f" % (
        TW[j], av[0], av[-1], gv[0], gv[-1], "monotone" if mono else "non-monotone", hi - lo))

# --------------------------------------------------------------------- 8
print("\n=== 8. IS THE LIME EXPLANATION STABLE? (M_REF, the twelve, test rows) ===")
print("the manuscript shows one LIME panel. LIME perturbs the instance through a random")
print("neighbourhood, so a single draw is a sample, not a fact. num_samples is fixed at 1000")
print("and only the perturbation seed varies, on the twelve columns M_REF actually reads.")
from lime.lime_tabular import LimeTabularExplainer

X12tr, X12te = np.asarray(trX)[:, t_pos], np.asarray(teX)[:, t_pos]
expl = {}
for seed in range(5):
    le = LimeTabularExplainer(training_data=X12tr, feature_names=TW,
                              class_names=["Benign", "Malignant"], mode="classification",
                              random_state=seed)
    expl[seed] = le

cases = [i for i in range(len(yte)) if yte[i] == 1 and p_ref[i] > 0.5][:12]
raw_tops = {}
for i in cases:
    raw_tops[i] = []
    for seed in range(5):
        e = expl[seed].explain_instance(X12te[i], m_ref.predict_proba, num_features=5,
                                        num_samples=1000, labels=(1,))
        names = [cond.split(" <=")[0].split(" >")[0].strip() for cond, _ in e.as_list(label=1)]
        raw_tops[i].append(set(n for n in names if n in TW))
    inter = len(set.intersection(*raw_tops[i]))
    union = len(set.union(*raw_tops[i]))
    print("  case %3d  P(M) %.3f  top-5 over 5 seeds: %d distinct names, %d present in every "
          "draw" % (i, p_ref[i], union, inter))
pair = []
for i in cases:
    for a in range(5):
        for b in range(a + 1, 5):
            A, B = raw_tops[i][a], raw_tops[i][b]
            pair.append(len(A & B) / float(len(A | B)) if A | B else 1.0)
print("  mean pairwise Jaccard of the top-5, 10 seed pairs x %d cases: %.3f (min %.2f, "
      "cases at 1.00: %d of %d)" % (
          len(cases), float(np.mean(pair)), float(np.min(pair)),
          int(sum(1 for v in pair if v >= 0.999)), len(pair)))
glob = []
for i in cases:
    s5 = set(TW[j] for j in np.argsort(-np.abs(phi[i]))[:5])
    for seed in range(5):
        glob.append(len(s5 & raw_tops[i][seed]) / float(len(s5 | raw_tops[i][seed])))
print("  LIME top-5 against the exact SHAP top-5 of the same case: mean Jaccard %.3f over %d"
      % (float(np.mean(glob)), len(glob)))
