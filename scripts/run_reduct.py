"""Redundancy, unsupervised reduction, clustering, and the one legitimate regression study.

Companion to _p1_panel.py. Everything is fitted on the 426 training rows; the 143 test rows
are used only for the held-out accuracy columns, and the redundancy regressions never touch
them at all (5-fold CV inside the training rows).

The regression half answers the only question on this data that has a continuous target: can
one measurement be reconstructed from the other twenty-nine. That is a redundancy statistic,
not a clinical model, and the log says so.
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
from sklearn.cluster import (AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans, MeanShift,
                             SpectralClustering)
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.feature_selection import RFE
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, log_loss,
                             normalized_mutual_info_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')

print("=" * 100)
print("REDUNDANCE / REDUCTION / CLUSTERS  | numpy %s | sklearn %s" % (
    np.__version__, sklearn.__version__))
print("=" * 100)

raw = pd.read_csv(DATA_PATH)
data = raw.drop([c for c in ("id", "Unnamed: 32", "diagnosis") if c in raw.columns], axis=1)
labels = raw["diagnosis"].map({"M": 1, "B": 0})
COLS = list(data.columns)
trX, teX, trY, teY = train_test_split(
    data, labels, random_state=42, shuffle=True, test_size=0.25)
sc = StandardScaler().fit(trX)
Ztr, Zte = sc.transform(trX), sc.transform(teX)
Zall = sc.transform(data)          # scaler fitted on training rows only
ytr = np.asarray(trY)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rfe = RFE(estimator=DecisionTreeClassifier(random_state=0),
              n_features_to_select=12).fit(trX, trY)
TW = sorted(c for c, v in zip(COLS, rfe.support_) if v)
print("the retained twelve: %s\n" % ", ".join(TW))

# --------------------------------------------------------------------- 1
print("=== 1. THE SPECTRUM OF THE 30 FEATURE CORRELATION MATRIX (training rows) ===")
R = np.corrcoef(Ztr, rowvar=False)
ev = np.sort(np.linalg.eigvalsh(R))[::-1]
ev = np.clip(ev, 0, None)
cum = np.cumsum(ev) / ev.sum()
kaiser = int((ev >= 1.0).sum())
print("eigenvalues >= 1 (Kaiser): %d of 30" % kaiser)
for k in (1, 2, 3, 5, 8, 10, 12, 16, 20, 30):
    print("  first %2d components carry %5.1f%% of the variance" % (k, 100 * cum[k - 1]))
n90 = int(np.searchsorted(cum, 0.90) + 1)
n95 = int(np.searchsorted(cum, 0.95) + 1)
print("  90%% of variance needs %d components, 95%% needs %d" % (n90, n95))
pairs = int(((np.abs(R - np.eye(len(R))) >= 0.9).sum()) // 2)
print("  largest off-diagonal |r|: %.3f ; %d of the 435 pairs at |r| >= 0.9" % (
    float(np.abs(R - np.diag(np.diag(R))).max()), pairs))

# how much of the class signal sits on each component
pcs = PCA(n_components=30).fit(Ztr)
S = pcs.transform(Ztr)
r_pb = [stats.pointbiserialr(ytr, S[:, j]).statistic for j in range(30)]
order = np.argsort(-np.abs(r_pb))
print("\ncomponents most associated with the label (training rows, point-biserial r):")
for j in order[:5]:
    auc = roc_auc_score(ytr, S[:, j])
    print("  PC%-2d  r %+.3f  AUC %.3f  variance %.1f%%" % (
        j + 1, r_pb[j], max(auc, 1 - auc), 100 * pcs.explained_variance_ratio_[j]))
fa = FactorAnalysis(n_components=kaiser, random_state=0).fit(Ztr)
recon = fa.transform(Ztr) @ fa.components_
communality = float(np.mean(np.var(recon, axis=0) / np.var(Ztr, axis=0)))
print("FactorAnalysis at %d factors reproduces %.3f of each variable's variance on average"
      % (kaiser, communality))

# --------------------------------------------------------------------- 1b
print("\n=== 1b. THE OTHER REDUCTIONS THE PANEL LOG PROMISES: ICA, KERNEL PCA, t-SNE ===")
ica = FastICA(n_components=kaiser, random_state=0, max_iter=2000).fit(Ztr)
A = ica.transform(Ztr)
r_ica = [stats.pointbiserialr(ytr, A[:, j]).statistic for j in range(A.shape[1])]
print("independent components, ranked by association with the label (training rows):")
for j in np.argsort(-np.abs(r_ica))[:4]:
    auc = roc_auc_score(ytr, A[:, j])
    print("  IC%-2d r %+.3f  AUC %.3f" % (j + 1, r_ica[j], max(auc, 1 - auc)))
print("  most label-associated IC is IC%d; the components are ordered by convergence, not by "
      "variance, so no ranking of this list is implied" % (int(np.argmax(np.abs(r_ica))) + 1))
try:
    kp = KernelPCA(n_components=kaiser, kernel="rbf",
                   gamma=1.0 / Ztr.shape[1], random_state=0).fit(Ztr)
    Kt = kp.transform(Ztr)
    evk = np.linalg.eigvalsh(np.cov(Kt, rowvar=False))
    print("kernel PCA (rbf, gamma=1/p) on the training rows: %d components, variance share of "
          "the first %.3f" % (kaiser, float(np.max(evk) / np.sum(np.clip(evk, 0, None)))))
    r_kp = [stats.pointbiserialr(ytr, Kt[:, j]).statistic for j in range(Kt.shape[1])]
    print("  strongest label association among them: |r| %.3f" % float(np.max(np.abs(r_kp))))
    print("  NOTE: an rbf kernel PCA has no exact out-of-sample map (the pre-image problem), so")
    print("  it is used here to describe the training rows only and is not scored on the 143.")
except Exception as exc:
    print("kernel PCA failed: %s: %s" % (type(exc).__name__, exc))

# t-SNE: the axes carry no meaning, so the honest question is what the 2-D map preserves
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

for perp in (10, 30, 50):
    try:
        Y = TSNE(n_components=2, perplexity=perp, random_state=0, max_iter=1000).fit_transform(Ztr)
        for tag, M in ("30-d standardised", Ztr), ("t-SNE 2-d (perp %d)" % perp, Y):
            nn = NearestNeighbors(n_neighbors=2).fit(M)
            d, i = nn.kneighbors(M)
            agree = float(np.mean(ytr[i[:, 1]] == ytr))   # leave-self-out 1-NN
            print("  1-nearest-neighbour label agreement in %-26s : %.3f" % (tag, agree))
    except Exception as exc:
        print("  t-SNE perplexity %d failed: %s" % (perp, exc))
print("  (t-SNE distances are not metric, so this compares neighbourhoods, not distances; it is")
print("   reported as a check on what the picture keeps, not as a classifier)")

# --------------------------------------------------------------------- 2
print("\n=== 2. SUPERVISED REDUCTION (RFE) AGAINST UNSUPERVISED REDUCTION (PCA), TEST ROWS ===")
print("held-out cases correct out of 143, XGBoost on top, classifier fixed:")
print("  %-22s %6s %6s %6s" % ("representation", "XGB", "LogReg", "LDA"))
pca_full = PCA().fit(Ztr)


def arm(cols=None, npca=None, nica=None):
    if nica is not None:
        return ica.transform(Ztr)[:, :nica], ica.transform(Zte)[:, :nica]
    if npca is not None:
        return pca_full.transform(Ztr)[:, :npca], pca_full.transform(Zte)[:, :npca]
    idx = [COLS.index(c) for c in cols]
    return Ztr[:, idx], Zte[:, idx]


def score(A, B, mk):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = mk()
        m.fit(A, ytr)
        p = np.asarray(m.predict(B)).astype(int)
    return int((p == np.asarray(teY)).sum())


for tag, kw in [("all 30 features", dict(cols=COLS)),
                ("RFE-12 (supervised)", dict(cols=TW)),
                ("PCA-2 components", dict(npca=2)),
                ("PCA-%d (90%% var)" % n90, dict(npca=n90)),
                ("PCA-5 components", dict(npca=5)),
                ("PCA-%d (95%% var)" % n95, dict(npca=n95)),
                ("PCA-12 components", dict(npca=12)),
                ("PCA-16", dict(npca=16)),
                ("PCA-20", dict(npca=20)),
                ("ICA-%d (Kaiser)" % kaiser, dict(nica=kaiser)),
                ("ICA-1 component", dict(nica=1))]:
    A, B = arm(**kw)
    row = []
    for mk in [lambda: XGBClassifier(objective="binary:logistic", nthread=4, seed=42),
               lambda: LogisticRegression(max_iter=2000),
               lambda: LinearDiscriminantAnalysis()]:
        row.append("%6d" % score(A, B, mk))
    print("  %-22s %s" % (tag, " ".join(row)))

# --------------------------------------------------------------------- 3
print("\n=== 3. DO THE TWELVE SELECT LIKE RFE DOES? (penalty-based selection, scaled) ===")
for C in (0.03, 0.1, 0.3, 1.0):
    for pen, kw in [("l1", dict(penalty="l1", solver="liblinear")),
                    ("elasticnet", dict(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                        max_iter=20000))]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(C=C, **kw).fit(Ztr, ytr)
        nz = [COLS[i] for i in range(30) if abs(m.coef_.ravel()[i]) > 1e-8]
        inter = len(set(nz) & set(TW))
        print("  C=%-5s %-10s keeps %2d features, %2d of them in the RFE twelve" % (
            C, pen, len(nz), inter))
imp = np.abs(rfe.estimator_.feature_importances_)
retained = [c for c in COLS if rfe.support_[COLS.index(c)]]
print("  (RFE's wrapper tree importances, retained twelve: min %.4f, max %.4f)" % (
    min(imp[retained.index(c)] for c in TW), max(imp[retained.index(c)] for c in TW)))

# --------------------------------------------------------------------- 4
print("\n=== 4. UNSUPERVISED STRUCTURE: DOES CLUSTERING FIND THE DIAGNOSIS? ===")
print("two-group partitions of the 426 training cases, scored against the true labels")
print("  %-24s %6s %6s %8s %8s" % ("method", "ARI", "NMI", "silh(k=2)", "largest purity"))


def clus_scores(Z, lab, name, pred):
    ari = adjusted_rand_score(lab, pred)
    nmi = normalized_mutual_info_score(lab, pred)
    k = len(set(pred))
    sil = silhouette_score(Z, pred) if 1 < k < len(Z) else float("nan")
    cm = confusion_matrix(lab, pred, labels=sorted(set(lab)))
    purity = float(np.sum(cm.max(axis=1)) / cm.sum()) if cm.size else float("nan")
    print("  %-24s %6.3f %6.3f %8.3f %8.3f" % (name, ari, nmi, sil, purity))


TWO = [("30 features", Ztr), ("12 retained", Ztr[:, [COLS.index(c) for c in TW]])]
for feat, Z in TWO:
    print("  -- %s --" % feat)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Z)
    clus_scores(Z, ytr, "K-means k=2", km.labels_)
    gm = GaussianMixture(2, random_state=0, n_init=5).fit(Z)
    clus_scores(Z, ytr, "Gaussian mixture k=2", gm.predict(Z))
    ag = AgglomerativeClustering(n_clusters=2).fit(Z)
    clus_scores(Z, ytr, "Ward k=2", ag.labels_)
    sp = SpectralClustering(n_clusters=2, random_state=0, affinity="nearest_neighbors").fit(Z)
    clus_scores(Z, ytr, "Spectral k=2", sp.labels_)
    ms = MeanShift().fit(Z)
    clus_scores(Z, ytr, "Mean shift (default h)", ms.labels_)
    db = DBSCAN(eps=float(np.median(km.transform(Z).min(axis=1)))).fit(Z)
    clus_scores(Z, ytr, "DBSCAN (median-heuristic)", db.labels_)
    hd = HDBSCAN().fit(Z)
    clus_scores(Z, ytr, "HDBSCAN (default)", hd.labels_)

# how far apart the true classes actually are, per representation
def separation(cols):
    idx = [COLS.index(c) for c in cols]
    a, b = Ztr[ytr == 0][:, idx], Ztr[ytr == 1][:, idx]
    d = a.mean(0) - b.mean(0)
    S = np.cov(np.vstack([a, b]).T)
    mahal = float(np.sqrt(d @ np.linalg.pinv(S) @ d))
    return mahal


print("  Mahalanobis separation between the true classes: 30 features %.2f, 12 retained %.2f"
      % (separation(COLS), separation(TW)))

# --------------------------------------------------------------------- 5
print("\n=== 5. REDUNDANCE AS A REGRESSION STUDY (the only continuous targets this file has) ===")
print("predict one measured quantity from the other 29, inside the 426 training rows only.")
print("5-fold CV R^2. This measures how much the measurements duplicate each other; it is not")
print("a diagnostic model and is reported nowhere as one.")
kf = KFold(5, shuffle=True, random_state=0)
REGS = [
    ("linear (OLS)", LinearRegression()),
    ("ridge L2", Ridge()),
    ("lasso L1", Lasso(max_iter=20000)),
    ("elastic net", ElasticNet(max_iter=20000)),
    ("Bayesian ridge", BayesianRidge()),
    ("SVR linear", SVR(kernel="linear")),
    ("SVR rbf", SVR(kernel="rbf")),
    ("nu-SVR", NuSVR()),
    ("kNN regressor", KNeighborsRegressor(5)),
    ("decision tree", DecisionTreeRegressor(random_state=0)),
    ("random forest", RandomForestRegressor(200, random_state=0, n_jobs=2)),
    ("extra trees", ExtraTreesRegressor(200, random_state=0, n_jobs=2)),
    ("gradient boosting", GradientBoostingRegressor(random_state=0)),
    ("PLS (2 comps)", PLSRegression(n_components=2)),
    ("GP regressor", GaussianProcessRegressor(alpha=1e-6, normalize_y=True)),
]
for target in ["radius_worst", "concave points_worst", "area_se", "texture_worst",
               "fractal_dimension_se"]:
    i = COLS.index(target)
    X = np.delete(Zall, i, axis=1)
    t = np.asarray(data[target])
    print("  target %-20s (sd %.2f after scaling the inputs)" % (target, float(np.std(t))))
    for name, est in REGS:
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r2 = cross_val_score(est, X, t, cv=kf, scoring="r2", n_jobs=2)
            print("     %-18s R2 mean %+.3f  min %+.3f  max %+.3f  (%.1fs)" % (
                name, float(np.mean(r2)), float(np.min(r2)), float(np.max(r2)),
                time.time() - t0))
        except Exception as exc:
            print("     %-18s FAILED %s: %s" % (name, type(exc).__name__, str(exc)[:60]))
    poly = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=10.0))
    r2 = cross_val_score(poly, X, t, cv=kf, scoring="r2", n_jobs=2)
    print("     %-18s R2 mean %+.3f  (polynomial degree 2, 465 terms)" % ("poly2 ridge",
                                                                          float(np.mean(r2))))
