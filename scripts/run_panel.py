"""Extended classifier panel for the conference paper, in two scaling arms.

Why two arms: _p1_lr.py showed the notebook's unscaled logistic fit never converges, and
every distance-, kernel- and penalty-based model on this list reads the native scales
wrongly. So each configuration is run (a) unscaled, faithful to the notebook, and (b)
standardised with the scaler fitted on the 426 training rows only.

Each configuration is also run on the twelve features that the seeded RFE wrapper retains,
which tests whether "the reduction costs nothing" is a property of XGBoost or of every family.

Nothing here is a regression, time-series or sequential model: the target is the binary
diagnosis label, so those families have no legitimate target on this file. See section 0 of
the log for the record of what was excluded and why.
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
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')

print("=" * 96)
print("PANEL  | xgboost %s | numpy %s | sklearn %s" % (
    xgboost.__version__, np.__version__, sklearn.__version__))
print("=" * 96)

print("""
=== 0. WHAT IS NOT RUN, AND WHY ===
Regression families (simple/multiple/polynomial/ridge/lasso/elastic-net/Bayesian/quantile
regression, tree and ensemble regressors, SVR, KNN regressor, LOWESS, GP regression): the
target is `diagnosis`, a binary label. There is no continuous outcome in data.csv, and
mapping M/B to 1/0 to feed a least-squares fit would be a linear-probability model presented
as a classifier - an invented target, not a measured one. Not run.
Time-series families (AR, MA, ARMA, ARIMA, SARIMA, SARIMAX, VAR, VECM, exponential
smoothing, Holt, Holt-Winters, Prophet, DeepAR, N-BEATS, N-HiTS): WDBC rows are one
measurement set per patient, with no time index and no ordering carrying meaning. The csv
row order is alphabetical by identifier. Any autoregressive structure fitted to it would be
an artefact of file layout. Not run. statsmodels is also not installed here.
Sequential deep models (RNN, LSTM, GRU, TCN, Transformer, TFT): same objection, plus 426
training rows. Not run.
Generative deep models (VAE, GAN, conditional GAN, diffusion, normalising flows): 569 rows,
30 features, no density-based claim the paper needs. Not run. The legitimate generative
content is already classical and testable: LDA and QDA are Gaussian class-conditional
density classifiers, and a two-component Gaussian mixture is the soft-clustering analogue of
fuzzy c-means (scikit-fuzzy is not installed).
Probabilistic sequence models (Bayesian networks, HMM with continuous emissions, Kalman and
particle filters): pgmpy/statsmodels not installed, and no sequence exists to filter. Not run.
Kernel PCA, PCA, Factor Analysis, ICA, t-SNE and the clustering panel are run in
_p1_reduct.py. LightGBM, CatBoost, UMAP and shap are not installed in either local stack.

Two column conventions in the tables below. Brier, log-loss and ECE are printed only where
the model really emits a probability: SVC, NuSVC and NearestCentroid do not at their default
settings, so their row shows n/a in those three columns and the AUC is marked as computed
from the decision margin. And "[it n]" is printed only for the iterative solvers where that
number means solver iterations; libsvm's n_iter_ counts dual-objective sweeps (it printed
6,265,550 for one linear SVC) and is not comparable.
One compatibility shim is used and is not a modelling choice: xgboost 1.7.2 declares
_estimator_type on the instance, so sklearn 1.7.2's tag-based is_classifier() rejects
XGBClassifier inside VotingClassifier. The voting member is a subclass that declares the same
attribute at class level. No hyper-parameter and no fitted value changes.""")

raw = pd.read_csv(DATA_PATH)
data = raw.drop([c for c in ("id", "Unnamed: 32", "diagnosis") if c in raw.columns], axis=1)
labels = raw["diagnosis"].map({"M": 1, "B": 0})
COLS = list(data.columns)
print("file %s | %d x %d | benign %d / malignant %d" % (
    os.path.basename(CANDIDATES[0]), data.shape[0], data.shape[1],
    int((labels == 0).sum()), int((labels == 1).sum())))


def wilson(k, n, z=1.959963984540054):
    p = k / float(n)
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def ece(p, y, bins=5):
    """Expected calibration error, equal-width bins on [0, 1]. sklearn's calibration_curve
    is not exposed in this build, so the bins are computed directly."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot, err = 0.0, 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        tot += m.sum()
        err += m.sum() * abs(p[m].mean() - y[m].mean())
    return err / tot if tot else float("nan")


def metrics(y_true, pred, proba, score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = int(tn + fp + fn + tp)
    k = int(tn + tp)
    sens = tp / float(tp + fn)
    spec = tn / float(tn + fp)
    prec = tp / float(tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    lo, hi = wilson(k, n)
    out = dict(k=k, n=n, fn=int(fn), fp=int(fp), sens=sens, spec=spec, prec=prec,
               f1=f1, lo=lo, hi=hi, auc=float("nan"), brier=float("nan"),
               ll=float("nan"), cal=float("nan"))
    # AUC may use a probability or any monotone margin; the three calibration columns may
    # only use something that is actually on [0, 1].
    ranking = proba if proba is not None else score
    if ranking is None:
        out["auc"] = roc_auc_score(y_true, pred)
    else:
        out["auc"] = roc_auc_score(y_true, ranking)
    if proba is not None:
        p = np.asarray(proba, dtype=float).ravel()
        out["brier"] = float(np.mean((p - np.asarray(y_true, dtype=float)) ** 2))
        out["ll"] = log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))
        out["cal"] = ece(p, np.asarray(y_true, dtype=float))
    return out


HEAD = ("%-26s %5s %5s %5s %4s %4s %6s %6s %6s %6s" %
        ("model", "corr", "sens", "spec", "FN", "FP", "AUC", "Brier", "logL", "ECE"))


def fmt(v):
    return "   n/a " if v is None or (isinstance(v, float) and math.isnan(v)) else "%6.3f" % v


def show(name, m, extra=""):
    print("%-26s %3d/%-3d %5.3f %5.3f %4d %4d %s %s %s %s %s" % (
        name, m["k"], m["n"], m["sens"], m["spec"], m["fn"], m["fp"],
        fmt(m["auc"]), fmt(m["brier"]), fmt(m["ll"]), fmt(m["cal"]), extra))


def xgb():
    return XGBClassifier(objective="binary:logistic", nthread=4, seed=42)


class XGBc(XGBClassifier):
    """XGBClassifier that sklearn 1.7 will accept inside an ensemble.

    xgboost 1.7.2 sets _estimator_type on the instance, so sklearn's tag-based
    is_classifier() reads the class attribute, finds nothing, and VotingClassifier rejects
    it. Declaring it at class level changes no hyper-parameter and no computation; it only
    lets the same fitted object be a voting member. Logged, not hidden.
    """

    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        t = super().__sklearn_tags__()
        t.estimator_type = "classifier"
        return t


def proba_of(clf, X):
    """Return (probabilities, decision_scores). SVC/NuSVC/NearestCentroid have no calibrated
    probability at their default settings, so their margin goes in the score slot and the
    Brier, log-loss and ECE columns are left empty rather than computed on a non-probability."""
    p, s = None, None
    if hasattr(clf, "predict_proba"):
        try:
            p = np.asarray(clf.predict_proba(X), dtype=float)[:, 1]
        except Exception:
            p = None
    if hasattr(clf, "decision_function"):
        try:
            s = np.asarray(clf.decision_function(X), dtype=float).ravel()
        except Exception:
            s = None
    return p, s


def panel(scaled):
    """The configuration list. `scaled` selects the arm; the scaler is fitted on train rows."""
    return [
        # the six the paper already reports, kept identical so the panel contains them
        ("KNN k=20", lambda: KNeighborsClassifier(n_neighbors=20)),
        ("SVM linear", lambda: SVC(kernel="linear")),
        ("LogReg L2 (lbfgs)", lambda: LogisticRegression(solver="lbfgs")),
        ("MLP default", lambda: MLPClassifier()),
        # the notebook passes no random_state, so its MLP row is not bit-reproducible; the
        # seeded twin is here to show what a reproducible call of the same shape gives
        ("MLP (seed 0)", lambda: MLPClassifier(random_state=0)),
        ("Random Forest 100", lambda: RandomForestClassifier(n_estimators=100,
                                                             random_state=0)),
        ("XGBoost", xgb),
        # regularised linear models: the classifier analogues of ridge / lasso / elastic net
        ("LogReg L1 (liblinear)", lambda: LogisticRegression(penalty="l1", solver="liblinear")),
        ("LogReg EN (saga)", lambda: LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5)),
        # class-conditional Gaussian densities: the generative entries that are legitimate here
        ("LDA (svd)", lambda: LinearDiscriminantAnalysis()),
        ("LDA (shrinkage)", lambda: LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto")),
        ("QDA", lambda: QuadraticDiscriminantAnalysis()),
        # probabilistic, non-parametric, at its library default kernel
        ("Gaussian Process clf", lambda: GaussianProcessClassifier()),
        # tree and boosting family members the paper does not yet carry
        ("Decision tree", lambda: DecisionTreeClassifier(random_state=0)),
        ("Extra trees", lambda: ExtraTreesClassifier(n_estimators=500, random_state=0)),
        ("GradBoost sklrn", lambda: GradientBoostingClassifier(random_state=0)),
        ("AdaBoost", lambda: AdaBoostClassifier(random_state=0)),
        # support vector variants and a distance-free baseline
        ("NuSVC", lambda: NuSVC()),
        ("SVM rbf", lambda: SVC(kernel="rbf")),
        ("Nearest centroid", lambda: NearestCentroid()),
        # one ensemble of the existing members, to test whether the panel buys anything
        ("Vote RF+XGB+LDA", lambda: VotingClassifier(
            [("rf", RandomForestClassifier(n_estimators=100, random_state=0)),
             ("xg", XGBc(objective="binary:logistic", nthread=4, seed=42)),
             ("lda", LinearDiscriminantAnalysis())], voting="soft")),
    ]


trX, teX, trY, teY = train_test_split(
    data, labels, random_state=42, shuffle=True, test_size=0.25)
print("\ntest partition: %d cases (%d benign / %d malignant)" % (
    len(teY), int((teY == 0).sum()), int((teY == 1).sum())))

results = {}
for arm in ["native", "scaled"]:
    print("\n" + "=" * 96)
    print("=== ARM: %s ===" % arm)
    print("=" * 96)
    if arm == "scaled":
        sc = StandardScaler().fit(trX)
        Xtr, Xte = sc.transform(trX), sc.transform(teX)
    else:
        Xtr, Xte = trX, teX

    # the retained twelve, refitted inside this arm's training rows only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfe = RFE(estimator=DecisionTreeClassifier(random_state=0),
                  n_features_to_select=12).fit(Xtr, trY)
    twelve = [COLS[i] for i, v in enumerate(rfe.support_) if v]
    twelve.sort()
    print("RFE-12 in this arm: %s" % ", ".join(twelve))
    keep = [c for c in COLS if c in twelve]

    for tag, which in [("full30", COLS), ("rfe12", keep)]:
        print("\n--- %s / %s, %d features ---" % (arm, tag, len(which)))
        print(HEAD)
        rows = {}
        # integer positions: Xtr/Xte are numpy arrays in the scaled arm
        pos = [COLS.index(c) for c in which]
        if isinstance(Xtr, np.ndarray):
            Xa, Xb = Xtr[:, pos], Xte[:, pos]
        else:
            Xa, Xb = Xtr.iloc[:, pos], Xte.iloc[:, pos]
        for name, ctor in panel(arm == "scaled"):
            t0 = time.time()
            note = ""
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    clf = ctor()
                    clf.fit(Xa, trY)
                    pred = np.asarray(clf.predict(Xb)).astype(int)
                    pb, ps = proba_of(clf, Xb)
                cw = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
                if cw:
                    note += " [not converged]"
                # n_iter_ only means "solver iterations" for the iterative solvers; for libsvm
                # it counts dual-objective sweeps and reaches seven figures without meaning it
                if isinstance(clf, (LogisticRegression, MLPClassifier)):
                    ni = getattr(clf, "n_iter_", None)
                    if ni is not None:
                        note += " [it %s]" % (int(np.ravel(ni)[0])
                                              if hasattr(ni, "__len__") else ni)
                if pb is None and ps is not None:
                    note += " [AUC from margin]"
                if pb is None and ps is None:
                    note += " [AUC from hard labels]"
            except Exception as exc:
                print("%-26s FAILED %s: %s" % (name, type(exc).__name__, str(exc)[:60]))
                continue
            m = metrics(teY, pred, pb, ps)
            m["secs"] = time.time() - t0
            rows[name] = m
            show(name, m, note)
        results[(arm, tag)] = rows

with open(os.path.join(HERE, "..", "logs", "panel_output.txt"), "w") as fh:
    import json
    json.dump({"%s/%s" % k: v for k, v in results.items()}, fh, indent=1, default=float)
print("\nsaved: logs/panel_output.txt")
