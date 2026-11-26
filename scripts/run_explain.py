"""Explanations for the reduced model, on the corrected protocol.

LIME panel:
  the notebook passed `feature_names=trainY` (labels, not column names) and
  `class_names=['Malignant','Benign']` (class 0 is benign, so both labels are reversed),
  and explained testX.iloc[55], which is a benign case the model calls benign with
  probability 0.995. The panel therefore read "Malignant 0.98" over a benign prediction.

SHAP:
  xgboost computes exact TreeSHAP itself (`pred_contribs=True`), so no extra dependency
  is needed and the attribution is the same computation shap.TreeExplainer performs.

The wrapper tree is seeded (random_state=0) so that the explained model is the one in
Table I of the manuscript and the subset is the one in Table II.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import xgboost
from lime.lime_tabular import LimeTabularExplainer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

raw = pd.read_csv(DATA_PATH)
data = raw.drop(["id", "Unnamed: 32", "diagnosis"], axis=1)
labels = raw["diagnosis"].map({"M": 1, "B": 0})
trX, teX, trY, teY = train_test_split(
    data, labels, random_state=42, shuffle=True, test_size=0.25)
print("xgboost %s | numpy %s" % (xgboost.__version__, np.__version__))

rfe = RFE(estimator=DecisionTreeClassifier(random_state=0), n_features_to_select=12)
pipe = Pipeline(steps=[("fs", rfe), ("model", XGBClassifier(
    objective="binary:logistic", nthread=4, seed=42))])
pipe.fit(trX, trY)
sel = list(data.columns[rfe.get_support()])
print("selected (%d): %s" % (len(sel), ", ".join(sel)))

prob = pipe.predict_proba(teX)
pred = pipe.predict(teX)
print("correct: %d/143" % int((pred == teY.values).sum()))

# --------------------------------------------------------------- the instance
mal_ok = [i for i in range(len(teY)) if teY.iloc[i] == 1 and pred[i] == 1]
conf = np.array([prob[i, 1] for i in mal_ok])
median = float(np.median(conf))
k = int(np.argmin(np.abs(conf - median)))
i = mal_ok[k]
print("\n=== INSTANCE test position %d ===" % i)
print("true label %d (malignant); P(malignant) %.4f, P(benign) %.4f" % (
    teY.iloc[i], prob[i, 1], prob[i, 0]))
print("confidence of the %d correctly classified malignant cases: %.3f - %.3f, median %.3f"
      % (len(mal_ok), conf.min(), conf.max(), median))

# --------------------------------------------------------------------- LIME
explainer = LimeTabularExplainer(
    training_data=np.asarray(trX),
    feature_names=list(data.columns),
    class_names=["Benign", "Malignant"],
    mode="classification",
    random_state=42,
)
exp = explainer.explain_instance(
    np.asarray(teX.iloc[i]), pipe.predict_proba, num_features=8, labels=(1,))
print("\n=== LIME ===")
pairs = exp.as_list(label=1)
for cond, w in pairs:
    print("  %-52s %+.4f" % (cond, w))

# --------------------------------------------------------------------- SHAP
booster = pipe.named_steps["model"].get_booster()
dmat = xgboost.DMatrix(teX[sel])
contrib = booster.predict(dmat, pred_contribs=True)   # rows x (features + bias)
shap_vals = contrib[:, :-1]
print("\n=== SHAP (exact TreeSHAP via pred_contribs) ===")
print("bias term: %.4f (constant for every row)" % contrib[0, -1])
mean_abs = np.abs(shap_vals).mean(axis=0)
order = np.argsort(-mean_abs)
print("global ranking by mean absolute contribution:")
for rank, j in enumerate(order, 1):
    print("  %2d %-24s %.4f" % (rank, sel[j], mean_abs[j]))
print("\nsingle case, same instance, contributions to the log-odds margin:")
row = shap_vals[i]
for j in np.argsort(-np.abs(row)):
    print("  %-24s %+.4f" % (sel[j], row[j]))
print("sum of contributions %.4f + bias %.4f = margin %.4f" % (
    row.sum(), contrib[0, -1], row.sum() + contrib[0, -1]))

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 6.0, "axes.labelsize": 6.0,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "font.family": "serif",
})

names = [c.replace("_", " ") for c, _ in pairs][::-1]
vals = [w for _, w in pairs][::-1]
colors = ["#b2182b" if v > 0 else "#2166ac" for v in vals]
fig, ax = plt.subplots(figsize=(3.45, 2.05))
ax.barh(range(len(vals)), vals, color=colors, height=0.66)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.axvline(0.0, color="#333333", linewidth=0.7)
ax.set_xlabel("LIME weight toward the malignant class")
ax.tick_params(length=2, pad=1.5)
ax.xaxis.grid(True, color="#dddddd", linewidth=0.5)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout(pad=0.15)
out = os.path.join(HERE, "paper1_conf", "figures", "fig_lime.png")
fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.02)
print("\nwritten %s" % out)
