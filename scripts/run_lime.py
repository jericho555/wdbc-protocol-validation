"""Regenerate the LIME panel.

The published figure has three defects, all traceable to two lines in the notebook:

  feature_names=trainY                 -- the 0/1 labels were passed as feature names,
                                          which is why the panel has no feature names in it
  class_names=['Malignant','Benign']   -- class 0 is benign, so every label is reversed

and the instance it explains, testX.iloc[55], is a benign case that the model calls benign
with probability 0.995, so the panel reports a confident benign prediction under the
heading "Malignant 0.98".

This script fits the same pipeline (RFE(DT, 12) -> XGBoost, seed 42) on the same split,
explains a correctly classified malignant case, and writes the panel with the real
feature names and the real class order.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer

DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')


def find_csv():
    for path in CANDIDATES:
        if os.path.exists(path):
            return path
    print("data.csv not found. Looked in:")
    for path in CANDIDATES:
        print("  " + path)
    sys.exit(2)


def main():
    print("sklearn %s | xgboost %s | numpy %s" % (
        sklearn.__version__, xgboost.__version__, np.__version__))

    raw = pd.read_csv(find_csv())
    drop = [c for c in ("id", "Unnamed: 32", "diagnosis") if c in raw.columns]
    data = raw.drop(drop, axis=1)
    labels = raw["diagnosis"].map({"M": 1, "B": 0})

    trainX, testX, trainY, testY = train_test_split(
        data, labels, random_state=42, shuffle=True, test_size=0.25)
    print("train / test    : %d / %d" % (len(trainY), len(testY)))

    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=12)
    xgb = XGBClassifier(objective="binary:logistic", nthread=4, seed=42)
    pipe = Pipeline(steps=[("Feature Selection", rfe), ("model", xgb)])
    pipe.fit(trainX, trainY)

    selected = list(data.columns[rfe.get_support()])
    print("selected (%d)    : %s" % (len(selected), ", ".join(selected)))

    prob = pipe.predict_proba(testX)
    pred = pipe.predict(testX)
    correct = int((pred == testY.values).sum())
    print("correct         : %d/%d" % (correct, len(testY)))

    # ------------------------------------------------ choose the instance
    # A correctly classified malignant case at the median confidence among
    # the correctly classified malignant cases: a typical positive, not the
    # most confident one, and not an arbitrary row index.
    mal_correct = [i for i in range(len(testY))
                   if testY.iloc[i] == 1 and pred[i] == 1]
    conf = np.array([prob[i, 1] for i in mal_correct])
    median = float(np.median(conf))
    k = int(np.argmin(np.abs(conf - median)))
    i = mal_correct[k]
    print("\n=== INSTANCE ===")
    print("correct malignant cases : %d of %d" % (len(mal_correct), int((testY == 1).sum())))
    print("confidence range        : %.4f - %.4f, median %.4f" % (
        conf.min(), conf.max(), median))
    print("chosen test position    : %d" % i)
    print("true label              : %d (malignant)" % testY.iloc[i])
    print("P(benign)               : %.4f" % prob[i, 0])
    print("P(malignant)            : %.4f" % prob[i, 1])
    print("predicted label         : %d" % pred[i])
    print("values:")
    for name in data.columns:
        print("  %-24s %.6f" % (name, testX.iloc[i][name]))

    # --------------------------------------------------------------- LIME
    explainer = LimeTabularExplainer(
        training_data=np.array(trainX),
        feature_names=list(data.columns),
        class_names=["Benign", "Malignant"],
        mode="classification",
        random_state=42,
    )
    exp = explainer.explain_instance(
        np.array(testX.iloc[i]), pipe.predict_proba, num_features=8, labels=(1,))

    print("\n=== LIME, class 1 (malignant) ===")
    print("surrogate predict_proba : benign %.4f  malignant %.4f" % tuple(exp.predict_proba))
    for cond, w in exp.as_list(label=1):
        print("  %-58s %+.4f" % (cond, w))

    # ------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 6.5,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "font.family": "serif",
    })

    pairs = exp.as_list(label=1)
    names = [c for c, _ in pairs][::-1]
    vals = [w for _, w in pairs][::-1]
    colors = ["#b2182b" if v > 0 else "#2166ac" for v in vals]

    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    ax.barh(range(len(vals)), vals, color=colors, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(0.0, color="#333333", linewidth=0.7)
    ax.set_xlabel("LIME weight toward malignant")
    ax.tick_params(axis="both", length=2, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)

    out = os.path.join(HERE, "paper1_conf", "figures", "fig_lime.png")
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    print("\nwritten: %s" % out)

    # Also report what the published panel was showing, for the record.
    j = 55
    print("\n=== PUBLISHED INSTANCE testX.iloc[55] ===")
    print("true label    : %d (%s)" % (testY.iloc[j], "malignant" if testY.iloc[j] else "benign"))
    print("P(benign)     : %.4f" % prob[j, 0])
    print("P(malignant)  : %.4f" % prob[j, 1])
    print("predicted     : %d" % pred[j])


if __name__ == "__main__":
    main()
