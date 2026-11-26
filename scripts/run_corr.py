"""Correlation facts about the 30 WDBC measurements.

Replaces the correlation figure with numbers: the figure is being dropped for space,
so the claim it supports has to be stated with the values that produced it.
"""
import os

import pandas as pd

raw = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper 2", "data.csv"))
data = raw.drop(["id", "Unnamed: 32", "diagnosis"], axis=1)
corr = data.corr().abs()

pairs = []
cols = list(data.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pairs.append((corr.iloc[i, j], cols[i], cols[j]))
pairs.sort(reverse=True)

print("feature pairs with |r| >= 0.9: %d of %d" % (
    sum(1 for v, _, _ in pairs if v >= 0.9), len(pairs)))
print("feature pairs with |r| >= 0.95: %d" % sum(1 for v, _, _ in pairs if v >= 0.95))
print("\ntwenty most correlated pairs:")
for v, a, b in pairs[:20]:
    print("  %.3f  %-24s %s" % (v, a, b))

print("\nmean |r| among the three size descriptors, worst form:")
for a, b in (("radius_worst", "perimeter_worst"), ("radius_worst", "area_worst"),
             ("perimeter_worst", "area_worst")):
    print("  %s / %s : %.3f" % (a, b, corr.loc[a, b]))

counts = (corr >= 0.9).sum(axis=1) - 1
print("\nfor each feature, how many of the other 29 it correlates with at |r| >= 0.9:")
for name, n in counts.sort_values(ascending=False).items():
    print("  %-24s %2d" % (name, n))
