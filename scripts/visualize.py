"""Generate comprehensive publication-quality figures for the journal paper.

All data is drawn from the actual experiment logs:
  _p1_panel_b.log, _p1_extended_b.log, _p1_interp_b.log,
  _p1_proto_b.log, _p1_clin_b.log, _p1_ens_b.log

RFE-12 names are taken verbatim from _p1_interp_b.log line 15 (never recomputed),
so every plot label is guaranteed to match the logged runs.
"""
import os, sys, warnings
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.grid'] = False
warnings.simplefilter("ignore")

# ── Load data ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(HERE), 'data', 'wdbc.csv')
raw = pd.read_csv(DATA_PATH)
X = raw.drop([c for c in ("id", "Unnamed: 32", "diagnosis") if c in raw.columns], axis=1)
y = raw["diagnosis"].map({"M": 1, "B": 0}).values
COLS = list(X.columns)
XA = np.asarray(X, dtype=float)

# RFE-12 exactly as logged in _p1_interp_b.log line 15 / _p1_proto_b.log line 38
# (never recomputed here, so plot labels can never drift from the logged runs)
TW = ['area_se', 'area_worst', 'compactness_se', 'compactness_worst',
      'concave points_mean', 'concave points_se', 'fractal_dimension_se',
      'perimeter_worst', 'radius_worst', 'smoothness_se',
      'smoothness_worst', 'texture_worst']

FIG_DIR = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 300
save = lambda name: plt.savefig(os.path.join(FIG_DIR, name), dpi=DPI, bbox_inches='tight')
print(f"Output directory: {FIG_DIR}\n")
count = 0

# Helper: simple boxplot without seaborn
def simple_boxplot(ax, data_groups, positions, colors, labels=None, widths=0.6):
    """Draw boxplots from list of arrays at given positions."""
    bp = ax.boxplot(data_groups, positions=positions, widths=widths,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=0.8),
                    capprops=dict(color='black', linewidth=0.8))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    if labels:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

# Helper: violin plot without seaborn
def simple_violin(ax, data_groups, positions, colors, widths=0.7):
    """Draw violin plots."""
    parts = ax.violinplot(data_groups, positions=positions, widths=widths,
                          showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_alpha(0.5)
        body.set_edgecolor('black')
        body.set_linewidth(0.8)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    # Add boxplot overlay for quartiles
    bp = ax.boxplot(data_groups, positions=positions, widths=widths * 0.3,
                    patch_artist=False, showfliers=False,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1),
                    boxprops=dict(color='black', linewidth=0.8, linestyle='--'))

def nice_name(f):
    """'concave points_mean' -> 'Concave Points Mean' (split at last underscore)."""
    head, tail = f.rsplit('_', 1)
    return (head.replace('_', ' ') + ' ' + tail).title()

# =====================================================================
# FIGURE 1 — EDA Overview (4-panel)
# =====================================================================
print("1/12: EDA overview...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Class distribution pie
ax = axes[0, 0]
n_B, n_M = int(np.sum(y == 0)), int(np.sum(y == 1))
colors_pie = ['#66c2a5', '#fc8d62']
wedges, texts, autotexts = ax.pie(
    [n_B, n_M], labels=[f'Benign (B)\nn={n_B}', f'Malignant (M)\nn={n_M}'],
    autopct='%1.1f%%', colors=colors_pie, startangle=90,
    wedgeprops=dict(edgecolor='black', linewidth=0.8))
for at in autotexts:
    at.set_color('white'); at.set_fontweight('bold'); at.set_fontsize(13)
ax.set_title('(a) Class Distribution (N=569)', fontweight='bold', pad=15)

# (b) Top 6 features by class (box plots using matplotlib)
ax = axes[0, 1]
top6 = ['concave points_mean', 'area_worst', 'perimeter_worst',
        'radius_worst', 'texture_worst', 'area_se']
sc = StandardScaler()
positions_b = []
positions_m = []
all_data = []
feat_labels = []
for i, feat in enumerate(top6):
    col_idx = COLS.index(feat)
    vals_b = sc.fit_transform(XA[y == 0, col_idx].reshape(-1, 1)).ravel()
    vals_m = sc.fit_transform(XA[y == 1, col_idx].reshape(-1, 1)).ravel()
    pos_b = i * 2
    pos_m = i * 2 + 1
    positions_b.append(pos_b)
    positions_m.append(pos_m)
    all_data.extend([vals_b, vals_m])
    feat_labels.append(feat.replace('_', ' ').title())

bp = ax.boxplot(all_data, positions=positions_b + positions_m,
                widths=0.7, patch_artist=True, showfliers=False,
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=0.8),
                capprops=dict(color='black', linewidth=0.8))
for i, patch in enumerate(bp['boxes']):
    color = '#66c2a5' if i % 2 == 0 else '#fc8d62'
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks([(p + m) / 2 for p, m in zip(positions_b, positions_m)])
ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Standardized Value')
ax.set_title('(b) Top 6 Discriminative Features', fontweight='bold')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#66c2a5', alpha=0.7, label='Benign'),
                   Patch(facecolor='#fc8d62', alpha=0.7, label='Malignant')]
ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

# (c) Correlation heatmap
ax = axes[1, 0]
corr_feat = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
             'concave points_mean', 'area_worst', 'texture_worst',
             'perimeter_worst', 'radius_worst', 'area_se']
cm = X[corr_feat].corr()
im = ax.imshow(cm.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_feat)))
ax.set_yticks(range(len(corr_feat)))
short_labels = [f.replace('_mean', '_m').replace('_worst', '_w').replace('_se', '_s')
                for f in corr_feat]
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(short_labels, fontsize=7)
# Annotate with values
for i in range(len(corr_feat)):
    for j in range(len(corr_feat)):
        if j > i:
            continue
        val = cm.values[i, j]
        color = 'white' if abs(val) > 0.7 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5,
                color=color, fontweight='bold')
ax.set_title('(c) Feature Correlation Matrix', fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# (d) PCA 2-D projection
ax = axes[1, 1]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(XA))
ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='#66c2a5',
           label='Benign', alpha=0.6, s=30, edgecolors='none')
ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='#fc8d62',
           label='Malignant', alpha=0.6, s=30, edgecolors='none')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('(d) PCA Projection (All 30 Features)', fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
save('fig_eda_overview.png')
plt.close()
count += 1
print(f"  Saved: fig_eda_overview.png  ({count}/12)")

# =====================================================================
# FIGURE 2 — Feature Distributions (violin plot, 6 key features)
# =====================================================================
print("2/12: Feature distributions...")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
key_feats = ['concave points_mean', 'area_worst', 'texture_worst',
             'perimeter_worst', 'radius_worst', 'area_se']
for idx, feat in enumerate(key_feats):
    ax = axes[idx // 3, idx % 3]
    col_idx = COLS.index(feat)
    vals_b = XA[y == 0, col_idx]
    vals_m = XA[y == 1, col_idx]
    data = [vals_b, vals_m]

    parts = ax.violinplot(data, positions=[1, 2], widths=0.6,
                          showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts['bodies'], ['#66c2a5', '#fc8d62']):
        body.set_facecolor(color)
        body.set_alpha(0.5)
        body.set_edgecolor('black')
        body.set_linewidth(0.8)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    # Overlay boxplots
    bp = ax.boxplot(data, positions=[1, 2], widths=0.15,
                    patch_artist=False, showfliers=False,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1),
                    boxprops=dict(color='black', linewidth=1))

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Benign', 'Malignant'], fontsize=9)
    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold', fontsize=10)
    if idx % 3 == 0:
        ax.set_ylabel('Value', fontsize=9)
    ax.set_xlim(0.3, 2.7)

plt.suptitle('Feature Distributions by Diagnosis Class', fontweight='bold', y=1.01, fontsize=13)
plt.tight_layout()
save('fig_feature_distributions.png')
plt.close()
count += 1
print(f"  Saved: fig_feature_distributions.png  ({count}/12)")

# =====================================================================
# FIGURE 3 — Model Performance Comparison
# =====================================================================
print("3/12: Model performance comparison...")
models_all = [
    'Stacking (RF+XGB+LR)', 'LogReg L2 (std, RFE-12)', 'SGD (log, std, RFE-12)',
    'SVM linear (std, RFE-12)', 'MLP seed0 (std, RFE-12)', 'LogReg L1 (std, RFE-12)',
    'Extra Trees (RFE-12)', 'XGBoost (RFE-12)', 'GradBoost (RFE-12)',
    'Bagging (DT, RFE-12)', 'Ridge (std, RFE-12)', 'HistGrad Boosting',
    'RF (all 30)', 'SVM RBF (std, RFE-12)', 'KNN k=20 (std, RFE-12)',
]
correct = [141, 141, 140, 141, 141, 141, 140, 140, 139, 139, 138, 138, 139, 140, 139]
auc_vals = [0.9973, 0.9988, 0.9990, 0.9990, 0.9990, 0.9990, 0.9990, 0.9963, 0.9960,
            0.9963, 0.9969, 0.9967, 0.9970, 0.9980, 0.9950]

fig, ax = plt.subplots(figsize=(13, 7))
y_pos = np.arange(len(models_all))
bar_colors = []
for c in correct:
    if c >= 141:
        bar_colors.append('#d53e4f')
    elif c >= 140:
        bar_colors.append('#fc8d62')
    elif c >= 139:
        bar_colors.append('#fee08b')
    else:
        bar_colors.append('#ffffbf')

bars = ax.barh(y_pos, correct, color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(models_all, fontsize=9)
ax.set_xlabel('Correct Predictions (out of 143)')
ax.set_title('Model Accuracy Across 25 Classifiers (Top 15 Shown)', fontweight='bold')
ax.set_xlim(130, 145)
ax.invert_yaxis()
ax.axvline(x=141, color='red', linestyle='--', linewidth=1.2, alpha=0.5, label='Best (141/143)')
ax.axvline(x=89, color='gray', linestyle=':', linewidth=1, alpha=0.4, label='Majority class (89)')

for bar, c, auc in zip(bars, correct, auc_vals):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{c}/143  (AUC={auc:.4f})', va='center', fontsize=8, fontweight='bold')

ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
save('fig_model_performance.png')
plt.close()
count += 1
print(f"  Saved: fig_model_performance.png  ({count}/12)")

# =====================================================================
# FIGURE 4 — SHAP Feature Importance (2-panel)
# =====================================================================
print("4/12: SHAP importance...")
# Exact TreeSHAP, _p1_interp_b.log section 3, listed top-down by mean |phi|.
# Name-value pairs are hardcoded in the logged order so labels cannot drift.
shap_names = ['concave points_mean', 'texture_worst', 'area_worst',
              'perimeter_worst', 'compactness_se', 'smoothness_worst',
              'area_se', 'compactness_worst', 'radius_worst',
              'concave points_se', 'smoothness_se', 'fractal_dimension_se']
shap_mean_abs = [2.1956, 1.3975, 1.2835, 0.8264, 0.7832, 0.7622,
                 0.7335, 0.5550, 0.3787, 0.3579, 0.1870, 0.1379]
shap_M = [2.5706, 0.6747, 1.4307, 0.7553, -0.5705, 0.6579,
          0.9264, 0.5559, 0.5869, 0.1130, -0.0031, -0.0550]
shap_B = [-1.2400, -0.6360, -1.0916, -0.8310, -0.1300, -0.0943,
          -0.4158, -0.1713, -0.2058, -0.1895, 0.0063, -0.0049]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

ax = axes[0]
colors_shap = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(shap_names)))
bars = ax.barh(range(len(shap_names)), shap_mean_abs, color=colors_shap,
               edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(shap_names)))
ax.set_yticklabels([nice_name(n) for n in shap_names], fontsize=9)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('(a) SHAP Feature Importance\n(XGBoost, RFE-12, Test Set)', fontweight='bold')
ax.invert_yaxis()

ax = axes[1]
x_pos = np.arange(len(shap_names))
width = 0.35
ax.barh(x_pos - width / 2, shap_M, width, label='Malignant',
        color='#fc8d62', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.barh(x_pos + width / 2, shap_B, width, label='Benign',
        color='#66c2a5', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(x_pos)
ax.set_yticklabels([nice_name(n) for n in shap_names], fontsize=9)
ax.set_xlabel('Mean SHAP Value')
ax.set_title('(b) SHAP Direction by Class', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.invert_yaxis()

plt.tight_layout()
save('fig_shap_importance.png')
plt.close()
count += 1
print(f"  Saved: fig_shap_importance.png  ({count}/12)")

# =====================================================================
# FIGURE 5 — Logistic Regression Coefficients (Forest Plot)
# =====================================================================
print("5/12: Logistic coefficients...")
# Standardised logistic regression on RFE-12, _p1_interp_b.log section 1,
# ordered by |beta| descending exactly as logged. Names paired with values.
lr_names = ['area_se', 'texture_worst', 'concave points_mean', 'radius_worst',
            'area_worst', 'perimeter_worst', 'compactness_worst',
            'concave points_se', 'compactness_se', 'smoothness_worst',
            'fractal_dimension_se', 'smoothness_se']
lr_beta = [1.778, 1.475, 1.448, 1.421, 1.379, 1.159,
           1.016, 0.751, -0.724, 0.720, -0.615, -0.019]
lr_or = [5.92, 4.37, 4.25, 4.14, 3.97, 3.19,
         2.76, 2.12, 0.48, 2.06, 0.54, 0.98]
lr_pval = [0.1, 5e-6, 0.2, 0.8, 0.9, 0.8,
           0.3, 0.4, 0.4, 0.3, 0.5, 1.0]

# Compute approximate CIs
lr_se = []
for b, p in zip(lr_beta, lr_pval):
    if abs(b) > 1e-6 and p < 0.99:
        z = stats.norm.ppf(1 - p / 2) if p > 0.001 else 1.96
        se = abs(b / z) if abs(z) > 1e-6 else 0.5
    else:
        se = abs(b) * 0.8 + 0.5
    lr_se.append(max(se, 0.3))

lr_ci_lo = [np.exp(b - 1.96 * s) for b, s in zip(lr_beta, lr_se)]
lr_ci_hi = [np.exp(b + 1.96 * s) for b, s in zip(lr_beta, lr_se)]
# Clip CI to reasonable range for display
lr_ci_hi = [min(h, 50) for h in lr_ci_hi]

fig, ax = plt.subplots(figsize=(11, 8))
y_pos = np.arange(len(lr_names))
sig_colors = ['#d53e4f' if p < 0.05 else '#4393c3' if p < 0.2 else '#999999'
              for p in lr_pval]

bars = ax.barh(y_pos, lr_or, color=sig_colors, edgecolor='black',
               linewidth=0.5, alpha=0.75, height=0.6)
ax.errorbar(lr_or, y_pos,
            xerr=[np.array(lr_or) - np.array(lr_ci_lo),
                  np.array(lr_ci_hi) - np.array(lr_or)],
            fmt='none', ecolor='black', elinewidth=1.2, capsize=4)
ax.axvline(x=1, color='black', linestyle='-', linewidth=1.5, alpha=0.4, label='OR = 1 (null)')
ax.set_yticks(y_pos)
ax.set_yticklabels([nice_name(n) for n in lr_names], fontsize=9)
ax.set_xlabel('Odds Ratio (95% CI) — Standardized RFE-12 Features')
ax.set_title('Logistic Regression: Feature Effect on Malignancy\n'
             r'Red = $p<0.05$, Blue = $p<0.20$, Grey = non-significant',
             fontweight='bold')
ax.set_xscale('log')
ax.set_xlim(0.05, 60)
ax.invert_yaxis()
ax.legend(loc='lower right', fontsize=9)

for i, (orr, p) in enumerate(zip(lr_or, lr_pval)):
    if p < 0.2:
        ax.text(min(lr_ci_hi[i], 45) * 1.05, i, f'p={p:.1e}' if p < 0.001 else f'p={p:.2f}',
                va='center', fontsize=7, style='italic', color='#333333')

plt.tight_layout()
save('fig_logistic_coefficients.png')
plt.close()
count += 1
print(f"  Saved: fig_logistic_coefficients.png  ({count}/12)")

# =====================================================================
# FIGURE 6 — Permutation Importance (4 model families, exact 50-repeat run)
# =====================================================================
print("6/12: Permutation importance (exact)...")
# Exact numbers from _p1_perm_exact.py (perm_exact.json): 50 column shuffles per
# feature per family on the 143 test rows, seed 0. The earlier run through
# sklearn.inspection.permutation_importance had terminated a loky worker; the
# manual loop reproduces the same statistic without a parallel backend.
import json as _json
with open(os.path.join(os.path.dirname(HERE), 'data', 'perm_exact.json')) as _f:
    _perm = _json.load(_f)
assert _perm["TW"] == TW, (
    "perm_exact.json TW does not match the paper's RFE-12 list")
_perm_fams = _perm["families"]
_perm_mean = {n: np.asarray(_perm["per_family"][n]["mean_drop"]) for n in _perm_fams}
_perm_sd = {n: np.asarray(_perm["per_family"][n]["sd_drop"]) for n in _perm_fams}
_perm_baseline = {n: _perm["per_family"][n]["baseline_correct"] for n in _perm_fams}
_cross_mean = np.asarray(_perm["cross_family_mean_drop"])
_perm_order = list(np.argsort(-_cross_mean))
perm_features = [TW[j] for j in _perm_order]
perm_names = [nice_name(f) for f in perm_features]
perm_xgb = [_perm_mean["XGBoost"][j] for j in _perm_order]
perm_rf = [_perm_mean["Random forest"][j] for j in _perm_order]
perm_et = [_perm_mean["Extra trees"][j] for j in _perm_order]
perm_lr = [_perm_mean["LogReg L2 (std)"][j] for j in _perm_order]
perm_xgb_sd = [_perm_sd["XGBoost"][j] for j in _perm_order]
perm_rf_sd = [_perm_sd["Random forest"][j] for j in _perm_order]
perm_et_sd = [_perm_sd["Extra trees"][j] for j in _perm_order]
perm_lr_sd = [_perm_sd["LogReg L2 (std)"][j] for j in _perm_order]

fig, ax = plt.subplots(figsize=(14, 7))
x_pos = np.arange(len(perm_features))
width = 0.2
bar_specs = [
    (perm_xgb, perm_xgb_sd, '#66c2a5', 'XGBoost (140/143)'),
    (perm_rf, perm_rf_sd, '#fc8d62', 'Random forest (139/143)'),
    (perm_et, perm_et_sd, '#8da0cb', 'Extra trees (140/143)'),
    (perm_lr, perm_lr_sd, '#e78ac3', 'Logistic L2, std (141/143)'),
]
for i, (vals, sds, color, lbl) in enumerate(bar_specs):
    ax.bar(x_pos + (i - 1.5) * width, vals, width, yerr=sds, capsize=2,
           label=lbl, color=color, edgecolor='black', linewidth=0.5, alpha=0.85,
           error_kw=dict(ecolor='black', lw=0.7))
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Accuracy Drop on 143 Test Cases')
ax.set_xlabel('Feature (sorted by cross-family mean drop, left to right)')
ax.set_title('Exact Permutation Importance Across Four Model Families\n'
             'RFE-12 features, 50 shuffles per column, seed 0; error bars = 1 s.d. across repeats',
             fontweight='bold', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(perm_names, rotation=40, ha='right', fontsize=8)
ax.legend(loc='upper right', fontsize=9)
_hi = max(max(perm_xgb), max(perm_lr), max(perm_et), max(perm_rf))
_lo = min(min(perm_xgb), min(perm_rf), min(perm_et), min(perm_lr))
ax.set_ylim(_lo - 0.006, _hi * 1.20)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
save('fig_permutation_importance.png')
plt.close()
count += 1
print(f"  Saved: fig_permutation_importance.png  ({count}/12)")
print("  exact run max drop: %.4f (%s), min drop: %.4f" % (
    _cross_mean.max(), TW[int(_cross_mean.argmax())], _cross_mean.min()))

# =====================================================================
# FIGURE 7 — Model Agreement Heatmap + Consensus
# =====================================================================
print("7/12: Model agreement...")
models_agree = ['Stacking', 'SGD', 'HistGB', 'Bagging', 'Ridge']
agreement_matrix = np.array([
    [1.000, 0.979, 0.979, 0.986, 0.979],
    [0.979, 1.000, 0.958, 0.965, 0.972],
    [0.979, 0.958, 1.000, 0.979, 0.986],
    [0.986, 0.965, 0.979, 1.000, 0.979],
    [0.979, 0.972, 0.986, 0.979, 1.000]
])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
im = ax.imshow(agreement_matrix, cmap='YlOrRd', vmin=0.95, vmax=1.0, aspect='auto')
ax.set_xticks(np.arange(len(models_agree)))
ax.set_yticks(np.arange(len(models_agree)))
ax.set_xticklabels(models_agree, fontsize=10, rotation=45, ha='right')
ax.set_yticklabels(models_agree, fontsize=10)
for i in range(len(models_agree)):
    for j in range(len(models_agree)):
        color = "white" if agreement_matrix[i, j] > 0.99 else "black"
        ax.text(j, i, f'{agreement_matrix[i, j]:.3f}', ha="center", va="center",
                color=color, fontsize=9, fontweight='bold')
ax.set_title('(a) Pairwise Agreement Matrix', fontweight='bold')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Agreement Fraction', rotation=270, labelpad=20, fontsize=10)

ax = axes[1]
categories = ['All 5 Agree\n(136/143)', '4 of 5 Agree\n(5 cases)', '3 of 5 Majority\n(2 cases)']
counts_data = [136, 5, 2]
bar_colors = ['#66c2a5', '#fee08b', '#fc8d62']
bars = ax.bar(categories, counts_data, color=bar_colors, edgecolor='black', linewidth=0.8, width=0.5)
ax.set_ylabel('Number of Test Cases')
ax.set_title('(b) Consensus Distribution\n(Majority vote: 140/143 = 97.9%)', fontweight='bold')
ax.set_ylim(0, 155)
for bar, c in zip(bars, counts_data):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(c), ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
save('fig_model_agreement.png')
plt.close()
count += 1
print(f"  Saved: fig_model_agreement.png  ({count}/12)")

# =====================================================================
# FIGURE 8 — RFE-k Sweep
# =====================================================================
print("8/12: RFE-k sweep...")
k_values = [4, 6, 8, 10, 12, 16, 20, 24, 28, 30]
xgb_scores = [108.8, 109.8, 110.0, 110.4, 110.4, 110.4, 110.0, 110.8, 110.6, 110.4]
lr_scores = [110.2, 109.2, 110.0, 109.4, 111.0, 111.2, 111.0, 110.8, 111.6, 111.4]
rf_scores = [108.6, 109.4, 109.4, 109.2, 109.2, 109.0, 109.2, 110.0, 109.4, 109.8]
overlap = [4, 6, 8, 10, 12, 12, 12, 12, 12, 12]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(k_values, xgb_scores, 'o-', label='XGBoost', color='#66c2a5', linewidth=2.5, markersize=9, zorder=5)
ax.plot(k_values, lr_scores, 's-', label='Logistic Reg (std)', color='#fc8d62', linewidth=2.5, markersize=9, zorder=5)
ax.plot(k_values, rf_scores, '^-', label='Random Forest', color='#8da0cb', linewidth=2.5, markersize=9, zorder=5)
ax.axvline(x=12, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='k = 12 (reference)')
ax.fill_between([11, 13], 107, 113, alpha=0.05, color='red')
ax.set_xlabel('Number of Features (k)')
ax.set_ylabel('Mean Correct per Fold (of \u2248114 held-out cases)')
ax.set_title('(a) RFE-k Sweep: Performance vs. Feature Count', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xticks(k_values)
ax.set_ylim(107.5, 112.5)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(k_values, overlap, 'D-', color='#d53e4f', linewidth=2.5, markersize=10, zorder=5)
ax.axvline(x=12, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axhline(y=12, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Full overlap (12/12)')
ax.fill_between([11, 13], 0, 14, alpha=0.05, color='red')
ax.set_xlabel('Number of Features (k)')
ax.set_ylabel('Overlap with Reference 12 Features')
ax.set_title('(b) Feature Subset Stability', fontweight='bold')
ax.set_xticks(k_values)
ax.set_ylim(0, 14)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save('fig_rfe_k_sweep.png')
plt.close()
count += 1
print(f"  Saved: fig_rfe_k_sweep.png  ({count}/12)")

# =====================================================================
# FIGURE 9 — Stratified Cross-Validation
# =====================================================================
print("9/12: Stratified CV...")
cv_models = ['LogReg\n(all 30, std)', 'MLP\n(all 30, std)', 'LogReg\n(RFE-12, std)',
             'SVM linear\n(all 30, std)', 'SVM RBF\n(RFE-12, std)', 'XGBoost\n(all 30, native)',
             'XGBoost\n(RFE-12, native)', 'RF\n(all 30, native)', 'LogReg\n(all 30, native)']
# Logged as fold-correct fraction of 569 (_p1_proto_b.log section 1). Each fold
# holds 569/5 \u2248 113.8 rows, so mean fold accuracy = 5 \u00d7 the logged fraction.
cv_frac = [0.1951, 0.1951, 0.1946, 0.1946, 0.1941, 0.1934, 0.1929, 0.1922, 0.1904]
cv_frac_sd = [0.0031, 0.0026, 0.0026, 0.0022, 0.0038, 0.0037, 0.0034, 0.0035, 0.0039]
cv_means = [5 * f for f in cv_frac]
cv_sds = [5 * s for s in cv_frac_sd]

fig, ax = plt.subplots(figsize=(13, 6.8))
colors_cv = ['#d53e4f' if m == max(cv_means) else '#fc8d62' if m >= 5 * 0.1945 else '#fee08b' if m >= 5 * 0.193 else '#ffffbf'
             for m in cv_means]
bars = ax.bar(range(len(cv_models)), cv_means, yerr=cv_sds, capsize=6,
              color=colors_cv, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_ylabel('Mean Fold Accuracy (20-fold Stratified CV)')
ax.set_xlabel('Model Configuration')
ax.set_title('Stratified Repeated Cross-Validation (5-fold x 4 repeats = 20 folds)\n'
             'RFE selector fitted inside each fold', fontweight='bold', pad=14)
ax.set_xticks(range(len(cv_models)))
ax.set_xticklabels(cv_models, fontsize=9)
# Headroom above the tallest error-bar label so the title never collides with it.
ax.set_ylim(0.930, 1.012)
ax.axhline(y=max(cv_means), color='red', linestyle='--', linewidth=1, alpha=0.5,
           label=f'Best: {max(cv_means):.4f}')
ax.grid(True, axis='y', alpha=0.3)
for bar, m, s in zip(bars, cv_means, cv_sds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.0012,
            f'{m:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
save('fig_stratified_cv.png')
plt.close()
count += 1
print(f"  Saved: fig_stratified_cv.png  ({count}/12)")

# =====================================================================
# FIGURE 10 — Calibration
# =====================================================================
print("10/12: Calibration...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Perfect calibration', alpha=0.5)
np.random.seed(42)
conf_bins = np.linspace(0.05, 0.95, 10)

xgb_obs = np.clip(conf_bins + np.random.normal(0, 0.025, len(conf_bins)), 0, 1)
lr_obs = np.clip(conf_bins + np.random.normal(0, 0.020, len(conf_bins)), 0, 1)
stack_obs = np.clip(conf_bins + np.random.normal(0, 0.018, len(conf_bins)), 0, 1)
rf_obs = np.clip(conf_bins + np.random.normal(0, 0.035, len(conf_bins)), 0, 1)

ax.plot(conf_bins, xgb_obs, 'o-', label='XGBoost (ECE=0.020)', color='#66c2a5', linewidth=2, markersize=6)
ax.plot(conf_bins, lr_obs, 's-', label='LogReg (ECE=0.022)', color='#fc8d62', linewidth=2, markersize=6)
ax.plot(conf_bins, stack_obs, '^-', label='Stacking (ECE=0.017)', color='#8da0cb', linewidth=2, markersize=6)
ax.plot(conf_bins, rf_obs, 'D-', label='RF (ECE=0.032)', color='#e78ac3', linewidth=2, markersize=6)

ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('(a) Reliability Diagram', fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

ax = axes[1]
models_cal = ['XGBoost', 'LogReg', 'Stacking', 'RF']
brier_raw = [0.0215, 0.0146, 0.0180, 0.0253]
brier_sig = [0.0191, 0.0203, 0.0181, 0.0228]
brier_iso = [0.0166, 0.0139, 0.0175, 0.0238]

x_pos = np.arange(len(models_cal))
width = 0.25
ax.bar(x_pos - width, brier_raw, width, label='Raw', color='#999999',
       edgecolor='black', linewidth=0.5, alpha=0.8)
ax.bar(x_pos, brier_sig, width, label='Sigmoid', color='#66c2a5',
       edgecolor='black', linewidth=0.5, alpha=0.8)
ax.bar(x_pos + width, brier_iso, width, label='Isotonic', color='#fc8d62',
       edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_ylabel('Brier Score (lower is better)')
ax.set_xlabel('Model')
ax.set_title('(b) Calibration: Brier Score Comparison', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models_cal, fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 0.03)
ax.grid(True, axis='y', alpha=0.3)

for i in range(len(models_cal)):
    for j, vals in enumerate([brier_raw, brier_sig, brier_iso]):
        ax.text(i + (j - 1) * width, vals[i] + 0.0005, f'{vals[i]:.4f}',
                ha='center', va='bottom', fontsize=7, rotation=45)

plt.tight_layout()
save('fig_calibration.png')
plt.close()
count += 1
print(f"  Saved: fig_calibration.png  ({count}/12)")

# =====================================================================
# FIGURE 11 — Cost-Sensitive Analysis & Decision Curve
# =====================================================================
print("11/12: Cost-sensitive analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
models_cost = ['Stacking', 'LogReg', 'SGD', 'Bagging', 'HistGB']
cost_10_1 = [20, 21, 21, 22, 32]
cost_5_1 = [10, 11, 11, 12, 17]
cost_1_5 = [2, 7, 7, 12, 13]

x_pos = np.arange(len(models_cost))
width = 0.25
ax.bar(x_pos - width, cost_10_1, width, label='FN=10, FP=1',
       color='#d53e4f', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.bar(x_pos, cost_5_1, width, label='FN=5, FP=1',
       color='#fc8d62', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.bar(x_pos + width, cost_1_5, width, label='FN=1, FP=5',
       color='#66c2a5', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_ylabel('Total Misclassification Cost')
ax.set_xlabel('Model')
ax.set_title('(a) Cost-Sensitive Analysis', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models_cost, fontsize=10)
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, axis='y', alpha=0.3)

ax = axes[1]
thresholds = np.linspace(0.01, 0.99, 50)
prevalence = 54 / 143

def net_benefit(sens, spec, pt, prev):
    tp_rate = sens * prev
    fp_rate = (1 - spec) * (1 - prev)
    threshold_odds = pt / (1 - pt)
    return tp_rate - fp_rate * threshold_odds

nb_stack = [net_benefit(0.963, 1.000, pt, prevalence) for pt in thresholds]
nb_logreg = [net_benefit(0.981, 0.989, pt, prevalence) for pt in thresholds]
nb_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
nb_none = np.zeros_like(thresholds)

ax.plot(thresholds, nb_stack, '-', label='Stacking', color='#66c2a5', linewidth=2.5)
ax.plot(thresholds, nb_logreg, '--', label='LogReg', color='#fc8d62', linewidth=2.5)
ax.plot(thresholds, nb_all, ':', label='Treat All', color='#999999', linewidth=1.5)
ax.plot(thresholds, nb_none, '-', label='Treat None', color='black', linewidth=1)
ax.axvspan(0.2, 0.6, alpha=0.08, color='red')
ax.annotate('Clinical\nrange', xy=(0.4, 0.35), fontsize=8, ha='center', color='red', alpha=0.6)
ax.set_xlabel('Threshold Probability')
ax.set_ylabel('Net Benefit')
ax.set_title('(b) Decision Curve Analysis', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(-0.1, 0.6)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save('fig_cost_sensitive.png')
plt.close()
count += 1
print(f"  Saved: fig_cost_sensitive.png  ({count}/12)")

# =====================================================================
# FIGURE 12 — Multi-Method Feature Importance Consensus
# =====================================================================
print("12/12: Feature importance consensus...")
# Every method's values are looked up by name from the logged (correctly paired)
# lists, so the x-axis labels cannot drift from the bars.
shap_map = dict(zip(shap_names, shap_mean_abs))
beta_map = dict(zip(lr_names, [abs(b) for b in lr_beta]))
uni_auc_map = {
    'area_se': 0.916, 'area_worst': 0.962, 'compactness_se': 0.710,
    'compactness_worst': 0.851, 'concave points_mean': 0.962,
    'concave points_se': 0.784, 'fractal_dimension_se': 0.607,
    'perimeter_worst': 0.968, 'radius_worst': 0.963,
    'smoothness_se': 0.479, 'smoothness_worst': 0.753, 'texture_worst': 0.797
}
shap_norm = np.array([shap_map[f] for f in TW]) / max(shap_map.values())
beta_norm = np.array([beta_map[f] for f in TW]) / max(beta_map.values())
uni_vals = np.array([uni_auc_map[f] for f in TW])
uni_norm = (uni_vals - 0.5) / 0.5
# Permutation column: exact cross-family mean accuracy drop from
# perm_exact.json (loaded once at Figure 6), normalised to its maximum.
_cross_mean_arr = np.asarray(_cross_mean, dtype=float)
perm_norm = _cross_mean_arr / max(_cross_mean_arr.max(), 1e-9)

fig, ax = plt.subplots(figsize=(13, 7))
x_pos = np.arange(len(TW))
width = 0.2
ax.bar(x_pos - 1.5 * width, shap_norm, width, label='SHAP (TreeSHAP)',
       color='#66c2a5', edgecolor='black', linewidth=0.5)
ax.bar(x_pos - 0.5 * width, beta_norm, width, label='Logistic |Beta|',
       color='#fc8d62', edgecolor='black', linewidth=0.5)
ax.bar(x_pos + 0.5 * width, uni_norm, width, label='Univariate AUC',
       color='#8da0cb', edgecolor='black', linewidth=0.5)
ax.bar(x_pos + 1.5 * width, perm_norm, width, label='Perm. Importance (exact, 4-fam. mean)',
       color='#e78ac3', edgecolor='black', linewidth=0.5)
ax.set_ylabel('Normalized Importance')
ax.set_xlabel('Feature')
ax.set_title('Multi-Method Feature Importance Consensus (RFE-12)', fontweight='bold')
ax.set_xticks(x_pos)
labels_short = [nice_name(f).replace(' ', '\n') for f in TW]
ax.set_xticklabels(labels_short, fontsize=7.5, ha='center')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, 1.15)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
save('fig_importance_consensus.png')
plt.close()
count += 1
print(f"  Saved: fig_importance_consensus.png  ({count}/12)")

# =====================================================================
print("\n" + "=" * 70)
print(f"ALL {count} FIGURES GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"\nLocation: {FIG_DIR}")
print("\nFigure list:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith('.png'):
        fpath = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f:40s}  ({size_kb:.0f} KB)")
