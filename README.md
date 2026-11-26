# WDBC Protocol Validation

Reproducible analysis for **"Twenty-five Classifiers with Recursive Feature Elimination for Breast Cancer Diagnosis: Protocol Validation, Multi-Method Interpretation, and Model Agreement Analysis"**

Author: Jeremiah Masaba (Makerere University)  
Target venue: *Computer Methods and Programs in Biomedicine* (Elsevier)

---

## What this repo contains

All code, data, experiment logs, and figures backing the manuscript. Every number reported in the paper can be regenerated from a fresh clone.

## Quick start

```bash
git clone https://github.com/jericho555/wdbc-protocol-validation.git
cd wdbc-protocol-validation
pip install -r requirements.txt
python scripts/run_panel.py        # Table 2: 25-model comparison
python scripts/visualize.py        # Regenerate all 15 figures
```

## Repository structure

```
├── data/
│   ├── wdbc.csv                 # UCI Wisconsin Diagnostic Breast Cancer (569 × 31)
│   └── perm_exact.json          # Exact 50-repeat permutation importance results
├── scripts/
│   ├── run_panel.py             # 25-model × 4-arm comparison (Table 2)
│   ├── run_reduct.py            # RFE stability, subset overlap, paired tests
│   ├── run_protocol.py          # Stratified 20-fold CV, nested tuning, RFE-k sweep
│   ├── run_interp.py            # SHAP (TreeSHAP), PDP, logistic coefficients
│   ├── run_perm_exact.py        # Exact permutation importance, 4 families × 50 repeats
│   ├── run_clinical.py          # Calibration (ECE, Brier), cost-sensitive, DCA
│   ├── run_ensemble.py          # Stacking, voting ensembles
│   ├── run_extended.py          # 5 additional classifiers + ROC/PR curves
│   ├── run_final.py             # Final results aggregation
│   ├── run_lime.py              # LIME stability analysis
│   ├── run_corr.py              # Feature correlation and redundancy
│   ├── run_validate.py          # Consistency checks (Wilson CI, paired tests)
│   ├── run_seeds.py             # Seed sensitivity
│   ├── run_explain.py           # Explanation agreement metrics
│   └── visualize.py             # Generates all 15 publication figures
├── figures/                     # Pre-generated PNGs (300 dpi, used in manuscript)
├── logs/                        # Raw stdout from each experiment run
├── manuscript/                  # main.tex + references.bib (elsarticle class)
└── requirements.txt             # Python dependencies
```

## Experiment protocol (summary)

| Parameter | Value |
|-----------|-------|
| Dataset | WDBC: 569 FNA samples, 30 features, binary (M/B) |
| Split | 75/25 stratified, `random_state=42` → 426 train / 143 test |
| Feature selection | RFE (DecisionTree, `random_state=0`), k=12, fitted on training rows only |
| Scaling | `StandardScaler` fitted on training rows only |
| Models | 25 classifiers across 7 families (see Table 2) |
| CV protocol | Repeated stratified 5-fold × 4 = 20 folds, selector inside each fold |
| Permutation importance | 50 column-wise shuffles per family, seed 0, manual numpy loop |
| SHAP | TreeSHAP via `xgboost.predict(output_contributions=True)` |
| Software | Python 3.12, scikit-learn 1.7.2, XGBoost 2.1, NumPy 2.x |

## Key findings

- Stacking and logistic regression L2 both achieve 141/143 on the held-out test partition
- RFE-12 vs. full-30: paired difference −0.1 cases, p = 0.400 (no significant loss)
- Concave points mean and worst-case texture are the dominant predictors (exact permutation + SHAP converge, Spearman ρ ≈ 0.70–0.92)
- Five-model agreement on 136/143 cases; one case (ID 82) missed by all
- Protocol validation (20-fold CV, k-sweep, nested tuning) confirms the headline numbers are not selection artefacts

## Citation

If you use this code in your research, please cite the paper (DOI will appear after acceptance).

## License

MIT — see LICENSE file.
