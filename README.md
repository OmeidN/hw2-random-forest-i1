# HW2: Random Forest ML Pipeline — i1 Nerve Cluster Classification

A complete Python + scikit-learn homework project for classifying i1 nerve cluster samples from gene expression data using a Random Forest pipeline.

## What This Project Does

- **Audits** the original CSV database (samples, features, class balance, missing values).
- **Splits** data into training and verification sets (removes 1× class 1 and 1× class 0 for verification).
- **Method 1:** Trains Random Forest models with OOB estimation and compares hyperparameters.
- **Method 2:** Runs manual 3-fold stratified cross-validation (no one-line CV shortcut).
- **Metrics:** Confusion matrix, accuracy, precision, recall, F1 (all to 5 decimal places).
- **Feature ranking:** Gini importance from the best model; top 10 features; overlap with biology ground truth (COL5A2, NDNF, FAT1).
- **Verification:** Runs the best model on the 2 held-out samples and reports predicted class and probabilities.
- **Saves** result tables and figures to CSV/PNG.

## Folder Structure

```
hw2_random_forest_i1/
├── data/
│   ├── raw/                    # Put "Original training DB i1 cluster.csv" here
│   ├── processed/              # training_db.csv, verification_db.csv (generated)
│   └── outputs/                # oob_results.csv, cv results, top10 features, etc.
├── notebooks/
│   └── hw2_rf_pipeline.ipynb   # Main workflow notebook
├── src/
│   ├── config.py               # Paths and constants
│   ├── data_utils.py            # Load, audit, split, save
│   ├── metrics_utils.py        # Metrics and formatting
│   ├── rf_oob.py               # OOB grid search
│   ├── rf_cv.py                # Manual 3-fold CV
│   ├── feature_rank.py         # Gini importance and ground-truth overlap
│   ├── runtime_test.py        # Verification predictions
│   └── report_helpers.py       # Save CSVs/figures, report text
├── reports/
│   └── figures/                # top10_features.png (generated)
├── requirements.txt
├── README.md
└── instructions.md             # Step-by-step for beginners
```

## How to Run

1. **Install dependencies** (from project root):
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your CSV** in `data/raw/`:
   - File name: `Original training DB i1 cluster.csv`

3. **Run the notebook:**
   - Open `notebooks/hw2_rf_pipeline.ipynb` in Jupyter or VS Code.
   - Run all cells from top to bottom (Kernel → Run All, or run each cell in order).

4. **Outputs** appear in:
   - `data/outputs/`: `oob_results.csv`, `cv_fold_results.csv`, `top10_features.csv`, `verification_predictions.csv`
   - `reports/figures/`: `top10_features.png`
   - `data/processed/`: `training_db.csv`, `verification_db.csv`

For detailed steps (virtual environment, first-time Jupyter, troubleshooting), see **instructions.md**.

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, openpyxl
