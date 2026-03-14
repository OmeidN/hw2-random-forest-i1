# HW 2 Random Forest ML Pipeline

**CSC 695 / 895 Spring 2026**  
**Instructor:** D. Petkovic  
**Author:** Omeid Nadery  
**Date:** 3/13/2026

---

## 1. Audit of Original Database

The original dataset has 871 samples and 609 columns. One column is the label and the rest are gene expression features, so there are 608 features. There are two classes, with label 1 for the i1 neuron cluster and label 0 for non-i1. The dataset is unbalanced, with most samples in class 0, about 90%, and about 10% in class 1. All feature columns are numeric. The notebook audit checks for missing values and none were reported. No columns were flagged as potentially personally identifying (e.g. no "id", "name", or "patient" in column names). The source file is Original training DB i1 cluster.csv.

The number of samples (871) is below the standard of 10 times the number of features (608), so the dataset is high-dimensional relative to sample size. Cross-validation and OOB evaluation are used to assess performance without relying on a single train/test split.

Relevant code for loading and auditing the raw data:

```python
def load_data(csv_path):
    path = Path(csv_path)
    return pd.read_csv(path)

def basic_audit(df, label_col):
    n_samples = len(df)
    feature_cols = [c for c in df.columns if c != label_col]
    n_features = len(feature_cols)
    class_counts = df[label_col].value_counts().sort_index().to_dict()
    missing_total = int(df.isna().sum().sum())
    # ... dtype check, PII heuristic, 10× features rule
    return {"n_samples": n_samples, "n_features": n_features, "class_counts": class_counts, ...}
```

*Comment:* `load_data` reads the CSV; `basic_audit` computes sample/feature counts, class distribution, missing values, and flags (e.g. high-dimensionality, imbalance) used to populate Table 1 and the narrative above.

**Table 1. Original dataset statistics**

| Statistic           | Value                              |
|--------------------|-------------------------------------|
| Number of samples  | 871                                 |
| Number of features | 608                                 |
| Number of classes  | 2                                   |
| Class 0 (non-i1)   | 781                                 |
| Class 1 (i1)       | 90                                  |
| Feature types      | Numeric                             |
| Missing values     | 0                                   |
| Dataset source     | Original training DB i1 cluster.csv |

---

## 2. Creation of Training DB and Verification DB

Two random samples were removed to form the verification set, one with class 1 and one with class 0, so the verification set has two samples (one per class). Holding out one sample per class ensures that the verification set is balanced and that we can check the model on both i1 and non-i1 examples. The training set is the original dataset minus these two samples, so it has 869 samples. The same random seed (42) was used so the split is reproducible.

Relevant code for the train/verification split:

```python
def split_train_verification(df, label_col, random_state=RANDOM_STATE):
    pos = df[df[label_col] == POS_LABEL]
    neg = df[df[label_col] == NEG_LABEL]
    pos_sample = pos.sample(n=1, random_state=random_state)
    neg_sample = neg.sample(n=1, random_state=random_state)
    verification_df = pd.concat([pos_sample, neg_sample]).copy()
    training_df = df.drop(index=verification_df.index).copy()
    return training_df, verification_df
```

*Comment:* One random sample from each class is drawn with a fixed seed; the rest stays as training data. This yields the 869 vs 2 split and reproducible Tables 2 and 3.

**Table 2. Training dataset statistics**

| Statistic           | Value |
|--------------------|-------|
| Number of samples  | 869   |
| Number of features | 608   |
| Class 0 count      | 780   |
| Class 1 count      | 89    |

**Table 3. Verification dataset statistics**

| Statistic           | Value |
|--------------------|-------|
| Number of samples  | 2     |
| Class 0 count      | 1     |
| Class 1 count      | 1     |

---

## 3. Software Tools

- **Python** – Runtime and scripting.

Paths and random seed are centralized so the pipeline is reproducible:

```python
# config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Original training DB i1 cluster.csv"
RANDOM_STATE = 42
```

*Comment:* Using a single project root and fixed `RANDOM_STATE` ensures the same train/verification split and CV folds across runs.

- **Jupyter Notebook** – Interactive workflow and documentation.
- **scikit-learn** – RandomForestClassifier for training, StratifiedKFold for fold indices, and metrics (confusion matrix, accuracy, precision, recall, F1).
- **pandas** – Loading CSV data, building tables, and saving outputs.
- **NumPy** – Numerical arrays and random state handling.
- **matplotlib / seaborn** – Top-10 feature importance plot (saved as reports/figures/top10_features.png).

---

## 4. Experimental Methods and Setup

**Method 1, Random Forest with OOB estimation.** Out-of-bag (OOB) estimation uses the fact that each tree in the forest is trained on a bootstrap sample of the data, so about one-third of the training samples are OOB for each tree. Those samples get predictions from the trees that did not use them, giving an internal validation estimate without a separate holdout set. A grid search was run over n_estimators (200, 500, 1000), max_features (mtry: sqrt, 2×sqrt ≈ 49, 5×sqrt ≈ 123, and 0.1 as fraction), and min_samples_leaf (1, 2, 5). Each model used RandomForestClassifier with bootstrap=True and oob_score=True. OOB predictions were taken from the out-of-bag decision function and thresholded at 0.5 to get predicted classes. From those, confusion matrix, accuracy, precision, recall, and F1 were computed. The best model was chosen by highest F1 because the dataset is imbalanced (few i1 samples). F1 balances precision and recall and is more informative than accuracy when the positive class is rare.

Key OOB grid and prediction logic:

```python
n_features = X_train.shape[1]
sqrt_n = math.sqrt(n_features)
max_features_list = ["sqrt", min(int(2 * sqrt_n), n_features), min(int(5 * sqrt_n), n_features), 0.1]
# ... loop over n_estimators, max_features, min_samples_leaf
clf = RandomForestClassifier(n_estimators=..., max_features=..., bootstrap=True, oob_score=True, ...)
clf.fit(X_train, y_train)
oob_probs = clf.oob_decision_function_[:, 1]
oob_pred = (oob_probs >= 0.5).astype(int)
metrics = compute_metrics(y_train, oob_pred)
```

*Comment:* The mtry grid is built from sqrt and multiples (49, 123 for 608 features); each model is evaluated via OOB class-1 probabilities thresholded at 0.5, then metrics (including F1) are computed for model selection.

**Method 2, Manual 3-fold cross-validation.** This method gives an estimate of how the model generalizes to unseen data by training on two-thirds of the data and evaluating on the remaining third, repeated three times so each sample is in the test set exactly once. StratifiedKFold with 3 splits and a fixed random state was used only to get train and test indices (no one-line CV shortcut). For each fold, a separate RandomForestClassifier was trained on the training indices and predictions were made on the test indices. Confusion matrix, accuracy, precision, recall, and F1 were computed per fold. Final accuracy and F1 are the averages over the three folds. The same hyperparameters as the best OOB model were used so that the CV results are comparable.

Key 3-fold CV loop (indices only; training and metrics are manual):

```python
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
    clf = RandomForestClassifier(n_estimators=..., max_features=..., ...)
    clf.fit(X_train_fold, y_train_fold)
    y_pred = clf.predict(X_test_fold)
    metrics = compute_metrics(y_test_fold, y_pred)
```

*Comment:* StratifiedKFold only supplies fold indices; a separate RF is trained per fold and evaluated on that fold’s test set, yielding the per-fold accuracy and F1 reported in Section 5.

---

## 5. Actual Results of RF Training

**Method 1 (OOB)**  
Best hyperparameters: n_estimators = 1000, max_features = 123 (5×sqrt), min_samples_leaf = 1.

Confusion matrix (rows = true class, columns = predicted class; order 0 then 1):

|             | Predicted 0 | Predicted 1 |
|-------------|-------------|-------------|
| True 0      | 779         | 1           |
| True 1      | 10          | 79          |

| Metric    | Value    |
|----------|----------|
| Accuracy | 0.98734  |
| Precision| 0.98750  |
| Recall   | 0.88764  |
| F1       | 0.93491  |

**Method 2 (3-fold CV)**  
Per-fold results (accuracy and F1 to 5 decimals):

| Fold | Accuracy | F1      |
|------|----------|---------|
| 1    | 0.97931  | 0.88889 |
| 2    | 0.98621  | 0.93103 |
| 3    | 0.98962  | 0.94545 |

Mean accuracy 0.98505, mean F1 0.92179. Per-fold precision and recall (to 5 decimals) are available in the notebook; fold 1 had recall 0.80 and fold 3 had recall about 0.90, so there is some variance in how many i1 samples are correctly identified across folds.

Relevant code for selecting the best OOB model and for computing metrics (used for both OOB and CV):

```python
def get_best_oob_model(results, sort_by="f1"):
    best = max(results, key=lambda x: x[sort_by])
    return best

def compute_metrics(y_true, y_pred, pos_label=1):
    cm = confusion_matrix(y_true, y_pred)
    return {"confusion_matrix": cm, "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
            "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
            "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)}
```

*Comment:* The result with highest F1 is chosen as the best OOB model; its confusion matrix and accuracy (and precision, recall, F1) are those in the tables above. All metrics are computed via scikit-learn with the positive class set to 1 (i1).

**Interpretation of the best OOB confusion matrix.** The model correctly classified 779 of 780 non-i1 samples (one false positive) and 79 of 89 i1 samples (10 false negatives). So most errors are missed i1 cases rather than non-i1 wrongly labeled as i1. That pattern is common with imbalanced data when the minority class is harder to separate. Precision (0.98750) is higher than recall (0.88764) because the model is somewhat conservative in predicting class 1; when it does predict i1, it is usually right, but it misses some true i1 samples.

**Comparison.** OOB and 3-fold CV both give accuracy around 98–99% and F1 around 0.92–0.94. OOB is slightly higher (0.98734 vs 0.98505) and the best OOB F1 (0.93491) is a bit higher than the mean CV F1 (0.92179). The two methods agree that the model performs well and is not strongly overfitting. The small gap between OOB and CV is consistent with OOB being a slightly optimistic estimate compared to true held-out evaluation.

---

## 6. Feature Ranking

Feature importance from the best OOB-trained Random Forest was computed using Gini importance (the total decrease in node impurity from splits on each feature across all trees). The top 10 are below. Figure 1 shows the same rankings as a bar chart.

Relevant code for Gini-based feature ranking:

```python
def get_feature_importance_table(model, feature_names):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "gini_importance": model.feature_importances_,
    }).sort_values("gini_importance", ascending=False)
    return importance_df.reset_index(drop=True)
```

*Comment:* scikit-learn’s `feature_importances_` gives Gini importance per feature; we sort descending to produce the ranking used in Table 4 and Figure 1.

![Top 10 features (Random Forest) by Gini importance](reports/figures/top10_features.png)

**Figure 1.** Top 10 features (Random Forest) by Gini importance.

**Table 4. Top 10 ranked features (Gini importance)**

| Rank | Feature   | Gini importance |
|------|-----------|-----------------|
| 1    | COL5A2.1  | 0.12725         |
| 2    | COL5A2.2  | 0.12706         |
| 3    | COL5A2    | 0.11970         |
| 4    | NDNF.2    | 0.04422         |
| 5    | MYO16     | 0.03574         |
| 6    | NDNF.1    | 0.03569         |
| 7    | NDNF      | 0.03266         |
| 8    | FAT1      | 0.03151         |
| 9    | NPNT      | 0.02999         |
| 10   | CBLN4     | 0.02176         |

The top three are COL5A2 family (COL5A2, COL5A2.1, COL5A2.2); together they account for about 37% of total Gini importance, so this gene family dominates the ranking. NDNF appears in three entries (NDNF, NDNF.1, NDNF.2) and FAT1 appears once. These match the biological markers (COL5A2, NDNF, FAT1) from Aevermann et al., so the model's most important features align with known i1 biology without being given that information during training. MYO16, NPNT, and CBLN4 are additional genes that may be relevant to i1 or correlated with the main markers.

---

## 7. RF Runtime Test

The best OOB model was used to classify the two verification samples. Results:

Relevant code for verification predictions:

```python
def run_verification_predictions(model, X_verify, y_verify):
    pred = model.predict(X_verify)
    prob = model.predict_proba(X_verify)
    results = pd.DataFrame({
        "true_class": y_verify.values,
        "predicted_class": pred,
        "prob_class_0": prob[:, 0],
        "prob_class_1": prob[:, 1],
        "correct": (pred == y_verify.values),
    })
    return results
```

*Comment:* The best fitted RF is run on the two held-out samples; `predict_proba` gives the probabilities reported in Table 5 and used to confirm both predictions are correct.

Results:

**Table 5. Verification sample predictions**

| Sample | True label | Predicted label | Probability (predicted class) | Correct |
|--------|------------|-----------------|--------------------------------|---------|
| 1      | 1          | 1               | 0.96900                        | Yes     |
| 2      | 0          | 0               | 0.74000                        | Yes     |

Both predictions are correct. The first sample (i1) is predicted as class 1 with high confidence (about 97% probability for class 1). The second (non-i1) is predicted as class 0 with moderate confidence (about 74% for class 0). Overall, the runtime test shows that the best model generalizes to the held-out verification set.

---

## 8. Summary

The pipeline audited the original i1 cluster dataset (871 samples, 608 features, two classes), built a training set (869 samples) and a two-sample verification set (one per class), and trained Random Forest models using OOB estimation and manual 3-fold CV. The grid used mtry values sqrt, 2×sqrt (49), 5×sqrt (123), and 0.1. The best model (1000 trees, max_features 123 [5×sqrt], min_samples_leaf 1) achieved OOB accuracy 0.98734 and F1 0.93491, with mean CV accuracy 0.98505 and mean F1 0.92179. Gini-based feature ranking placed the COL5A2 family, NDNF, and FAT1 at the top, matching known i1 markers from the literature. Both verification samples were classified correctly. The results support that the Random Forest pipeline is suitable for this classification task and that the learned feature importance is biologically interpretable.

---

## 9. References

Aevermann et al. (2018). Cell type discovery using single-cell transcriptomics: implications for ontological representation. Human Molecular Genetics.

Scikit-learn documentation. https://scikit-learn.org/

ChatGPT was used to generate the report structure and instructions for this assignment.
