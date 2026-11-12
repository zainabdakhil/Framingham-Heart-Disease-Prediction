# Predicting 10-Year Coronary Heart Disease Risk (Framingham)
# Logistic Regression + Random Forest + XGBoost, each after Lasso feature selection.
# Runs in PyCharm (reads the local CSV) and on Kaggle (finds the CSV in /kaggle/input).
# Note: every plot opens in a window; close it to continue.

# ======================================================================
# Predicting 10-Year Coronary Heart Disease Risk
# ======================================================================

# ======================================================================
# 3.1 Libraries and loading the data
# ======================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay)

# newer scikit-learn versions print deprecation warnings for `penalty`; results are not affected
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values")

# find framingham.csv automatically (Kaggle), otherwise use the local file name
DATA_PATH = "framingham.csv"
LOCAL_PATH = r"C:\Users\lenovo\OneDrive\Desktop\framingham.csv"   # change if your file is elsewhere
if os.path.exists(LOCAL_PATH):
    DATA_PATH = LOCAL_PATH
for root, _, files in os.walk("/kaggle/input"):
    if "framingham.csv" in files:
        DATA_PATH = os.path.join(root, "framingham.csv")
        break

df = pd.read_csv(DATA_PATH)
print("Loaded from:", DATA_PATH)
print(df.shape)

print(df.head())

df.info()

print(df.describe().T)

missing = df.isna().sum()
print(missing[missing > 0])
print((df.isna().mean() * 100).round(2)[missing > 0])

# ======================================================================
# 3.2 Exploratory data analysis
# ======================================================================

print(df["TenYearCHD"].value_counts())
print(df["TenYearCHD"].value_counts(normalize=True).round(3))

sns.countplot(x="TenYearCHD", data=df)
plt.title("Target Distribution")
plt.show()

cont = ["age", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "cigsPerDay"]

df[cont].hist(bins=30, figsize=(14, 8))
plt.tight_layout()
plt.show()

print(df[cont].skew().round(2))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.ravel(), cont):
    sns.boxplot(x="TenYearCHD", y=col, data=df, ax=ax)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# ======================================================================
# 3.3 Train/test split
# ======================================================================

X = df.drop(columns=["TenYearCHD"])
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(X_train.shape, X_test.shape)
print(y_train.value_counts(normalize=True).round(3))
print(y_test.value_counts(normalize=True).round(3))

# ======================================================================
# 3.4 Preprocessing inside a pipeline
# ======================================================================

log_cols = ["glucose"]
num_cols = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate"]
cat_cols = ["male", "education", "currentSmoker", "BPMeds",
            "prevalentStroke", "prevalentHyp", "diabetes"]

# glucose: median -> log1p -> scale
log_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p)),
    ("scaler", StandardScaler()),
])

# other continuous variables: median -> scale
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# binary / categorical variables: most frequent only (no scaling)
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor = ColumnTransformer([
    ("log", log_pipe, log_cols),
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# sanity check: fit on train only, transform test
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print(X_train_prep.shape, X_test_prep.shape)
print(np.isnan(X_train_prep).sum(), np.isnan(X_test_prep).sum())   # must be 0 and 0

# ======================================================================
# 3.5 Lasso as dimensionality reduction
# ======================================================================

selector = SelectFromModel(
    LogisticRegression(penalty="l1", solver="liblinear",
                       C=0.1, class_weight="balanced", random_state=42)
)

# ======================================================================
# 3.6 Building the models
# ======================================================================

def build(clf):
    return Pipeline([
        ("prep", clone(preprocessor)),      # imputation + log + scaling
        ("select", clone(selector)),        # Lasso keeps the important variables
        ("clf", clf),
    ])

models = {
    "Lasso + Logistic": build(
        LogisticRegression(class_weight="balanced", random_state=42)),

    "Lasso + RandomForest": build(
        RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10,
                               class_weight="balanced_subsample", random_state=42)),

    "Lasso + XGBoost": build(
        XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=5.6,          # about 3594 / 644 (class imbalance)
                      eval_metric="logloss", random_state=42)),
}

# ======================================================================
# 3.7 Training and evaluation
# ======================================================================

results = []

for name, model in models.items():
    model.fit(X_train, y_train)              # preprocessing + Lasso + classifier, train only
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })
    print(name)
    print(confusion_matrix(y_test, y_pred))
    print()

results_df = pd.DataFrame(results).round(3)
print(results_df)

# ======================================================================
# ROC curves
# ======================================================================

plt.figure(figsize=(7, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], color="black", linestyle="dashed", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ======================================================================
# Confusion matrices
# ======================================================================

for name, model in models.items():
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["No CHD", "CHD"], cmap="Blues"
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# ======================================================================
# 3.8 Interpreting the models
# ======================================================================

best = models["Lasso + Logistic"]
feature_names = np.array(log_cols + num_cols + cat_cols)   # same order as the ColumnTransformer
mask = best.named_steps["select"].get_support()

print("Selected features:", feature_names[mask].tolist())
print("Removed features :", feature_names[~mask].tolist())

# Lasso coefficients for all variables (zero = removed)
lasso_coefs = best.named_steps["select"].estimator_.coef_[0]
coef_df = pd.DataFrame({"Feature": feature_names, "Lasso Coefficient": lasso_coefs})
coef_df["Abs"] = coef_df["Lasso Coefficient"].abs()
coef_df = coef_df.sort_values("Abs", ascending=False).drop(columns="Abs")
print(coef_df.round(3))

plt.figure(figsize=(8, 6))
sns.barplot(x="Lasso Coefficient", y="Feature", data=coef_df)
plt.axvline(0, color="k", linewidth=0.8)
plt.title("Lasso Coefficients (Feature Selection)")
plt.tight_layout()
plt.show()

# ======================================================================
# **Final Logistic Regression coefficients** (on the selected variables only). A positive coefficient raises the estimated risk.
# ======================================================================

final_coefs = best.named_steps["clf"].coef_[0]
final_df = pd.DataFrame({"Feature": feature_names[mask], "Logistic Coefficient": final_coefs})
final_df = final_df.reindex(final_df["Logistic Coefficient"].abs().sort_values(ascending=False).index)
print(final_df.round(3))

# ======================================================================
# **Feature importance of the tree models** (they have no coefficients).
# ======================================================================

for name, title in [("Lasso + RandomForest", "Random Forest Feature Importance"),
                    ("Lasso + XGBoost", "XGBoost Feature Importance")]:
    imp_df = pd.DataFrame({
        "Feature": feature_names[mask],
        "Importance": models[name].named_steps["clf"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    print(name)
    print(imp_df.round(3))
    print()

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=imp_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ======================================================================
# 3.9 Effect of the decision threshold (Logistic Regression only)
# ======================================================================

best = models["Lasso + Logistic"]

# 1) out-of-fold probabilities on the training set only (the test set is not touched)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_prob = cross_val_predict(best, X_train, y_train, cv=cv,
                               method="predict_proba")[:, 1]

# 2) choose the threshold with the best F1 on the training set
thresholds = np.arange(0.10, 0.90, 0.01)
f1_scores = [f1_score(y_train, train_prob >= t) for t in thresholds]
best_t = thresholds[int(np.argmax(f1_scores))]
print("Best threshold (from train CV):", round(best_t, 2))

plt.figure(figsize=(7, 5))
plt.plot(thresholds, f1_scores)
plt.axvline(best_t, color="red", linestyle="dashed", label=f"Best = {best_t:.2f}")
plt.xlabel("Threshold")
plt.ylabel("F1 (train CV)")
plt.title("F1 vs Threshold")
plt.legend()
plt.show()

# 3) apply it once to the test set
test_prob = best.predict_proba(X_test)[:, 1]

rows = []
for label, t in [("Default (0.50)", 0.5), (f"Tuned ({best_t:.2f})", best_t)]:
    pred = (test_prob >= t).astype(int)
    rows.append({
        "Threshold": label,
        "Recall": recall_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Accuracy": accuracy_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, test_prob),
    })
    print(label)
    print(confusion_matrix(y_test, pred))
    print()

print(pd.DataFrame(rows).round(3))

# ======================================================================
# 4. Final comparison
# ======================================================================

print(results_df.to_string(index=False))

# ======================================================================
# 6. Conclusion
# ======================================================================
