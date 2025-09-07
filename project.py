# Basic Imports for Data Analysis
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# Imports for Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Import for SMOTE Preprocessing
from imblearn.over_sampling import SMOTE

# Imports for Feature Selection algorithms
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# -------------------- (New) FIGURE OUTPUT SETUP --------------------
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name: str, dpi: int = 150):
    """
    Save both PNG and SVG with uniform 1280x720 sizing and padding.
    Keeps all figures consistent for the portfolio carousel.
    """
    base = os.path.join(FIG_DIR, name)
    fig = plt.gcf()
    fig.set_size_inches(1280 / dpi, 720 / dpi)  # 1280x720 canvas
    # letterbox/padding inside the saved file so labels/titles aren't clipped
    plt.savefig(base + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.3)
    plt.close()

# -------------------- TASK 1: EXPLORATIVE DATA ANALYSIS AND DATA PREPROCESSING --------------------

# Load dataset
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
# reading data, note that we tell Pandas to treat ? values as NaN
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    names=columns,
    na_values='?',
    skipinitialspace=True
)

df.head()

# Before cleaning
print("Before Cleaning:")
print(f"Total rows: {df.shape[0]}")
print("Missing values per column:")
print(df.isnull().sum())
print(f"Total missing rows: {df.isnull().any(axis=1).sum()}")
print()

# Drop unnecessary columns fnlwgt and education
df.drop(columns=['fnlwgt', 'education'], inplace=True)
# Drop rows with missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# After cleaning
print("After Cleaning:")
print(f"Total rows: {df.shape[0]}")
print("Missing values per column:")
print(df.isnull().sum())

# -------------------- TASK 2: ALGORITHM TESTING WTIHOUT FEATURE SELECTION --------------------

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and target
X = df.drop("income", axis=1)
y = df["income"]

# For nicer plot labels later
income_le = label_encoders.get("income")
income_labels = list(income_le.classes_) if income_le is not None else ["<=50K", ">50K"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training data and testing data, 80% train 20% test
# (small tweak: stratify for balanced holdout; harmless to remove if you prefer exact original)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to training data (from task 1)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Calling the different models (logistic regression and SVM use class_weight='balanced' from task 1 to solve class imbalance)
models = {
    "Logistic Regression (weighted)": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest (weighted)": RandomForestClassifier(class_weight='balanced'),
    "Naive Bayes": GaussianNB(),
    # (new: probability=True enables ROC curves; otherwise unchanged)
    "SVM (weighted)": SVC(class_weight='balanced', probability=True),
    "k-NN": KNeighborsClassifier()
}

# Evalulate all models using the training data (SMOTE) against the test data
results = []
y_preds = {}     # (new) store predictions for confusion matrices grid
roc_ready = {}   # (new) store (y_true, y_score) for ROC curves

for name, model in models.items():
    # Train on SMOTE-resampled training data
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)

    # collect predictions for visuals
    y_preds[name] = y_pred

    # try to get scores for ROC
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        y_score = MinMaxScaler().fit_transform(raw.reshape(-1, 1)).ravel()

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

    if y_score is not None:
        roc_ready[name] = (y_test, y_score)

# Display results
results_df = pd.DataFrame(results).set_index("Model")
print("\nModel Performance (with Class Balancing):\n", results_df)

# -------------------- (New) VISUALS FROM TASK 2 --------------------

# ROC curves (all models that support it)
if roc_ready:
    plt.figure(figsize=(7.5, 6.5))
    for name, (yt, ys) in roc_ready.items():
        RocCurveDisplay.from_predictions(yt, ys, name=name)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title("ROC Curves (Test Set)")
    savefig("roc_curves")

# Confusion matrices grid (2x3)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()
for i, (name, y_pred) in enumerate(y_preds.items()):
    ax = axes[i]
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=income_labels, cmap="Blues", ax=ax
    )
    ax.set_title(name)
# hide unused cell if any
for j in range(len(y_preds), len(axes)):
    axes[j].axis("off")
fig.suptitle("Confusion Matrices — All Models", y=1.02)
plt.tight_layout()
savefig("cm_grid")

# Leaderboard table (Accuracy/Precision/Recall/F1)
plt.figure(figsize=(8.5, 2 + 0.38 * len(results_df)))
plt.axis("off")
tbl = plt.table(
    cellText=results_df.round(3).values,
    rowLabels=results_df.index,
    colLabels=results_df.columns,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.35)
plt.title("Model Performance — SMOTE(train) → Test", pad=12)
savefig("metrics_table")

# Class balance: raw vs SMOTE(train)
raw_counts = pd.Series(y).value_counts().sort_index()
smote_counts = pd.Series(y_train_smote).value_counts().sort_index()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
ax[0].bar(range(len(raw_counts)), raw_counts.values)
ax[0].set_title("Raw Class Distribution (All Data)")
ax[0].set_xticks(range(len(raw_counts)))
ax[0].set_xticklabels(income_labels)
ax[0].set_ylabel("Count")

ax[1].bar(range(len(smote_counts)), smote_counts.values, color="#6EDC72")
ax[1].set_title("Train Distribution After SMOTE")
ax[1].set_xticks(range(len(smote_counts)))
ax[1].set_xticklabels(income_labels)

fig.suptitle("Income Class Balance — Raw vs SMOTE", y=1.03)
plt.tight_layout()
savefig("class_balance_before_after")

# -------------------- TASK 3: FEATURE SELECTION ALGORITHMS --------------------

# Using Random Forest to get feature importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances and their corresponding feature names
importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

# Print the top 10 features
print("\nTop 10 Features (Random Forest Importance):")
for i in range(10):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# (New) Plot top-10 feature importances
top = 10
top_idx = indices[:top]
plt.figure(figsize=(9, 5))
plt.barh(range(top)[::-1], importances[top_idx][::-1])
plt.yticks(range(top)[::-1], feature_names[top_idx][::-1])
plt.xlabel("Importance")
plt.title("Top-10 Feature Importance (Random Forest)")
savefig("feature_importance_top10_rf")

# SelectKBest algorithm using mutual information
selector = SelectKBest(mutual_info_classif, k=10)
selector.fit(X_train, y_train)
scores = selector.scores_

print("\nTop 10 Features (SelectKBest - Mutual Info):")
top_k_indices = np.argsort(scores)[::-1][:10]
for i in top_k_indices:
    print(f"{feature_names[i]}: {scores[i]:.4f}")

# -------------------- TASK 4: FEATURE SELECTION ALGORITHM TESTING --------------------

# Select the top 3 features from the Random Forest feature importance
top3_features = ['age', 'education-num', 'relationship']
X_top3 = df[top3_features]

# Scale the selected features
X_top3_scaled = scaler.fit_transform(X_top3)

# Split the selected data into training data and testing data, 80% train 20% test
X_train_top3, X_test_top3, y_train_top3, y_test_top3 = train_test_split(
    X_top3_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE on the reduced feature set
X_train_top3_smote, y_train_top3_smote = smote.fit_resample(X_train_top3, y_train_top3)

# Evalulate all models using the (new) training data (SMOTE) against the test data
results_top3 = []
for name, model in models.items():
    model.fit(X_train_top3_smote, y_train_top3_smote)
    y_pred_top3 = model.predict(X_test_top3)

    results_top3.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_top3, y_pred_top3),
        "Precision": precision_score(y_test_top3, y_pred_top3),
        "Recall": recall_score(y_test_top3, y_pred_top3),
        "F1 Score": f1_score(y_test_top3, y_pred_top3)
    })

# Display results
results_top3_df = pd.DataFrame(results_top3).set_index("Model")
print("Model Performance (Top 3 Features Only):\n", results_top3_df)

# (New) Optional: save a compact table for the Top-3 experiment
plt.figure(figsize=(8.5, 2 + 0.38 * len(results_top3_df)))
plt.axis("off")
tbl2 = plt.table(
    cellText=results_top3_df.round(3).values,
    rowLabels=results_top3_df.index,
    colLabels=results_top3_df.columns,
    cellLoc='center',
    loc='center'
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(10)
tbl2.scale(1, 1.35)
plt.title("Model Performance — Top 3 Features Only", pad=12)
savefig("metrics_table_top3")

# -------------------- End --------------------
print("\nSaved figures to ./figures (PNG + SVG):")
for f in [
    "roc_curves",
    "cm_grid",
    "feature_importance_top10_rf",
    "metrics_table",
    "class_balance_before_after",
    "metrics_table_top3",
]:
    print(f" - {f}.png / {f}.svg")
