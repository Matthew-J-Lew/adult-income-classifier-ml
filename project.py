# Basic Imports for Data Analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
import numpy as np

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
X = df.drop("income", axis = 1)
y = df["income"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training data and testing data, 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

# Apply SMOTE to training data (from task 1)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Calling the different models (logistic regression and SVM use class_weight='balanced' from task 1 to solve class imbalance)
models = {
    "Logistic Regression (weighted)": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest (weighted)": RandomForestClassifier(class_weight='balanced'),
    "Naive Bayes": GaussianNB(),
    "SVM (weighted)": SVC(class_weight='balanced'),
    "k-NN": KNeighborsClassifier()
}

# Evalulate all models using the training data (SMOTE) against the test data
results = []
for name, model in models.items():
    # Train on SMOTE-resampled training data
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

# Display results
results_df = pd.DataFrame(results).set_index("Model")
print("\nModel Performance (with Class Balancing):\n", results_df)

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
X_train_top3, X_test_top3, y_train_top3, y_test_top3 = train_test_split(X_top3_scaled, y, test_size = 0.2, random_state = 42)

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