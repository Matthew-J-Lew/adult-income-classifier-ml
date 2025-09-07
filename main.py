# main.py
import os
import pandas as pd
import matplotlib.pyplot as plt

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    names=columns,
    na_values='?',
    skipinitialspace=True
)

print("Before Cleaning:")
print(f"Total rows: {df.shape[0]}")
print("Missing values per column:")
print(df.isnull().sum())
print(f"Total missing rows: {df.isnull().any(axis=1).sum()}")
print()

df.drop(columns=['fnlwgt', 'education'], inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("After Cleaning:")
print(f"Total rows: {df.shape[0]}")
print("Missing values per column:")
print(df.isnull().sum())

# Optional quick class distribution
os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(8, 5))
df['income'].value_counts().sort_index().plot(kind='bar')
plt.title("Income Class Distribution (Raw, Cleaned Dataset)")
plt.xlabel("Income Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/adult_class_distribution_raw_main.png", dpi=180)
plt.savefig("figures/adult_class_distribution_raw_main.svg")
plt.close()
