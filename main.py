import pandas as pd

# Load dataset from UCI URL
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    names=columns,
    na_values='?',
    skipinitialspace=True
)

# Before cleaning
print("Before Cleaning:")
print(f"Total rows: {df.shape[0]}")
print("Missing values per column:")
print(df.isnull().sum())
print(f"Total missing rows: {df.isnull().any(axis=1).sum()}")
print()

# Drop unnecessary columns
df.drop(columns=['fnlwgt', 'education'], inplace=True)
# Drop rows with missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# After cleaning
print("After Cleaning:")
print(f"Total rows: {df.shape[0]}")
print("Missing values per column:")
print(df.isnull().sum())

# Show class distribution
income_counts = df['income'].value_counts()
income_percentages = df['income'].value_counts(normalize=True) * 100

income_summary = pd.DataFrame({
    'count': income_counts,
    'percentage': income_percentages.round(2)
})

print("Income Class Distribution:")
print(income_summary)
