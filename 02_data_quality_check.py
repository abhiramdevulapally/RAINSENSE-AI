
import pandas as pd

# Load dataset
df = pd.read_csv("data/railway_brake_data.csv")

# 1. Check duplicate records
print("Duplicate Rows:", df.duplicated().sum())

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False))

# 3. Missing percentage
print("\nMissing Percentage:")
print((df.isnull().mean() * 100).round(2))

# 4. Check data types
print("\nData Types:")
print(df.dtypes)

# 5. Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# 6. Check categorical columns
print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns.tolist())

# 7. Check target distribution
print("\nTarget Distribution:")
print(df["maintenance_required"].value_counts())
