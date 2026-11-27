
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("data/railway_brake_data.csv")

# Step 2: Check dataset dimensions
print("Dataset Shape:", df.shape)

# Step 3: Display first 5 rows
print("\nFirst 5 Records:")
print(df.head())

# Step 4: Check column names
print("\nColumn Names:")
print(df.columns.tolist())

# Step 5: Check data types
print("\nDataset Information:")
df.info()

# Step 6: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 7: Check target distribution
print("\nMaintenance Required:")
print(df["maintenance_required"].value_counts())
