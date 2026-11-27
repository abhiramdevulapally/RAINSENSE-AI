
import pandas as pd

# Step 1: Load original dataset
df = pd.read_csv("data/railway_brake_data.csv")

print("Original Dataset:", df.shape)

# Step 2: Remove duplicate records
df = df.drop_duplicates()

print("After Removing Duplicates:", df.shape)

# Step 3: Remove columns unavailable at prediction time
df = df.drop(columns=[
    "failure_type",
    "failure_severity"
])

# Step 4: Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sort_values(ascending=False))

# Step 5: Save cleaned dataset
df.to_csv("data/railway_cleaned.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")
print("Final Dataset Shape:", df.shape)
