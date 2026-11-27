import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load cleaned dataset
df = pd.read_csv("data/railway_cleaned.csv")

# Step 2: Select target
target = "brake_pad_wear_percent"

# Step 3: Remove rows where target is missing
df = df.dropna(subset=[target])

# Step 4: Separate features and target
X = df.drop(columns=[
    target,
    "maintenance_required",
    "train_id"
], errors="ignore")

y = df[target]

# Step 5: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 6: Save datasets
X_train.to_csv("data/regression_X_train.csv", index=False)
X_test.to_csv("data/regression_X_test.csv", index=False)
y_train.to_csv("data/regression_y_train.csv", index=False)
y_test.to_csv("data/regression_y_test.csv", index=False)

print("Regression data preparation completed!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])
print("Target:", target)