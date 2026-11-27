
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

# Step 1: Load encoded datasets
X_train = pd.read_csv("data/classification_train_encoded.csv")
X_test = pd.read_csv("data/classification_test_encoded.csv")

# Step 2: Identify numerical columns
# One-hot encoded columns contain only 0 and 1.
numeric_cols = [
    col for col in X_train.columns
    if X_train[col].nunique() > 2
]

print("Numerical columns to scale:", len(numeric_cols))

# Step 3: Create StandardScaler
scaler = StandardScaler()

# Step 4: Fit and transform training data
X_train[numeric_cols] = scaler.fit_transform(
    X_train[numeric_cols]
)

# Step 5: Transform testing data
X_test[numeric_cols] = scaler.transform(
    X_test[numeric_cols]
)

# Step 6: Save scaler
joblib.dump(scaler, "data/standard_scaler.joblib")

# Step 7: Save scaled datasets
X_train.to_csv("data/classification_train_scaled.csv", index=False)
X_test.to_csv("data/classification_test_scaled.csv", index=False)

print("\nFeature Scaling Completed!")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("\nScaled Data Sample:")
print(X_train[numeric_cols].head())
