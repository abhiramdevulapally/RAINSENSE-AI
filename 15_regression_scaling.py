import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load encoded datasets
X_train = pd.read_csv("data/regression_X_train_encoded.csv")
X_test = pd.read_csv("data/regression_X_test_encoded.csv")

# Select numerical columns
numeric_cols = X_train.select_dtypes(include="number").columns.tolist()

# Scale features
scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Save scaled datasets
X_train.to_csv("data/regression_X_train_scaled.csv", index=False)
X_test.to_csv("data/regression_X_test_scaled.csv", index=False)

joblib.dump(scaler, "data/regression_scaler.joblib")

print("Feature scaling completed!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Number of scaled features:", len(numeric_cols))
