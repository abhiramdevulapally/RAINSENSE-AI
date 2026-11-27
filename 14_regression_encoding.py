import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load imputed data
X_train = pd.read_csv("data/regression_X_train_imputed.csv")
X_test = pd.read_csv("data/regression_X_test_imputed.csv")

# Identify categorical columns
categorical_cols = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("Categorical columns:", categorical_cols)

# One-hot encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

train_encoded = encoder.fit_transform(X_train[categorical_cols])
test_encoded = encoder.transform(X_test[categorical_cols])

# Convert encoded data into DataFrames
encoded_cols = encoder.get_feature_names_out(categorical_cols)

train_encoded_df = pd.DataFrame(
    train_encoded, columns=encoded_cols, index=X_train.index
)
test_encoded_df = pd.DataFrame(
    test_encoded, columns=encoded_cols, index=X_test.index
)

# Combine numerical and encoded categorical features
X_train = X_train.drop(columns=categorical_cols)
X_test = X_test.drop(columns=categorical_cols)

X_train = pd.concat([X_train, train_encoded_df], axis=1)
X_test = pd.concat([X_test, test_encoded_df], axis=1)

# Save
X_train.to_csv("data/regression_X_train_encoded.csv", index=False)
X_test.to_csv("data/regression_X_test_encoded.csv", index=False)

joblib.dump(encoder, "data/regression_encoder.joblib")

print("\nEncoding completed!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
