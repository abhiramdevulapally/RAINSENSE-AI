
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load imputed datasets
train = pd.read_csv("data/classification_train_imputed.csv")
test = pd.read_csv("data/classification_test_imputed.csv")

# Step 2: Separate target variable
X_train = train.drop(columns=["maintenance_required"])
X_test = test.drop(columns=["maintenance_required"])

y_train = train["maintenance_required"]
y_test = test["maintenance_required"]

# Step 3: Identify categorical columns
categorical_cols = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("Categorical Columns:", categorical_cols)

# Step 4: Create OneHotEncoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Step 5: Fit and transform training data
train_encoded = encoder.fit_transform(
    X_train[categorical_cols]
)

# Step 6: Transform testing data
test_encoded = encoder.transform(
    X_test[categorical_cols]
)

# Step 7: Convert encoded arrays into DataFrames
encoded_columns = encoder.get_feature_names_out(
    categorical_cols
)

train_encoded_df = pd.DataFrame(
    train_encoded,
    columns=encoded_columns
)

test_encoded_df = pd.DataFrame(
    test_encoded,
    columns=encoded_columns
)

# Step 8: Keep numerical columns
X_train_num = X_train.drop(columns=categorical_cols).reset_index(drop=True)
X_test_num = X_test.drop(columns=categorical_cols).reset_index(drop=True)

# Step 9: Combine numerical and encoded features
X_train_final = pd.concat(
    [X_train_num, train_encoded_df], axis=1
)

X_test_final = pd.concat(
    [X_test_num, test_encoded_df], axis=1
)

# Step 10: Save processed datasets
X_train_final.to_csv("data/classification_train_encoded.csv", index=False)
X_test_final.to_csv("data/classification_test_encoded.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nEncoding Completed Successfully!")
print("Training Shape:", X_train_final.shape)
print("Testing Shape:", X_test_final.shape)
