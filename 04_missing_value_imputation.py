
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Step 1: Load cleaned dataset
df = pd.read_csv("data/railway_cleaned.csv")

# Step 2: Separate target and features
X = df.drop(columns=["maintenance_required", "train_id"])
y = df["maintenance_required"]

# Step 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Records:", X_train.shape)
print("Testing Records:", X_test.shape)

# Step 4: Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include="number").columns
categorical_cols = X_train.select_dtypes(exclude="number").columns

# Step 5: Numerical missing value imputation
num_imputer = SimpleImputer(strategy="median")

X_train_num = pd.DataFrame(
    num_imputer.fit_transform(X_train[numerical_cols]),
    columns=numerical_cols,
    index=X_train.index
)

X_test_num = pd.DataFrame(
    num_imputer.transform(X_test[numerical_cols]),
    columns=numerical_cols,
    index=X_test.index
)

# Step 6: Categorical missing value imputation
cat_imputer = SimpleImputer(strategy="most_frequent")

X_train_cat = pd.DataFrame(
    cat_imputer.fit_transform(X_train[categorical_cols]),
    columns=categorical_cols,
    index=X_train.index
)

X_test_cat = pd.DataFrame(
    cat_imputer.transform(X_test[categorical_cols]),
    columns=categorical_cols,
    index=X_test.index
)

# Step 7: Combine processed features
X_train_clean = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_clean = pd.concat([X_test_num, X_test_cat], axis=1)

# Step 8: Verify missing values
print("\nMissing values in training data:",
      X_train_clean.isnull().sum().sum())

print("Missing values in testing data:",
      X_test_clean.isnull().sum().sum())

print("\nImputation completed successfully!")

# Save imputed training and testing datasets

train_data = X_train_clean.copy()
train_data["maintenance_required"] = y_train

test_data = X_test_clean.copy()
test_data["maintenance_required"] = y_test

train_data.to_csv(
    "data/classification_train_imputed.csv",
    index=False
)

test_data.to_csv(
    "data/classification_test_imputed.csv",
    index=False
)

print("Imputed datasets saved successfully!")
