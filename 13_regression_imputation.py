import pandas as pd
from sklearn.impute import SimpleImputer
import joblib

# Load data
X_train = pd.read_csv("data/regression_X_train.csv")
X_test = pd.read_csv("data/regression_X_test.csv")

# Identify numerical and categorical columns
numeric_cols = X_train.select_dtypes(include="number").columns
categorical_cols = X_train.select_dtypes(exclude="number").columns

# Numerical missing values: Median
num_imputer = SimpleImputer(strategy="median")

X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

# Categorical missing values: Most frequent
cat_imputer = SimpleImputer(strategy="most_frequent")

if len(categorical_cols) > 0:
    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Save files
X_train.to_csv("data/regression_X_train_imputed.csv", index=False)
X_test.to_csv("data/regression_X_test_imputed.csv", index=False)

joblib.dump(num_imputer, "data/regression_num_imputer.joblib")
if len(categorical_cols) > 0:
    joblib.dump(cat_imputer, "data/regression_cat_imputer.joblib")

print("Missing values handled successfully!")
print("Remaining training missing values:", X_train.isnull().sum().sum())
print("Remaining testing missing values:", X_test.isnull().sum().sum())