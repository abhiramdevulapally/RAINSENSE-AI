import pandas as pd

# Load training and testing data
X_train = pd.read_csv("data/regression_X_train.csv")
X_test = pd.read_csv("data/regression_X_test.csv")

print("TRAINING MISSING VALUES")
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

print("\nTESTING MISSING VALUES")
print(X_test.isnull().sum()[X_test.isnull().sum() > 0])

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)