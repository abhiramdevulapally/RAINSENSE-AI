
import pandas as pd

y_train = pd.read_csv("data/regression_y_train.csv").squeeze()
y_test = pd.read_csv("data/regression_y_test.csv").squeeze()

print("TRAIN TARGET")
print(y_train.describe())

print("\nTEST TARGET")
print(y_test.describe())

print("\nTRAIN UNIQUE VALUES:", y_train.nunique())
print("TEST UNIQUE VALUES:", y_test.nunique())

print("\nFIRST 15 TRAIN VALUES:")
print(y_train.head(15).to_list())
