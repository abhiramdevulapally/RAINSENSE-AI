
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

X = pd.read_csv("data/regression_X_train_scaled.csv")
y = pd.read_csv("data/regression_y_train.csv").squeeze()

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("FOLD-WISE VALIDATION")

for i, (train_idx, test_idx) in enumerate(cv.split(X), 1):
    model = LinearRegression()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    pred = model.predict(X.iloc[test_idx])
    actual = y.iloc[test_idx]

    print(f"\nFold {i}")
    print("Target SD:", actual.std())
    print("MAE:", mean_absolute_error(actual, pred))
    print("RMSE:", np.sqrt(mean_squared_error(actual, pred)))
    print("R2:", r2_score(actual, pred))
