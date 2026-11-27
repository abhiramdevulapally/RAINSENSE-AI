
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the same train/test split
X_train = pd.read_csv("data/regression_X_train_scaled.csv")
X_test = pd.read_csv("data/regression_X_test_scaled.csv")

y_train = pd.read_csv("data/regression_y_train.csv").squeeze()
y_test = pd.read_csv("data/regression_y_test.csv").squeeze()

# Compare identical splits and targets
experiments = {
    "With risk_score": (X_train, X_test),
    "Without risk_score": (
        X_train.drop(columns=["risk_score"], errors="ignore"),
        X_test.drop(columns=["risk_score"], errors="ignore")
    )
}

results = []

for name, (Xtr, Xte) in experiments.items():
    model = LinearRegression()
    model.fit(Xtr, y_train)

    train_pred = model.predict(Xtr)
    test_pred = model.predict(Xte)

    results.append({
        "Experiment": name,
        "Features": Xtr.shape[1],
        "Train R2": r2_score(y_train, train_pred),
        "Test R2": r2_score(y_test, test_pred),
        "MAE": mean_absolute_error(y_test, test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, test_pred))
    })

result_df = pd.DataFrame(results)
print("\nFAIR MODEL COMPARISON")
print(result_df.to_string(index=False))

result_df.to_csv("risk_score_experiment_comparison.csv", index=False)
print("\nSaved risk_score_experiment_comparison.csv")

