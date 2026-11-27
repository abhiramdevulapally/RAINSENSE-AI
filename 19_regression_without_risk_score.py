import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load data
X_train = pd.read_csv("data/regression_X_train_scaled.csv")
X_test = pd.read_csv("data/regression_X_test_scaled.csv")

y_train = pd.read_csv("data/regression_y_train.csv").squeeze()
y_test = pd.read_csv("data/regression_y_test.csv").squeeze()

# Remove risk_score
X_train = X_train.drop(columns=["risk_score"])
X_test = X_test.drop(columns=["risk_score"])

print("risk_score removed successfully!")
print("Remaining features:", X_train.shape[1])

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100, max_depth=6,
        learning_rate=0.1, random_state=42, n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name} without risk_score...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions)
    })

results_df = pd.DataFrame(results).sort_values("RMSE")

print("\nMODEL COMPARISON WITHOUT RISK SCORE")
print(results_df.to_string(index=False))

results_df.to_csv(
    "regression_comparison_without_risk_score.csv",
    index=False
)
