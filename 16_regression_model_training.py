
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load scaled features and target
X_train = pd.read_csv("data/regression_X_train_scaled.csv")
X_test = pd.read_csv("data/regression_X_test_scaled.csv")

y_train = pd.read_csv("data/regression_y_train.csv").squeeze()
y_test = pd.read_csv("data/regression_y_test.csv").squeeze()




print("Features used:", X_train.shape[1])
print("Risk score included successfully!")

# Define models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
}

results = []

# Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    filename = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{filename}_no_risk_regressor.joblib")

# Compare results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE")

print("\nREGRESSION MODEL COMPARISON WITH RISK SCORE")
print(results_df.to_string(index=False))

results_df.to_csv(
    "regression_model_comparison_no_risk.csv",
    index=False
)

print("\nAll models trained successfully!")
