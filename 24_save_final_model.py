
import pandas as pd
import joblib
import json

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

X = pd.read_csv("data/regression_X_train_scaled.csv")
y = pd.read_csv("data/regression_y_train.csv").squeeze()

model = LinearRegression()
model.fit(X, y)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

joblib.dump(model, "models/final_brake_wear_model.joblib")

metrics = {
    "model": "Linear Regression",
    "features": X.shape[1],
    "cross_validation_r2_mean": float(scores.mean()),
    "cross_validation_r2_std": float(scores.std()),
    "cross_validation_r2_percent": float(scores.mean() * 100)
}

with open("models/final_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("FINAL MODEL SAVED!")
print(metrics)
