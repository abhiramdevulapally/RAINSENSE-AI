import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 1: Load best model
model = joblib.load("models/linear_regression_regressor.joblib")

# Step 2: Load test data
X_test = pd.read_csv("data/regression_X_test_scaled.csv")
y_test = pd.read_csv("data/regression_y_test.csv").squeeze()

# Step 3: Predict
predictions = model.predict(X_test)

# Step 4: Calculate metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nREGRESSION MODEL DIAGNOSTICS")
print("----------------------------")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Step 5: Actual vs predicted sample
comparison = pd.DataFrame({
    "Actual Wear (%)": y_test.values[:20],
    "Predicted Wear (%)": predictions[:20]
})

print("\nACTUAL VS PREDICTED")
print(comparison.round(2).to_string(index=False))

# Save predictions
comparison.to_csv("regression_predictions_sample.csv", index=False)

# Step 6: Actual vs predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.4)
plt.xlabel("Actual Brake Pad Wear (%)")
plt.ylabel("Predicted Brake Pad Wear (%)")
plt.title("Actual vs Predicted Brake Pad Wear")
plt.savefig("actual_vs_predicted.png")
plt.close()

# Step 7: Residual analysis
residuals = y_test - predictions

plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals, alpha=0.4)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Wear (%)")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.savefig("regression_residuals.png")
plt.close()

print("\nDiagnostics completed!")
print("Graphs saved successfully!")
