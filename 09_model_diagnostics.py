
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

X_test = pd.read_csv("data/classification_test_scaled.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

models = {
    "Decision Tree": "models/optimized_decision_tree.joblib",
    "Random Forest": "models/optimized_random_forest.joblib",
    "Logistic Regression": "models/optimized_logistic_regression.joblib",
    "XGBoost": "models/optimized_xgboost.joblib"
}

for name, path in models.items():

    print("\n==============================")
    print(name)
    print("==============================")

    model = joblib.load(path)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    print("Probability minimum:", probabilities.min())
    print("Probability maximum:", probabilities.max())
    print("Probability mean:", probabilities.mean())

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        zero_division=0
    ))

    print("ROC AUC:", roc_auc_score(y_test, probabilities))
