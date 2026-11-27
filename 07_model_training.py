
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Step 1: Load scaled datasets
X_train = pd.read_csv("data/classification_train_scaled.csv")
X_test = pd.read_csv("data/classification_test_scaled.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Step 2: Temporarily exclude risk_score
X_train = X_train.drop(columns=["risk_score"])
X_test = X_test.drop(columns=["risk_score"])

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Step 3: Calculate class imbalance ratio
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_ratio = negative / positive

print("\nClass Distribution:")
print("No Maintenance:", negative)
print("Maintenance Required:", positive)

# Step 4: Define ML models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_ratio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# Step 5: Create models folder
os.makedirs("models", exist_ok=True)

results = []

# Step 6: Train and evaluate each model
for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save trained model
    filename = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{filename}.joblib")

# Step 7: Compare all models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Step 8: Save comparison report
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nAll models trained successfully!")
print("Models saved inside models folder.")
print("Comparison report saved successfully!")
