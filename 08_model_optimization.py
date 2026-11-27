
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix
)

from xgboost import XGBClassifier

# Step 1: Load processed datasets
X_train = pd.read_csv("data/classification_train_scaled.csv")
X_test = pd.read_csv("data/classification_test_scaled.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Step 2: Split training data into training and validation
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# Step 3: Calculate class imbalance ratio
negative = (y_fit == 0).sum()
positive = (y_fit == 1).sum()

scale_ratio = negative / positive

print("\nClass distribution:")
print("No maintenance:", negative)
print("Maintenance required:", positive)
print("Imbalance ratio:", round(scale_ratio, 2))

# Step 4: Define optimized models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=15,
        min_samples_split=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_ratio,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )
}

# Step 5: Train and optimize decision thresholds
results = []
os.makedirs("models", exist_ok=True)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_fit, y_fit)

    # Find the best threshold using validation data
    val_prob = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_prob
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = np.argmax(f1_values)
    best_threshold = thresholds[best_index]

    # Evaluate on untouched test data
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, test_pred)
    precision_score_value = precision_score(
        y_test, test_pred, zero_division=0
    )
    recall_score_value = recall_score(
        y_test, test_pred, zero_division=0
    )
    f1 = f1_score(y_test, test_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision_score_value,
        "Recall": recall_score_value,
        "F1 Score": f1,
        "Threshold": best_threshold
    })

    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        f"models/optimized_{filename}.joblib"
    )

    print("Best threshold:", round(best_threshold, 4))
    print("Test F1-score:", round(f1, 4))
    print("Test Recall:", round(recall_score_value, 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

# Step 6: Compare optimized models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== OPTIMIZED MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Step 7: Save results
results_df.to_csv(
    "optimized_model_comparison.csv",
    index=False
)

print("\nOptimization completed successfully!")
print("Models saved inside models folder.")
print("Comparison report saved successfully!")
