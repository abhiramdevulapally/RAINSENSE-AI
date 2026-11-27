import pandas as pd

# Load original cleaned dataset
df = pd.read_csv("data/railway_cleaned.csv")

target = "brake_pad_wear_percent"

# Correlation with target
correlations = df.corr(numeric_only=True)[target]
correlations = correlations.drop(target).sort_values(
    ascending=False,
    key=abs
)

print("\nFEATURE CORRELATION WITH BRAKE PAD WEAR")
print(correlations.to_string())

# Feature means
print("\nFEATURE MEANS BY WEAR LEVEL")

df["wear_group"] = pd.cut(
    df[target],
    bins=[-1, 25, 50, 75, 100],
    labels=["Low", "Moderate", "High", "Very High"]
)

print(
    df.groupby("wear_group", observed=True)
    .mean(numeric_only=True)["brake_pad_wear_percent"]
)

# Check possible target-related columns
print("\nFEATURES USED BY MODEL")
X = pd.read_csv("data/regression_X_train_scaled.csv")
print(X.columns.tolist())
X = pd.read_csv("data/regression_X_train_scaled.csv")
X = X.drop(columns=["risk_score"], errors="ignore")

print("Features used by model:")
print(X.columns.tolist())
print("Total features:", X.shape[1])