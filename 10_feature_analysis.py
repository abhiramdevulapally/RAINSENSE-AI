
import pandas as pd

# Load original dataset
df = pd.read_csv("data/railway_cleaned.csv")

target = "maintenance_required"

# Numerical feature correlation
correlations = df.corr(numeric_only=True)[target]
correlations = correlations.drop(target).sort_values(
    ascending=False,
    key=abs
)

print("\nFEATURE CORRELATION WITH MAINTENANCE")
print(correlations.to_string())

# Target distribution
print("\nTARGET DISTRIBUTION")
print(df[target].value_counts(normalize=True))

# Compare numerical features by target
print("\nFEATURE MEANS BY MAINTENANCE CLASS")
print(
    df.groupby(target).mean(numeric_only=True).T
)

# Check categorical relationships
for col in df.select_dtypes(include="object").columns:
    print(f"\n{col} vs {target}")
    print(pd.crosstab(df[col], df[target], normalize="index").round(3))
