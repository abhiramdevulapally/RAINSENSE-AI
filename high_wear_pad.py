
import pandas as pd

df = pd.read_csv("data/railway_cleaned.csv")

high_wear = df[
    df["brake_pad_wear_percent"].between(85, 95)
]

if high_wear.empty:
    print("No records found between 85%, and 95%.")
else:
    print(high_wear.head(1).T)
