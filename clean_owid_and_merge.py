"""
Step 3 of the climate part:
 1) clean OWID CO2 data (World rows only)  (data/raw/owid-co2-data.csv)
 2) merge with climate_cleaned.csv (temperature + Mauna Loa CO2)
Run:  python clean_owid_and_merge.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

OWID_FILE = "data/raw/owid-co2-data.csv"
CLIMATE_FILE = "data/clean/climate_cleaned.csv"   # made by clean_co2_and_merge.py

# columns we would like (the script keeps only those that exist)
WANTED = ["year", "co2", "co2_per_capita", "methane",
          "nitrous_oxide", "total_ghg", "population"]
MAX_MISSING = 0.20      # drop a column if more than 20% is missing in 1959+

os.makedirs("data/clean", exist_ok=True)

# ---------- 1. Load and keep the World rows ----------
df = pd.read_csv(OWID_FILE)
df.columns = df.columns.str.strip().str.lower()
print("Raw shape:", df.shape)
df = df[df["country"] == "World"].copy()
print("World rows:", df.shape)

# ---------- 2. Keep useful columns ----------
cols = [c for c in WANTED if c in df.columns]
print("Columns found:", cols)
df = df[cols].drop_duplicates()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)
df = df.groupby("year", as_index=False).mean().set_index("year")

# ---------- 3. Match the project years (1959 onward) ----------
df = df[df.index >= 1959]
print("Missing share per column:\n", df.isna().mean().round(2))
df = df.loc[:, df.isna().mean() <= MAX_MISSING]    # drop mostly-empty columns
df = df.interpolate(method="linear", limit=3, limit_direction="both")
df = df.dropna()
df = df.add_prefix("owid_")
df.to_csv("data/clean/owid_cleaned.csv")

# ---------- 4. Merge with the existing climate table ----------
climate = pd.read_csv(CLIMATE_FILE, index_col="year")
full = climate.join(df, how="inner")
print("Merged shape:", full.shape)
print("Years:", full.index.min(), "to", full.index.max())
full.to_csv("data/clean/climate_full_cleaned.csv")

# ---------- 5. Correlation (very high pairs = redundant features) ----------
corr = full.corr(method="pearson").round(2)
print(corr)

# ---------- 6. Scaled versions ----------
z = pd.DataFrame(StandardScaler().fit_transform(full),
                 index=full.index, columns=full.columns)
z.to_csv("data/clean/climate_full_zscore.csv")
mm = pd.DataFrame(MinMaxScaler().fit_transform(full),
                  index=full.index, columns=full.columns)
mm.to_csv("data/clean/climate_full_minmax.csv")

print("Done. Main file: data/clean/climate_full_cleaned.csv")
