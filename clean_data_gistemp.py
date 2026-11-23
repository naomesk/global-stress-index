"""
Cleaning script for NASA GISTEMP (GLB.Ts+dSST.csv)
Folder layout:
    clean_data.py            <- this file
    data/raw/GLB.Ts+dSST.csv <- the downloaded file
    data/clean/              <- output (created automatically)
Run:  python clean_data.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

RAW_FILE = "data/raw/GLB.Ts+dSST.csv"
START_YEAR, END_YEAR = 1880, 2025

os.makedirs("data/clean", exist_ok=True)

# ---------- 1. Load (line 1 is a title, "***" means missing) ----------
df = pd.read_csv(RAW_FILE, skiprows=1, na_values="***")
print("Shape:", df.shape)
print(df.head())
print("Missing values per column:\n", df.isna().sum())

# ---------- 2. Tidy column names and keep what we need ----------
df.columns = df.columns.str.strip().str.lower()
df = df[["year", "j-d"]].copy()
df = df.rename(columns={"j-d": "temp_anomaly"})   # yearly average anomaly (deg C)

# ---------- 3. Basic cleaning ----------
df = df.drop_duplicates()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# ---------- 4. Common yearly axis ----------
df = df.groupby("year", as_index=False).mean()
df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]
df = df.set_index("year").reindex(range(START_YEAR, END_YEAR + 1))

# ---------- 5. Missing values (interpolate, do not fill with 0) ----------
df = df.interpolate(method="linear", limit=3, limit_direction="both")
print("Still missing after interpolation:\n", df.isna().sum())
df = df.dropna()

# ---------- 6. Outliers (IQR rule, flag only) ----------
q1, q3 = df["temp_anomaly"].quantile([0.25, 0.75])
iqr = q3 - q1
n_out = ((df["temp_anomaly"] < q1 - 1.5 * iqr) | (df["temp_anomaly"] > q3 + 1.5 * iqr)).sum()
print("Possible outliers:", n_out)

# ---------- 7. Save ----------
df.to_csv("data/clean/gistemp_cleaned.csv")

z = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)
z.to_csv("data/clean/gistemp_zscore.csv")          # Lecture 3: Z = (X - mu) / sigma

mm = pd.DataFrame(MinMaxScaler().fit_transform(df), index=df.index, columns=df.columns)
mm.to_csv("data/clean/gistemp_minmax.csv")

print(df.describe())
print("Done. Files saved in data/clean/")
