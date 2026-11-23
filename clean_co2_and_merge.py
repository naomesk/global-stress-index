"""
Step 2 of the climate part:
 1) clean NOAA Mauna Loa annual CO2  (data/raw/co2_annmean_mlo.csv)
 2) merge it with the cleaned GISTEMP file you already made
Run:  python clean_co2_and_merge.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

CO2_FILE = "data/raw/co2_annmean_mlo.csv"
TEMP_FILE = "data/clean/gistemp_cleaned.csv"   # made by clean_data_gistemp.py

os.makedirs("data/clean", exist_ok=True)

# ---------- 1. Load CO2 (lines starting with # are comments) ----------
co2 = pd.read_csv(CO2_FILE, comment="#")
co2.columns = co2.columns.str.strip().str.lower()
print(co2.head())
print("Missing values:\n", co2.isna().sum())

# ---------- 2. Clean CO2 ----------
co2 = co2[["year", "mean"]].rename(columns={"mean": "co2_ppm"})
co2 = co2.drop_duplicates()
co2.replace([np.inf, -np.inf], np.nan, inplace=True)
for c in co2.columns:
    co2[c] = pd.to_numeric(co2[c], errors="coerce")
co2 = co2.dropna(subset=["year"])
co2["year"] = co2["year"].astype(int)
co2 = co2.groupby("year", as_index=False).mean()
co2 = co2.set_index("year")
co2 = co2.interpolate(method="linear", limit=3, limit_direction="both")
co2 = co2.dropna()
co2.to_csv("data/clean/co2_cleaned.csv")

# ---------- 3. Load cleaned temperature and merge on year ----------
temp = pd.read_csv(TEMP_FILE, index_col="year")
climate = temp.join(co2, how="inner")      # keep only years present in BOTH
print("Merged shape:", climate.shape)
print("Years:", climate.index.min(), "to", climate.index.max())
climate.to_csv("data/clean/climate_cleaned.csv")

# ---------- 4. Correlation (Lecture 3: Pearson r) ----------
print("Pearson correlation:\n", climate.corr(method="pearson").round(3))

# ---------- 5. Scaled versions ----------
z = pd.DataFrame(StandardScaler().fit_transform(climate),
                 index=climate.index, columns=climate.columns)
z.to_csv("data/clean/climate_zscore.csv")

mm = pd.DataFrame(MinMaxScaler().fit_transform(climate),
                  index=climate.index, columns=climate.columns)
mm.to_csv("data/clean/climate_minmax.csv")

print(climate.describe())
print("Done. Main file for your leader: data/clean/climate_cleaned.csv")
