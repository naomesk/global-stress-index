"""
Step 5 of the project: natural-disaster deaths (EM-DAT via Our World in Data)
Folder layout (same as your other cleaning scripts):
    clean_disasters.py
    data/raw/annual-deaths-from-all-natural-disasters.csv   <- from OWID (Download full data)
    data/clean/climate_full_cleaned.csv                     <- made earlier (needs owid_population)
    data/clean/                                             <- output (created automatically)

Run:  python clean_disasters.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

RAW_FILE = "data/raw/annual-deaths-from-all-natural-disasters.csv"
CLIMATE_FILE = "data/clean/climate_full_cleaned.csv"   # only used for world population
START_YEAR, END_YEAR = 1959, 2024    # 2025 is a partial year in EM-DAT, so we stop at 2024

os.makedirs("data/clean", exist_ok=True)

# ---------- 1. Load and keep the World rows ----------
df = pd.read_csv(RAW_FILE)
print("Raw shape:", df.shape)
df.columns = df.columns.str.strip()
df = df[df["Entity"] == "World"].copy()
print("World rows:", df.shape, "| years", df["Year"].min(), "to", df["Year"].max())

df = df.rename(columns={"Year": "year", "All disasters": "disaster_deaths"})
df = df.drop_duplicates(subset="year").set_index("year").sort_index()

# ---------- 2. Keep the total plus two types (types are used for the report only) ----------
types = ["Droughts", "Earthquakes", "Floods", "Storms", "Extreme temperatures"]
df = df[["disaster_deaths"] + types]
df.columns = [c.lower().replace(" ", "_") for c in df.columns]
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ---------- 3. Common yearly axis ----------
df = df.reindex(range(START_YEAR, END_YEAR + 1))
print("Missing values in project years:\n", df.isna().sum())

# In EM-DAT a blank cell means "no events recorded" (zero deaths), so here
# 0 is correct. We do NOT interpolate, because a disaster year is not
# "between" its neighbours. The total column has no gaps at all.
df = df.fillna(0)
assert df["disaster_deaths"].isna().sum() == 0

# ---------- 4. Per million people (population from your OWID climate table) ----------
pop = pd.read_csv(CLIMATE_FILE, index_col="year")["owid_population"]
pop = pop.reindex(df.index)
assert pop.isna().sum() == 0, "population is missing for some years"
df["disaster_deaths_per_million"] = df["disaster_deaths"] / pop * 1e6

# ---------- 5. Skewness: a few huge years dominate (log transform) ----------
# One bad year can have 100x the deaths of a normal year. Without a log,
# PCA would just learn "which year had the worst disaster".
skew_before = df["disaster_deaths_per_million"].skew()
df["disaster_log_per_million"] = np.log1p(df["disaster_deaths_per_million"])
skew_after = df["disaster_log_per_million"].skew()
print(f"Skewness of deaths per million: {skew_before:.2f} -> after log1p: {skew_after:.2f}")

# ---------- 6. Outliers (IQR rule on the log scale, flag only, we keep them) ----------
col = df["disaster_log_per_million"]
q1, q3 = col.quantile([0.25, 0.75])
iqr = q3 - q1
mask = (col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)
print("Possible outliers on the log scale:", int(mask.sum()), list(col.index[mask]))
print("Top 5 years by deaths:\n", df["disaster_deaths"].sort_values(ascending=False).head(5))

# ---------- 7. Correlation with the climate columns (Lecture 3: Pearson r) ----------
climate = pd.read_csv(CLIMATE_FILE, index_col="year")[["temp_anomaly", "co2_ppm"]]
check = df[["disaster_log_per_million"]].join(climate)
print("Pearson correlation with climate:\n", check.corr(method="pearson").round(3))

# ---------- 8. Save ----------
df.to_csv("data/clean/disasters_cleaned.csv")      # file used by end_of_the_world.py

z = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)
z.to_csv("data/clean/disasters_zscore.csv")
mm = pd.DataFrame(MinMaxScaler().fit_transform(df), index=df.index, columns=df.columns)
mm.to_csv("data/clean/disasters_minmax.csv")

print(df.describe().round(2))
print("Done. Main file: data/clean/disasters_cleaned.csv")
