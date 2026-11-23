"""
Step 4 of the project: birth rate, death rate and life expectancy (World Bank)
Folder layout (same as your other cleaning scripts):
    clean_demography.py
    data/raw/   <- the 3 World Bank CSV files you downloaded
    data/clean/ <- output (created automatically)

Download the 3 files from the World Bank (use the "CSV" download button):
    birth rate, crude (per 1,000 people)  SP.DYN.CBRT.IN
    death rate, crude (per 1,000 people)  SP.DYN.CDRT.IN
    life expectancy at birth, total       SP.DYN.LE00.IN
Page address:  https://data.worldbank.org/indicator/SP.DYN.CBRT.IN  (change the code at the end for the others)
Each zip contains a file named like API_SP.DYN.CBRT.IN_DS2_en_csv_v2_XXXXXX.csv.
Put the 3 API_...csv files in data/raw/ (you do not need to rename them).

Run:  python clean_demography.py
"""
import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

RAW_DIR = "data/raw"
START_YEAR, END_YEAR = 1959, 2024
INDICATORS = {                       # indicator code -> our column name
    "SP.DYN.CBRT.IN": "birth_rate",
    "SP.DYN.CDRT.IN": "death_rate",
    "SP.DYN.LE00.IN": "life_expectancy",
}

os.makedirs("data/clean", exist_ok=True)


# ---------- 1. Load one World Bank file -> yearly Series for the World ----------
def load_world_bank(code, name):
    files = glob.glob(f"{RAW_DIR}/API_{code}*.csv")
    if not files:
        raise FileNotFoundError(f"No file like {RAW_DIR}/API_{code}*.csv found")
    # first 4 lines are a title block, line 5 is the header
    df = pd.read_csv(files[0], skiprows=4)
    df.columns = df.columns.str.strip()
    print(f"{code}: shape {df.shape}")
    world = df[df["Country Code"] == "WLD"]            # World aggregate
    assert len(world) == 1, "Could not find the World (WLD) row"
    year_cols = [c for c in df.columns if c.isdigit()]
    s = world[year_cols].T.squeeze()                  # years become the index
    s.index = s.index.astype(int)
    s = pd.to_numeric(s, errors="coerce").rename(name)
    s.index.name = "year"
    return s


series = [load_world_bank(code, name) for code, name in INDICATORS.items()]
df = pd.concat(series, axis=1)
print(df.head())
print("Missing values per column (all years):\n", df.isna().sum())

# ---------- 2. Basic cleaning ----------
df = df.drop_duplicates()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ---------- 3. Common yearly axis ----------
df = df.reindex(range(START_YEAR, END_YEAR + 1))
print("Missing values in the project years:\n", df.isna().sum())

# Warn if the newest years are missing (the World Bank often lags 1-2 years).
# Interpolation cannot invent future values; it would just repeat the last one.
for c in df.columns:
    first = df[c].first_valid_index()
    if first > START_YEAR:
        print(f"WARNING: {c} starts in {first}. Years {START_YEAR}-{first - 1} "
              f"will be a copy of {first} (World Bank data usually starts in 1960).")
    last = df[c].last_valid_index()
    if last < END_YEAR:
        print(f"WARNING: {c} ends in {last}. Years {last + 1}-{END_YEAR} "
              f"will be a copy of {last} (say this in your report), or set END_YEAR = {last}.")

# ---------- 4. Missing values (interpolate, do not fill with 0) ----------
df = df.interpolate(method="linear", limit=3, limit_direction="both")
print("Still missing after interpolation:\n", df.isna().sum())
df = df.dropna()

# ---------- 5. Outliers (IQR rule, flag only) ----------
for c in df.columns:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    print(f"Possible outliers in {c}: {n_out}")

# ---------- 6. Correlation (Lecture 3: Pearson r) ----------
print("Pearson correlation:\n", df.corr(method="pearson").round(3))

# ---------- 7. Save ----------
df.to_csv("data/clean/demography_cleaned.csv")
# the file used by end_of_the_world.py
df.to_csv("data/demography_world.csv")

z = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)
z.to_csv("data/clean/demography_zscore.csv")
mm = pd.DataFrame(MinMaxScaler().fit_transform(df), index=df.index, columns=df.columns)
mm.to_csv("data/clean/demography_minmax.csv")

print(df.describe())
print("Done. Main file: data/clean/demography_cleaned.csv (copy also in data/demography_world.csv)")
