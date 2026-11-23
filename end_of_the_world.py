# End of the World: Global Stress Index and Peak Forecasting
# Introduction to Machine Learning (Fall 2026)
#
# In VS Code: open this file, then click "Run Cell" above any "# %%" line
# (or "Run All"). Results show in the Interactive Window. Figures and tables
# are saved in the "outputs" folder.

# %% 0. Setup ------------------------------------------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

START, END = 1959, 2024          # common yearly axis (OWID methane/N2O end in 2024)
TRAIN_END = 2015                 # train: <= 2015, test: 2016-2025
HORIZON = 2100                   # forecast horizon

# How the Global Stress Index is built:
#   "domains" = PCA compresses the redundant CLIMATE indicators into one score,
#               then climate, mortality, disasters and pandemic count equally
#               (every indicator is oriented so higher = more stress).
#   "pca"     = one PCA over all indicators (PC1 ends up as a time trend).
INDEX_MODE = "domains"
DOMAINS = {
    "climate": ["temp_anomaly", "co2_ppm", "co2_per_capita", "methane", "nitrous_oxide"],
    "mortality": ["death_rate"],
    "disasters": ["disaster_log_per_million"],
    "pandemic": ["covid_log_per_million"],
}

# File names inside the data/ folder. Change them to match your downloads.
FILES = {
    "climate": "clean/climate_full_cleaned.csv", # your cleaned climate table (1959-2025)
    "demography": "clean/demography_cleaned.csv",  # made by clean_demography.py
    "disasters": "clean/disasters_cleaned.csv",    # made by clean_disasters.py
    "covid": "raw/owid-covid-data.csv",        # OWID COVID-19 (raw file is read directly)
    "quakes": "clean/quakes_yearly.csv",         # optional: year,quake_count
}


def years(df):
    """Keep only the common yearly axis."""
    return df[(df.index >= START) & (df.index <= END)]


# %% 1. Load each source into a one-row-per-year table -------------------
def load_climate():
    """Your cleaned climate table. Drops redundant columns (r ~ 1.00 with others)."""
    d = pd.read_csv(DATA / FILES["climate"], index_col="year")
    d = d.rename(columns={"owid_co2_per_capita": "co2_per_capita",
                          "owid_methane": "methane",
                          "owid_nitrous_oxide": "nitrous_oxide",
                          "owid_population": "population"})
    # owid_co2 and owid_total_ghg are ~99-100% correlated with co2_ppm/each other
    return d.drop(columns=["owid_co2", "owid_total_ghg"], errors="ignore")


def load_simple(key):
    """CSV you prepared yourself, with a 'year' column."""
    path = DATA / FILES[key]
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    return pd.read_csv(path).set_index("year")


def load_covid(population_by_year):
    d = pd.read_csv(DATA / FILES["covid"], usecols=["location", "date", "new_deaths"])
    d = d[d["location"] == "World"].copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    deaths = d.groupby("year")["new_deaths"].sum()
    pop = population_by_year.reindex(deaths.index)
    return pd.DataFrame({"covid_deaths_per_million": deaths / pop * 1e6})


parts = [load_climate()]
demo = load_simple("demography")
dis = load_simple("disasters")
quakes = load_simple("quakes")
if dis is not None:
    # clean_disasters.py already made the per-million, log-scaled column
    dis = dis[["disaster_log_per_million"]]
for p in (demo, dis, quakes):
    if p is not None:
        # population comes from your climate table (OWID), so ignore duplicates
        parts.append(p.drop(columns=["population"], errors="ignore"))

# %% 2. Merge, fill gaps, per-capita scaling -----------------------------
df = pd.concat(parts, axis=1)
df = years(df)

# Population for per-capita scaling (prefer your demography file, else OWID)
if "population" in df.columns:
    population = df["population"].interpolate(limit_direction="both")
else:
    population = pd.Series(np.nan, index=df.index)

# Pandemic: COVID only exists from 2020, so earlier years are 0
if (DATA / FILES["covid"]).exists():
    covid = load_covid(population)
    df = df.join(covid)
    df["covid_deaths_per_million"] = df["covid_deaths_per_million"].fillna(0)
    # one-year spike: log scale (same reason as disasters) so it does not swamp everything
    df["covid_log_per_million"] = np.log1p(df.pop("covid_deaths_per_million"))

# Per-capita versions of disaster totals
for col in ("disaster_deaths", "disaster_affected", "disaster_damage"):
    if col in df.columns:
        df[col + "_per_million"] = df[col] / population * 1e6
        df = df.drop(columns=col)
df = df.drop(columns=["population", "co2_emissions"], errors="ignore")  # co2_per_capita kept

# Redundant columns: life_expectancy has r = -0.96 / -0.97 with birth and death
# rate (checked in clean_demography.py), so it would count the same trend 3 times.
DROP_COLS = ["life_expectancy"]
if INDEX_MODE == "domains":
    # birth rate has no clear "stress" direction (a falling birth rate is not
    # obviously bad), so it is not part of the stress index in this mode
    DROP_COLS.append("birth_rate")
df = df.drop(columns=DROP_COLS, errors="ignore")

print("Missing values before filling:\n", df.isna().sum())
df = df.interpolate(limit_direction="both")
assert df.isna().sum().sum() == 0, "Still missing values, check your files"
df.to_csv(OUT / "merged_yearly_table.csv")
print(df.shape)
df.head()

# %% 3. Global Stress Index -----------------------------------------------
features = list(df.columns)

# Make every indicator point the same way: higher value = more stress.
# Life expectancy going UP is good, so flip it. Change this dict if you
# disagree (and explain your choice in the report).
DIRECTION = {"life_expectancy": -1}
for c, s in DIRECTION.items():
    if c in df.columns:
        df[c] = s * df[c]

scaler = StandardScaler().fit(df[features])
Z = pd.DataFrame(scaler.transform(df[features]), index=df.index, columns=features)

if INDEX_MODE == "pca":
    pca = PCA(n_components=2).fit(Z)
    print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=["PC1", "PC2"])
    print(loadings.round(3))
    sign = 1 if loadings["PC1"].sum() >= 0 else -1

    def raw_index(frame):
        return sign * (scaler.transform(frame[features]) @ pca.components_[0])
else:
    cl = [c for c in DOMAINS["climate"] if c in features]
    pca_cl = PCA(n_components=1).fit(Z[cl])
    print("Climate PCA, explained variance:", pca_cl.explained_variance_ratio_.round(3))
    loadings = pd.DataFrame(pca_cl.components_.T, index=cl, columns=["PC1"])
    print(loadings.round(3))
    s_cl = 1 if loadings["PC1"].sum() >= 0 else -1          # higher = more climate stress
    cl_hist = s_cl * (Z[cl].values @ pca_cl.components_[0])
    cl_mu, cl_sd = cl_hist.mean(), cl_hist.std()

    def domain_scores(frame):
        zf = pd.DataFrame(scaler.transform(frame[features]), index=frame.index, columns=features)
        out = {"climate": (s_cl * (zf[cl].values @ pca_cl.components_[0]) - cl_mu) / cl_sd}
        for d, cols in DOMAINS.items():
            cols = [c for c in cols if c in features]
            if d != "climate" and cols:
                out[d] = zf[cols].mean(axis=1).values
        return pd.DataFrame(out, index=frame.index)

    def raw_index(frame):
        return domain_scores(frame).mean(axis=1).values      # equal weight per domain

    ds = domain_scores(df)
    ds.to_csv(OUT / "domain_scores.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ds.plot(ax=ax)
    ax.set(title="Stress score per domain (z-scores, higher = more stress)", xlabel="Year")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "domain_scores.png", dpi=150, bbox_inches="tight")

raw = raw_index(df)
index = pd.Series((raw - raw.min()) / (raw.max() - raw.min()) * 100, index=df.index, name="stress_index")
print(f"Observed peak of the index: {int(index.idxmax())} (value {index.max():.1f})")

loadings.to_csv(OUT / "pca_loadings.csv")
fig, ax = plt.subplots(figsize=(9, 4))
index.plot(ax=ax)
ax.set(title="Global Stress Index (0-100)", xlabel="Year", ylabel="Index")
ax.grid(alpha=0.3)
fig.savefig(OUT / "stress_index.png", dpi=150, bbox_inches="tight")

# %% 4. Calm / stressed / crisis with K-means ----------------------------
X = index.to_frame()
rows = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    rows.append((k, km.inertia_, silhouette_score(X, km.labels_)))
choice = pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])
print(choice.round(3))   # elbow plot uses inertia; proposal uses k=3

km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
order = pd.Series(km.cluster_centers_.ravel()).sort_values().index    # low -> high
names = {order[0]: "calm", order[1]: "stressed", order[2]: "crisis"}
period = pd.Series(km.labels_, index=df.index).map(names).rename("period")
print(period.value_counts())

colors = {"calm": "tab:green", "stressed": "tab:orange", "crisis": "tab:red"}
fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(index.index, index.values, c=period.map(colors), s=25)
ax.plot(index.index, index.values, color="gray", alpha=0.3)
ax.set(title="Calm / stressed / crisis periods", xlabel="Year", ylabel="Stress index")
fig.savefig(OUT / "periods.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(choice.k, choice.inertia, "o-"); ax[0].set(title="Elbow", xlabel="k")
ax[1].plot(choice.k, choice.silhouette, "o-"); ax[1].set(title="Silhouette", xlabel="k")
fig.savefig(OUT / "kmeans_choice.png", dpi=150, bbox_inches="tight")

# %% 5. Trend fitting: linear vs polynomial (time-based split) ----------
tr = index[index.index <= TRAIN_END]
te = index[index.index > TRAIN_END]
results = []
fits = {}
for name, deg in [("linear", 1), ("poly2", 2), ("poly3", 3)]:
    coef = np.polyfit(tr.index - START, tr.values, deg)
    pred = np.polyval(coef, te.index - START)
    results.append((name,
                    mean_absolute_error(te, pred),
                    np.sqrt(mean_squared_error(te, pred))))
    # refit on all years for forecasting
    fits[name] = np.polyfit(index.index - START, index.values, deg)
trend_table = pd.DataFrame(results, columns=["model", "MAE", "RMSE"])
print(trend_table.round(3))
trend_table.to_csv(OUT / "trend_errors.csv", index=False)

future = np.arange(START, HORIZON + 1)
best = trend_table.sort_values("RMSE").iloc[0]["model"]
curve = np.polyval(fits[best], future - START)
peak_year_trend = int(future[np.argmax(curve)])
print(f"Best trend model: {best}; peak of fitted curve within {HORIZON}: {peak_year_trend}")

# %% 6. Scenarios: optimistic / baseline / pessimistic -------------------
# Idea: extend each indicator with its recent linear trend (last 20 years),
# change the slope of the CLIMATE indicators per scenario, then pass the
# future indicators through the same scaler + PCA to get a future index.
climate_cols = [c for c in features if c in DOMAINS["climate"]]
NO_NEW_SHOCK = ["covid_log_per_million"]   # assumption: no new pandemic in the future
SCENARIOS = {"optimistic": 0.5, "baseline": 1.0, "pessimistic": 1.5}   # slope multipliers
recent = df[df.index >= END - 19]

future_years = np.arange(END + 1, HORIZON + 1)
all_years = np.concatenate([df.index.values, future_years])
scenario_index = {}
peaks = []
for sname, mult in SCENARIOS.items():
    fut = pd.DataFrame(index=future_years, columns=features, dtype=float)
    for c in features:
        if c in NO_NEW_SHOCK:
            fut[c] = 0.0
            continue
        slope, intercept = np.polyfit(recent.index, recent[c].values, 1)
        if c in climate_cols:
            slope *= mult
        last = recent[c].iloc[-1]
        fut[c] = last + slope * (future_years - END)
        if c not in climate_cols:                    # keep non-climate values realistic
            fut[c] = fut[c].clip(lower=df[c].min())
    full = pd.concat([df[features], fut])
    s = raw_index(full)
    s = (s - raw.min()) / (raw.max() - raw.min()) * 100     # same scaling as history
    s = pd.Series(s, index=full.index)
    scenario_index[sname] = s
    s_fut = s[s.index > END]                       # peak within the forecast period only
    peak_year = int(s_fut.idxmax())
    reached_horizon = peak_year == HORIZON
    peaks.append((sname, mult, peak_year, round(float(s_fut.max()), 1),
                  round(float(s.loc[HORIZON]), 1),
                  "no peak before horizon (still rising)" if reached_horizon else "peak"))

peak_table = pd.DataFrame(peaks, columns=["scenario", "climate_slope_x", "forecast_peak_year",
                                          "forecast_peak_value", "value_2100", "note"])
print(peak_table)
peak_table.to_csv(OUT / "scenario_peaks.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4.5))
for sname, s in scenario_index.items():
    ax.plot(s.index, s.values, label=sname)
ax.axvline(END, color="k", ls=":", alpha=0.5)
ax.set(title="Global Stress Index forecast by scenario", xlabel="Year", ylabel="Index")
ax.legend(); ax.grid(alpha=0.3)
fig.savefig(OUT / "scenarios.png", dpi=150, bbox_inches="tight")

# %% 7. Random Forest with lag features ---------------------------------
lag = df[features].shift(1).add_suffix("_lag1")
lag2 = df[features].shift(2).add_suffix("_lag2")
data = pd.concat([lag, lag2, index], axis=1).dropna()
Xf, yf = data.drop(columns="stress_index"), data["stress_index"]
train = data.index <= TRAIN_END

rf = RandomForestRegressor(n_estimators=300, random_state=0)
rf.fit(Xf[train], yf[train])
pred = rf.predict(Xf[~train])
rf_mae = mean_absolute_error(yf[~train], pred)
rf_rmse = np.sqrt(mean_squared_error(yf[~train], pred))
print(f"Random Forest  MAE={rf_mae:.3f}  RMSE={rf_rmse:.3f}")

imp = pd.Series(rf.feature_importances_, index=Xf.columns).sort_values(ascending=False)
print(imp.head(10).round(3))
imp.to_csv(OUT / "rf_importance.csv")

fig, ax = plt.subplots(figsize=(8, 5))
imp.head(10)[::-1].plot.barh(ax=ax)
ax.set(title="Random Forest: top 10 important indicators")
fig.savefig(OUT / "rf_importance.png", dpi=150, bbox_inches="tight")

# Note for the report: the index is built from these same indicators, so
# the importance ranking is partly circular, and there are only ~65 rows.

# %% 8. Summary table for the report ------------------------------------
summary = pd.concat([
    trend_table.assign(kind="trend"),
    pd.DataFrame([("random_forest", rf_mae, rf_rmse)], columns=["model", "MAE", "RMSE"]).assign(kind="RF lag"),
])
print(summary.round(3))
summary.to_csv(OUT / "model_comparison.csv", index=False)
print("Done. Check the outputs/ folder.")