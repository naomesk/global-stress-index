# End of the World: Global Stress Index and Peak Forecasting

Introduction to Machine Learning, Fall 2026, Woosong University.

We combine climate, mortality, disaster and pandemic data (1959-2024) into one yearly
Global Stress Index, group the years into calm / stressed / crisis periods, and project the
index to 2100 under optimistic, baseline and pessimistic scenarios.

**This is a data-driven hypothesis, not a prediction of a real "end of the world".**

## Folder layout

```
data/raw/      downloaded source files (not all are stored in this repo, see below)
data/clean/    cleaned yearly tables made by the cleaning scripts
outputs/       figures and result tables made by end_of_the_world.py
clean_data_gistemp.py, clean_co2_and_merge.py, clean_owid_and_merge.py   climate cleaning
clean_demography.py     birth rate, death rate, life expectancy (World Bank)
clean_disasters.py      natural-disaster deaths (EM-DAT via Our World in Data)
end_of_the_world.py     index, K-means, label spreading, trend fits, scenarios, Random Forest
run_all.py              runs everything in order
```

## Data sources

| Domain | Source | File in `data/raw/` |
|---|---|---|
| Temperature | NASA GISTEMP | `GLB.Ts+dSST.csv` |
| Atmospheric CO2 | NOAA Mauna Loa | `co2_annmean_mlo.csv` |
| Emissions, methane, N2O, population | Our World in Data CO2 data | `owid-co2-data.csv` |
| Birth rate, death rate, life expectancy | World Bank Open Data (SP.DYN.CBRT.IN, SP.DYN.CDRT.IN, SP.DYN.LE00.IN) | `API_SP.DYN.*.csv` |
| Disaster deaths | EM-DAT (CRED / UCLouvain) via Our World in Data | `annual-deaths-from-all-natural-disasters.csv` |
| Pandemic | Our World in Data COVID-19 data | `owid-covid-data.csv` (over 100 MB, **not in this repo**: download from github.com/owid/covid-19-data) |

## How to run

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
python run_all.py
```

Or run the scripts one by one in this order: climate scripts, `clean_demography.py`,
`clean_disasters.py`, then `end_of_the_world.py` (cell by cell in VS Code).

## Method in short

1. Align all sources to one yearly axis (1959-2024), scale and standardize them.
2. **PCA** compresses the five redundant climate indicators into one climate score.
3. The stress index is the equal-weight average of four domain scores: climate, mortality,
   disasters, pandemic (higher always means more stress).
4. **K-means** (k = 3) groups years into calm, stressed and crisis.
5. **Semi-supervised label spreading** starts from the 20% most typical years of each period
   and labels the rest, as a check on the K-means periods.
6. **Linear and polynomial regression** and a **Random Forest with lag features**, evaluated with a
   time-based train/test split using MAE and RMSE, compared with naive baselines (see below).
7. **Scenarios**: climate trends are slowed (x0.5) or accelerated (x1.5) and the index is projected to 2100.

## Train / test split

| Set | Years | Count | Share |
|---|---|---|---|
| Training | 1959-2015 | 57 | about 86% |
| Test | 2016-2024 | 9 | about 14% |

The split is by time and the data is **not shuffled**. This is a time series, and a random split would let
the model see years on both sides of each test year, which makes the errors look better than they are.
The Random Forest uses the previous 1-2 years as inputs, so it has about 55 usable training rows.
The label spreading check uses its own split: 14 known years (about 21%) and 52 hidden years (about 79%).

## Main results

- The most stressed observed year is 2021 (COVID-19).
- No peak before 2100 in any scenario: the index is still rising in 2100 (about 52, 87 and 122 for
  the optimistic, baseline and pessimistic scenarios).
- First year the index reaches the historical crisis level: none by 2100 (optimistic), 2083 (baseline), 2058 (pessimistic).
- Label spreading with 14 known years recovers 88% of the labels of the other 52 years.
- The forecasting models do not clearly beat naive guesses on the 2016-2024 test years, which contain the pandemic.

## Limitations

- Disaster reporting is less complete in the early decades.
- COVID-19 is one pandemic event.
- The index is built from the same indicators the Random Forest uses, so the importance ranking is partly circular.
- The scenarios assume no new pandemic and extend recent linear trends.
- World Bank data starts in 1960, so 1959 repeats 1960.

## Team

[Add team member names here]
- Nur Shevval Rasim Kyzy
- Naomi Eskinder Woudneh
- Noé Challa
