"""
Runs every step in the right order. Stops at the first error so you can fix it.
Run from the project folder:   python run_all.py
"""
import subprocess
import sys
from pathlib import Path

STEPS = [
    "clean_data_gistemp.py",     # 1. temperature
    "clean_co2_and_merge.py",    # 2. Mauna Loa CO2 + merge
    "clean_owid_and_merge.py",   # 3. OWID gases + merge -> climate_full_cleaned.csv
    "clean_demography.py",       # 4. birth / death / life expectancy
    "clean_disasters.py",        # 5. disasters (once we write it)
    "end_of_the_world.py",       # 6. index, clusters, forecast, random forest
]

for step in STEPS:
    if not Path(step).exists():
        print(f"\n--- SKIP {step} (file not found) ---")
        continue
    print(f"\n=== RUNNING {step} ===")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"\nSTOPPED: {step} failed. Fix it, then run again.")
        sys.exit(1)

print("\nAll steps finished. Check data/clean/ and outputs/.")
