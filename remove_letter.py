
"""
remove_letter.py

Removes J and Z samples from the raw training dataset.
"""

import os
import pandas as pd

DATA_FILE = "data/data.csv"

if not os.path.exists(DATA_FILE):
    raise SystemExit(f"{DATA_FILE} not found.")

df = pd.read_csv(DATA_FILE, header=None)

before = len(df)

# Remove motion-based letters J and Z
df = df[~df.iloc[:, -1].isin(["J", "Z"])]

after = len(df)

df.to_csv(DATA_FILE, header=False, index=False)

print(f"Removed {before - after} J/Z samples.")
print(f"Remaining samples: {after}")
print(f"Updated file: {DATA_FILE}")
