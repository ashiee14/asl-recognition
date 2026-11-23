"""
migrate_data.py

ONE-TIME USE. Your existing data/data.csv was recorded using raw
(non-normalized) landmark coordinates. landmarks.py now normalizes
coordinates (relative to wrist, scaled by hand size) to make the model
better at telling apart similar handshapes like M/N/S/T and I/Y.

This script reads your EXISTING data/data.csv and writes a normalized
version to data/data_normalized.csv, so you don't have to re-record
every letter from scratch. Your original data.csv is left untouched.

HOW TO USE:
  Run this once:
    python migrate_data.py

  Then train.py (which reads data/data_normalized.csv) will pick it up
  automatically. Any NEW samples you record with collect_data.py from
  now on are normalized automatically at collection time, so future
  recordings go straight into data.csv already-normalized -- you'd only
  need to rerun this migration if you mix old raw recordings back in.
"""

import os
import pandas as pd
from landmarks import normalize_landmarks

DATA_FILE = "data/data.csv"
OUT_FILE = "data/data_normalized.csv"

if not os.path.exists(DATA_FILE):
    raise SystemExit(f"{DATA_FILE} not found -- nothing to migrate.")

df = pd.read_csv(DATA_FILE, header=None)

rows = []
for _, row in df.iterrows():
    coords = row.iloc[:-1].tolist()          # first 63 columns
    label = row.iloc[-1]                      # last column = letter
    points = [(coords[i], coords[i + 1], coords[i + 2])
              for i in range(0, len(coords), 3)]
    normalized = normalize_landmarks(points)
    rows.append(normalized + [label])

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_FILE, header=False, index=False)

print(f"Migrated {len(out_df)} rows to normalized coordinates in {OUT_FILE}")
print("Your original data/data.csv was left untouched.")
print("Now run: python train.py")
