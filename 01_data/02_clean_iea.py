import pandas as pd
from pathlib import Path
import numpy as np

"""
Clean and standardize TRL2024 columns in the IEA Clean Tech Guide dataset.
Parses TRL values.
    - Converts '6' -> 6.0
    - Handles ranges like '6-7':
        - If range is within 1–5, takes the LOWER bound
        - If range is within 6–9, takes the UPPER bound
    - Returns NaN for invalid inputs
    - Caps final TRL values at 9.0
    - trl2024 columns with modified values are saved in a new column 'trl_final'
Saves the modified dataset to a new CSV file with '_with_trl_final'

"""

def parse_trl(val):
  
    if pd.isna(val):
        return np.nan
    
    s = str(val).strip()
    
    # Handle ranges
    if '-' in s:
        try:
            low_str, high_str = s.split('-', 1)
            low = float(low_str.strip())
            high = float(high_str.strip())
        except ValueError:
            return np.nan

        # If range is within 1–5, take lower bound; if within 6–9, take upper bound
        if low <= 5 and high <= 5:
            return low
        if low >= 6 and high >= 6:
            return high
        # Fallback: return upper bound if mixed range
        return high
    
    # Handle standard numbers
    try:
        return float(s)
    except ValueError:
        return np.nan

input_path = Path(r"C:\Users\Melusine\.venv\IEA_Clean_Tech_Guide (1).csv")
if not input_path.exists():
    raise FileNotFoundError(f"{input_path} not found")

df = pd.read_csv(input_path)

# find columns that match trl2024 (exact or starting with)
cols = [c for c in df.columns if c.lower().startswith("trl2024") or "trl2024" in c.lower()]

if not cols:
    raise ValueError("No column matching 'trl2024' found in the CSV")

# Apply the custom parser to each identified column
for col in cols:
    df[col] = df[col].apply(parse_trl)

# if multiple matching columns -> row-wise max, else copy the single column
if len(cols) > 1:
    df["trl_final"] = df[cols].max(axis=1, skipna=True)
else:
    df["trl_final"] = df[cols[0]]

# Cap values > 9 to 9 (instead of removing them)
initial_count_gt_9 = len(df[df["trl_final"] > 9])
df.loc[df["trl_final"] > 9, "trl_final"] = 9.0
print(f"Capped {initial_count_gt_9} rows where trl_final > 9 to 9")

# save to a new file (does not overwrite original)
output_path = input_path.with_name(input_path.stem + "_with_trl_final" + input_path.suffix)
df.to_csv(output_path, index=False)

print(f"Saved with new column 'trl_final' to: {output_path}")

print("\n--- Top 10 rows for trl2024 and trl_final ---")
# We use the 'cols' list which contains the actual column name(s) found in the CSV
print(df[cols + ['trl_final']].head(30))