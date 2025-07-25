import pandas as pd

# Load your DataFrame
df = pd.read_csv("data/2022-23_to_2024-25_clean.csv", low_memory=False)

# Define valid FPL positions
valid_positions = ['GKP', 'DEF', 'MID', 'FWD']

# Filter out invalid positions (e.g., AM, PEP, etc.)
df = df[df['position'].isin(valid_positions)].copy()

# One-hot encode the 'position' column
position_dummies = pd.get_dummies(df['position'], prefix='pos', dtype=int)

# Join back to main DataFrame
df = pd.concat([df, position_dummies], axis=1)

# Optional: sanity check
print(df[['position'] + list(position_dummies.columns)].drop_duplicates())

# Save updated DataFrame
df.to_csv("../data/2022-23_to_2024-25_clean.csv", index=False)