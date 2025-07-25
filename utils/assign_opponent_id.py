import pandas as pd
import json
import os

# === FILE PATHS ===
data_dir = "data"
csv_input_path = os.path.join(data_dir, "2022-23_to_2024-25_clean.csv")
csv_output_path = os.path.join(data_dir, "2022-23_to_2024-25_clean.csv")
team_id_file = os.path.join(data_dir, "team_ids.json")

# === LOAD CSV ===
df = pd.read_csv(csv_input_path, low_memory=False)
print(f"Loaded data from {csv_input_path} with {len(df)} rows.")

# === LOAD team_ids.json ===
if not os.path.exists(team_id_file):
    raise FileNotFoundError(f"Missing team ID mapping file: {team_id_file}")

with open(team_id_file, 'r') as f:
    team_to_id = json.load(f)

# === Map opponent team names to their team ID ===
if 'opponent_team' not in df.columns:
    raise KeyError("Column 'opponent_team' not found in DataFrame.")

df['opponent_team_id'] = df['opponent_team'].map(team_to_id)

# === Check for unmapped teams ===
unmapped = df[df['opponent_team_id'].isna()]['opponent_team'].unique()
if len(unmapped) > 0:
    print("Unmapped opponent teams found:")
    for team in unmapped:
        print("-", team)

# === SAVE updated file ===
df.to_csv(csv_output_path, index=False)
print(f"Saved updated DataFrame to {csv_output_path}")