import pandas as pd
import json
import os
import requests

# Define data directory relative to current notebook
data_dir = "data"
team_id_file = os.path.join(data_dir, "team_ids.json")
csv_input_path = os.path.join(data_dir, "2022-23_to_2024-25_clean.csv")
csv_output_path = os.path.join(data_dir, "2022-23_to_2024-25_clean.csv")

# Ensure the directory exists
os.makedirs(data_dir, exist_ok=True)

# Load your main DataFrame
df = pd.read_csv(csv_input_path, low_memory=False)

# Step 1: Collect all known teams from your dataset
dataset_teams = set(df['team'].dropna().unique())

# Step 2: Fetch current teams from FPL API
try:
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    api_data = requests.get(url).json()
    api_teams = [t['name'] for t in api_data['teams']]
    print(f"Found {len(api_teams)} teams from FPL API.")
except Exception as e:
    print("Failed to fetch FPL API teams:", e)
    api_teams = []

# Step 3: Combine both
all_teams = sorted(set(dataset_teams).union(api_teams))

# Step 4: Load existing team_id mapping or create new one
if os.path.exists(team_id_file):
    with open(team_id_file, 'r') as f:
        team_to_id = json.load(f)
    print("Loaded existing team ID mapping.")
else:
    team_to_id = {}
    print("Creating new team ID mapping.")

# Step 5: Assign IDs to unseen teams
current_max_id = max(team_to_id.values(), default=-1)
for team in all_teams:
    if team not in team_to_id:
        current_max_id += 1
        team_to_id[team] = current_max_id
        print(f"Assigned team '{team}' ID {current_max_id}")

# Step 6: Save the updated mapping
with open(team_id_file, 'w') as f:
    json.dump(team_to_id, f)
    print(f"Saved team ID mapping to {team_id_file}")

# Step 7: Apply to your DataFrame
df['team_id'] = df['team'].map(team_to_id)

# Save updated df
df.to_csv(csv_output_path, index=False)
print(f"Saved updated DataFrame to {csv_output_path}")