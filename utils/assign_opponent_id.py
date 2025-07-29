import pandas as pd
import json

# === Load your main FPL data ===
df = pd.read_csv("data/2022-23_to_2024-25_clean.csv", low_memory=False)

# === Load your local master list ===
master = pd.read_csv("data/master_team_list.csv", dtype={"team": int})
# Expecting columns exactly: season,team,team_name

# Build a dict: (season,team) -> team_name
master_map = {
    (row.season, row.team): row.team_name
    for row in master.itertuples(index=False)
}

# Map directly into df
df["opponent_team_name"] = df.apply(
    lambda r: master_map.get((r["season"], r["opponent_team"])),
    axis=1
)

# Optional check: see which combos didn’t find a name
missing = (
    df[df["opponent_team_name"].isna()]
      [["season","opponent_team"]]
      .drop_duplicates()
)
if not missing.empty:
    print("Missing team names for these (season,team) pairs:\n", missing)

# === Load your global name->ID map ===
with open("../data/team_ids.json") as f:
    global_map = json.load(f)
if isinstance(global_map, list):
    global_map = {e["name"]: e["id"] for e in global_map}

# Now map names to IDs (will produce NaN for any unmapped names)
df["opponent_team_id"] = df["opponent_team_name"].map(global_map)

# Optional: check which names didn’t get an ID
bad_ids = (
    df[df["opponent_team_id"].isna()]
      ["opponent_team_name"]
      .drop_duplicates()
)
if not bad_ids.empty:
    print("These team names didn’t map to a global ID:\n", bad_ids)

# === Save the result ===
df.to_csv("data/2022-23_to_2024-25_clean_with_ids.csv", index=False)
print("Done — your CSV now has opponent_team_name and opponent_team_id columns.")