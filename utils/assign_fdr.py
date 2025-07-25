import pandas as pd
import json

# Load your match-level FPL data
df = pd.read_csv("data/2022-23_to_2024-25_clean.csv")

# Load relative FDR JSON
with open("data/relative_fdr.json", "r") as f:
    relative_fdr_map = json.load(f)

# Define function to lookup FDR
def get_relative_fdr(row):
    team = row["team"]
    opponent = row["opponent_team_name"]
    side = "H" if row["was_home"] else "A"
    
    try:
        return relative_fdr_map[team][opponent][side]
    except KeyError:
        return None  # if data missing

# Apply to each row
df["relative_fdr"] = df.apply(get_relative_fdr, axis=1)

# Save updated version
df.to_csv("data/2022-23_to_2024-25_clean.csv", index=False)
print("Saved with relative FDRs.")