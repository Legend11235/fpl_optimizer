import pandas as pd

# Load your full merged data
df = pd.read_csv("data/2022-23_to_2024-25_full.csv", low_memory=False)

# Create a DataFrame with unique (season, element) combos
player_keys = df[['season', 'element', 'name']].drop_duplicates()

# We'll assign global IDs based on fuzzy-matching names across seasons
from collections import defaultdict
from fuzzywuzzy import fuzz

global_id = 1
global_id_map = {}
name_to_id = {}

# Process season by season
seasons = sorted(player_keys['season'].unique())
for season in seasons:
    season_players = player_keys[player_keys['season'] == season]
    for _, row in season_players.iterrows():
        elem = row['element']
        name = row['name']
        matched_id = None
        
        for existing_name in name_to_id:
            score = fuzz.token_sort_ratio(name, existing_name)
            if score > 90:
                matched_id = name_to_id[existing_name]
                break
        
        if matched_id is None:
            matched_id = global_id
            global_id += 1
        
        global_id_map[(season, elem)] = matched_id
        name_to_id[name] = matched_id

#Add Global ID to data:
# Map the global IDs back to your main DataFrame
df['global_player_id'] = df.apply(lambda row: global_id_map.get((row['season'], row['element']), -1), axis=1)

# Optional sanity check
print(df[['name', 'season', 'element', 'global_player_id']].drop_duplicates().sort_values(by='global_player_id').head(20))

# Save updated file
df.to_csv("data/2022-23_to_2024-25_clean.csv", index=False)