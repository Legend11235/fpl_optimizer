import pandas as pd

seasons = ["2022-23", "2023-24", "2024-25"]
all_dfs = []

for season in seasons:
    print(f"Loading season: {season}")
    for gw in range(1, 39):  # GWs 1 through 38
        gw_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/gw{gw}.csv"
        try:
            df = pd.read_csv(gw_url, encoding='ISO-8859-1', on_bad_lines='skip')
            df["GW"] = gw
            df["season"] = season
            all_dfs.append(df)
            print(f"  Loaded GW{gw}")
        except Exception as e:
            print(f"  Failed to load GW{gw} of {season}: {e}")

# Merge and save
if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv("data/2022-23_to_2024-25_full.csv", index=False)
    print("✅ All gameweeks successfully merged and saved.")
else:
    print("❌ No data loaded.")