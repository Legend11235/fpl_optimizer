import pandas as pd

seasons = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
all_dfs = []

for season in seasons:
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
    print(f"loading {season}...")
    try:
        df = pd.read_csv(url, encoding='ISO-8859-1', on_bad_lines='skip')  # <-- key change here
        df['season'] = season
        all_dfs.append(df)
    except Exception as e:
        print(f"Error loading {season} data: {e}")

if all_dfs:
    print("Merging all data...")
    full_df = pd.concat(all_dfs, ignore_index=True)

    output_path = "data/all_seasons_merged.csv"
    full_df.to_csv(output_path, index=False)
    print(f"All data saved to: {output_path}")
else:
    print("No data loaded. Check your source or error logs.")

