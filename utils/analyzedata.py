import pandas as pd

# Load the full dataset
df = pd.read_csv("data/all_seasons_merged.csv")

# Filter for Ruben Dias in 2024–25 season
player_name = "Rúben Dias"  # or "Ruben Dias" depending on name formatting
season = "2024-25"

filtered_df = df[(df["name"].str.lower().str.contains("alves dias")) & (df["season"] == season)]

# Save to CSV
output_path = "data/ruben_dias_2024_25.csv"
filtered_df.to_csv(output_path, index=False)

# Print summary
print(f"✅ Found {len(filtered_df)} records for Rúben Dias in {season}")
print(f"📁 Saved to: {output_path}")
