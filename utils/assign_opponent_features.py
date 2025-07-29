import pandas as pd
import json

# === Load Data ===
df = pd.read_csv("data/2022-23_to_2024-25_clean.csv", low_memory=False, parse_dates=["kickoff_time"])
with open("data/relative_fdr.json", "r") as f:
    relative_fdr_map = json.load(f)

# === Helper ===
def get_relative_fdr(team, opponent, was_home):
    side = "H" if was_home else "A"
    try:
        return relative_fdr_map[team][opponent][side]
    except:
        return None

# === Preprocessing ===
df = df.sort_values(by=["global_player_id", "kickoff_time"]).reset_index(drop=True)

# === Next Opponent Features ===
df["next_opponent_team"] = df.groupby("global_player_id")["opponent_team_name"].shift(-1)
df["next_opponent_fdr"] = df.groupby("global_player_id")["relative_fdr"].shift(-1)

# --- Vectorized next opponent team stats ---
def compute_team_form(df, stat_col, prefix):
    for n in [1, 3, 5]:
        result_col = f"{prefix}_last_{n}_gw"
        stat_values = []

        for team, group in df.groupby("team"):
            sorted_group = group.sort_values("kickoff_time")
            rolling = sorted_group[stat_col].rolling(window=n, min_periods=1).mean().shift(1)
            temp = sorted_group[["kickoff_time"]].copy()
            temp[result_col] = rolling.values
            temp["team"] = team
            stat_values.append(temp)

        form_df = pd.concat(stat_values)
        merged = df.merge(form_df, on=["team", "kickoff_time"], how="left")
        df[result_col] = merged[result_col]

    return df

# For each stat from opponent team
temp_team_col = df["team"].copy()  # save original
for stat, prefix in [
    ("clean_sheets", "next_opponent_clean_sheets"),
    ("goals_conceded", "next_opponent_goals_conceded"),
    ("goals_scored", "next_opponent_goals_scored"),
]:
    df["team"] = df["next_opponent_team"]
    df = compute_team_form(df, stat, prefix)
df["team"] = temp_team_col  # restore original

# === Past Opponent Features ===
df = df.sort_values(by=["global_player_id", "kickoff_time"]).reset_index(drop=True)

# Vectorized FDR-based rolling stats
df["avg_opponent_fdr_last_1_gw"] = df.groupby("global_player_id")["relative_fdr"].shift(1)
df["avg_opponent_fdr_last_3_gw"] = (
    df.groupby("global_player_id")["relative_fdr"].shift(1)
    .rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
)
df["avg_opponent_fdr_last_5_gw"] = (
    df.groupby("global_player_id")["relative_fdr"].shift(1)
    .rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
)

# Past form for team-based stats
for stat in ["clean_sheets", "goals_conceded", "goals_scored"]:
    for n in [1, 3, 5]:
        colname = f"avg_opponent_{stat}_last_{n}"
        df[colname] = (
            df.groupby("global_player_id")
            .apply(lambda group: group.shift(1).rolling(n, min_periods=1)[stat].mean())
            .reset_index(level=0, drop=True)
        )

# === Save Updated ===
df.to_csv("data/2022-23_to_2024-25_cleanTest.csv", index=False)
print("✅ Done: All opponent-based features added and saved with kickoff_time ordering.")