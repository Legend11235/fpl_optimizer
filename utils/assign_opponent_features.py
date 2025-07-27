import pandas as pd
import json

# === Load Data ===
df = pd.read_csv("data/2022-23_to_2024-25_clean.csv", low_memory=False)
with open("data/relative_fdr.json", "r") as f:
    relative_fdr_map = json.load(f)

# === Define Helper Functions ===
def get_relative_fdr(team, opponent, was_home):
    side = "H" if was_home else "A"
    try:
        return relative_fdr_map[team][opponent][side]
    except:
        return None

# === Next Opponent Features ===
df = df.sort_values(by=["global_player_id", "round"]).reset_index(drop=True)
df["next_opponent_team"] = df.groupby("global_player_id")["opponent_team_name"].shift(-1)
df["next_opponent_fdr"] = df.groupby("global_player_id")["relative_fdr"].shift(-1)

# Define stats for team-based aggregation
team_stats = [
    ("clean_sheets", "next_opponent_clean_sheets"),
    ("goals_conceded", "next_opponent_goals_conceded"),
    ("goals_scored", "next_opponent_goals_scored"),
]

# Calculate last N gameweeks stats per team
def get_team_stat_last_n(df, team_col, stat_col, window):
    result = []
    for idx, row in df.iterrows():
        gw = row["round"]
        team = row[team_col]
        hist = df[(df["team"] == team) & (df["round"] < gw)].sort_values("round", ascending=False)
        values = hist[stat_col].head(window).tolist()
        values = [v for v in values if pd.notnull(v)]
        result.append(sum(values) / len(values) if values else None)
    return result

# For each stat and window, compute for next opponent
df["next_opponent_team"] = df.groupby("global_player_id")["opponent_team_name"].shift(-1)
for stat, prefix in team_stats:
    for n in [1, 3, 5]:
        col_name = f"{prefix}_last_{n}_gw"
        df[col_name] = get_team_stat_last_n(df, "next_opponent_team", stat, n)

# === Past Opponent Features ===
def get_avg_past_stat(df, pid_col, round_col, team_col, stat_col, window):
    result = []
    for idx, row in df.iterrows():
        pid = row[pid_col]
        gw = row[round_col]
        past_rows = df[(df[pid_col] == pid) & (df[round_col] < gw)].sort_values(round_col, ascending=False)
        teams = past_rows[team_col].head(window)
        values = []
        for team in teams:
            team_stats = df[(df["team"] == team) & (df["round"] < gw)][stat_col].tail(1).values
            if len(team_stats) > 0:
                values.append(team_stats[0])
        values = [v for v in values if pd.notnull(v)]
        result.append(sum(values) / len(values) if values else None)
    return result

# Relative FDR rolling stats
df["avg_opponent_fdr_last_1_gw"] = df.groupby("global_player_id")["relative_fdr"].shift(1)
df["avg_opponent_fdr_last_3_gw"] = df.groupby("global_player_id")["relative_fdr"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
df["avg_opponent_fdr_last_5_gw"] = df.groupby("global_player_id")["relative_fdr"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

# Defensive/attacking past stats from opponent team
for stat, _ in team_stats:
    for n in [1, 3, 5]:
        col_name = f"avg_opponent_{stat}_last_{n}"
        df[col_name] = get_avg_past_stat(df, "global_player_id", "round", "opponent_team_name", stat, n)

# === Save updated DataFrame ===
df.to_csv("data/2022-23_to_2024-25_cleanTEST.csv", index=False)
print("All opponent-related FDR and team stats computed and saved.")