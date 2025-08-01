import pandas as pd
import json

# === Load Data ===
df = pd.read_csv("data/2022-23_to_2024-25_clean.csv", low_memory=False)
with open("data/relative_fdr.json", "r") as f:
    relative_fdr_map = json.load(f)

# === Step 0: Parse kickoff_time and sort by player + date ===
df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
df = df.sort_values(by=["global_player_id", "kickoff_time"]).reset_index(drop=True)

# === Step 1: Add next_opponent_team and next_opponent_fdr ===
df["next_opponent_team"] = df.groupby("global_player_id")["opponent_team_name"].shift(-1)
df["next_opponent_fdr"] = df.groupby("global_player_id")["relative_fdr"].shift(-1)

# === Step 2: Compute stats for opponent team (not player team!) ===
def get_opponent_stats(row):
    if row["was_home"]:
        return pd.Series({
            "opp_goals_scored": row["team_a_score"],
            "opp_goals_conceded": row["team_h_score"],
            "opp_clean_sheets": 1 if row["team_h_score"] == 0 else 0
        })
    else:
        return pd.Series({
            "opp_goals_scored": row["team_h_score"],
            "opp_goals_conceded": row["team_a_score"],
            "opp_clean_sheets": 1 if row["team_a_score"] == 0 else 0
        })

df[["opp_goals_scored", "opp_goals_conceded", "opp_clean_sheets"]] = df.apply(get_opponent_stats, axis=1)

# === Step 3: Average FDR of previous opponents over 1/3/5 GWs ===
for n in [1, 3, 5]:
    df[f"avg_opponent_fdr_last_{n}_gw"] = (
        df.groupby("global_player_id")["relative_fdr"]
        .shift(1)
        .rolling(window=n, min_periods=n)  # full window required
        .mean()
        .reset_index(level=0, drop=True)
    )

# === Step 4: Average form of past opponents over 1/3/5 GWs ===
for stat in ["opp_clean_sheets", "opp_goals_conceded", "opp_goals_scored"]:
    for n in [1, 3, 5]:
        df[f"avg_opponent_{stat}_last_{n}"] = (
            df.groupby("global_player_id")[stat]
            .shift(1)
            .rolling(window=n, min_periods=n)
            .mean()
            .reset_index(level=0, drop=True)
        )

# === Step 5: Rolling form of NEXT opponent team (e.g. Burnley's recent stats) ===

# Temporarily use next_opponent_team in place of `team` but save the original
df["team_original"] = df["team"]
df["team"] = df["next_opponent_team"]

def compute_next_opp_form(df, stat, prefix):
    results = []
    for team, group in df.groupby("team"):
        group = group.sort_values("kickoff_time")
        for n in [1, 3, 5]:
            group[f"{prefix}_last_{n}_gw"] = (
                group[stat]
                .rolling(window=n, min_periods=n)
                .mean()
                .shift(1)
            )
        results.append(group)
    return pd.concat(results)

df = compute_next_opp_form(df, "opp_clean_sheets", "next_opponent_clean_sheets")
df = compute_next_opp_form(df, "opp_goals_conceded", "next_opponent_goals_conceded")
df = compute_next_opp_form(df, "opp_goals_scored", "next_opponent_goals_scored")

# === Finalize: Restore team column and sort ===
df["team"] = df["team_original"]
df.drop(columns=["team_original"], inplace=True)
df = df.sort_values(by=["global_player_id", "kickoff_time"]).reset_index(drop=True)

# === Save ===
df.to_csv("data/2022-23_to_2024-25_cleanV2.csv", index=False)
print("All features added successfully and saved to cleanV2.")