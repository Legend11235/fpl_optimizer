import pandas as pd
import json

# === 0) Load & prep ===
df = pd.read_csv("../data/2022-23_to_2024-25_clean.csv", low_memory=False)
with open("../data/relative_fdr_V2.json", "r") as f:
    relative_fdr_map = json.load(f)

# parse dates & sort
df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
df = df.sort_values(["global_player_id", "kickoff_time"]).reset_index(drop=True)

# === 1) Next-opponent info ===
df["next_kickoff_time"]  = df.groupby("global_player_id")["kickoff_time"].shift(-1)
df["next_opponent_team"] = df.groupby("global_player_id")["opponent_team_name"].shift(-1)
df["next_opponent_fdr"]  = df.groupby("global_player_id")["relative_fdr"].shift(-1)

# === 2) Per-row opponent stats ===
def get_opponent_stats(r):
    if r.was_home:
        sc, conc = r.team_a_score, r.team_h_score
    else:
        sc, conc = r.team_h_score, r.team_a_score
    return pd.Series({
        "opp_clean_sheets":   1 if conc == 0 else 0,
        "opp_goals_conceded": conc,
        "opp_goals_scored":   sc
    })

df[["opp_clean_sheets",
    "opp_goals_conceded",
    "opp_goals_scored"]] = df.apply(get_opponent_stats, axis=1)

# === 3) avg_opponent_fdr_last_N_gw (unchanged) ===
for n in (1, 3, 5):
    df[f"avg_opponent_fdr_last_{n}_gw"] = (
        df.groupby("global_player_id")["relative_fdr"]
          .shift(1)
          .rolling(window=n, min_periods=n)
          .mean()
          .reset_index(level=0, drop=True)
    )

# === 4) avg_opponent_form_last_N (unchanged) ===
for stat in ("opp_clean_sheets","opp_goals_conceded","opp_goals_scored"):
    for n in (1, 3, 5):
        df[f"avg_opponent_{stat}_last_{n}"] = (
            df.groupby("global_player_id")[stat]
              .shift(1)
              .rolling(window=n, min_periods=n)
              .mean()
              .reset_index(level=0, drop=True)
        )

# === 5) Build a true fixture-level table ===
fixtures = (
    df[[
      "season",
      "opponent_team_name",
      "kickoff_time",
      "opp_clean_sheets",
      "opp_goals_conceded",
      "opp_goals_scored"
    ]]
    .rename(columns={"opponent_team_name": "team"})
    .drop_duplicates(subset=["season","team","kickoff_time"])
    .sort_values(["team","kickoff_time"])
    .reset_index(drop=True)
)

# === 6) Compute rolling stats on that table (cross-season carryover) ===
for stat in ("opp_clean_sheets","opp_goals_conceded","opp_goals_scored"):
    short = stat.replace("opp_","")  # e.g. "clean_sheets"
    grp = fixtures.groupby("team")[stat]
    for n in (1, 3, 5):
        fixtures[f"next_opponent_{short}_last_{n}_gw"] = (
            grp
              .rolling(window=n, min_periods=n)
              .mean()
              .shift(1)
              .reset_index(level=0, drop=True)
        )

# === 7) Merge those next-opponent metrics back in ===
# prepare for join
merge_cols = ["team","kickoff_time"] + [
    c for c in fixtures.columns
    if c.startswith("next_opponent_") and c.endswith("_gw")
]
# rename to match df’s “next_…” columns
fixtures_merge = fixtures[merge_cols].rename(
    columns={"team":"next_opponent_team",
             "kickoff_time":"next_kickoff_time"}
)

df = df.merge(
    fixtures_merge,
    how="left",
    on=["next_opponent_team","next_kickoff_time"]
)

# === 8) Final cleanup & save ===
df = df.sort_values(["global_player_id", "kickoff_time"]).reset_index(drop=True)
df.to_csv("../data/2022-23_to_2024-25_cleanV2.csv", index=False)

print("Succesfuly, added the features")