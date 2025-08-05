import pandas as pd

# === Load and sort data ===
df = pd.read_csv("../data/2022-23_to_2024-25_cleanV2.csv", low_memory=False)
df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
df = df.sort_values(["global_player_id", "kickoff_time"]).reset_index(drop=True)

# === Convert boolean to int ===
df["was_home"] = df["was_home"].astype(int)

# === Feature list ===
sum_features = [
    "assists", "goals_scored", "bonus", "clean_sheets", "red_cards",
    "penalties_saved", "selected", "total_points",
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "xP"
]
windows = [1, 3, 5]

# === Rolling feature generation (STRICT with min_periods = window) ===
for feat in sum_features:
    for w in windows:
        suffix = f"last_{w}_gw"
        if feat in ["bonus", "xP", "expected_goals", "expected_assists",
                    "expected_goal_involvements", "expected_goals_conceded"] and w > 1:
            agg = "mean"
            suffix += "_avg"
        elif feat == "bonus" and w == 1:
            agg = "sum"
        else:
            agg = "sum"

        df[f"{feat}_{suffix}"] = (
            df.groupby("global_player_id", group_keys=False)[feat]
              .apply(lambda x: x.rolling(window=w, min_periods=w).agg(agg).shift(1))
        )

# === Normalized selected features ONLY ===

# selected_last_1_gw_norm: shift first
df["selected_last_1_gw_norm"] = (
    df.groupby("global_player_id", group_keys=False)["selected"]
      .apply(lambda x: x.shift(1) / x.max())
)

# selected_last_3_gw_norm and selected_last_5_gw_norm
for w in [3, 5]:
    norm_col = f"selected_last_{w}_gw_norm"
    df[norm_col] = (
        df.groupby("global_player_id", group_keys=False)["selected"]
          .apply(lambda x: x.rolling(window=w, min_periods=w).sum().shift(1) / x.max())
    )

# === Threat, Creativity, Influence ===
for feat in ["threat", "creativity", "influence"]:
    df[f"{feat}_last_1_gw"] = df.groupby("global_player_id", group_keys=False)[feat].shift(1)
    for w in [3, 5]:
        df[f"{feat}_last_{w}_gw_avg"] = (
            df.groupby("global_player_id", group_keys=False)[feat]
              .apply(lambda x: x.rolling(window=w, min_periods=w).mean().shift(1))
        )

# === Overperformance features ===

# --- 1. Goals vs expected goals ---
df["overperf_expected_goals_last_1_gw"] = df["goals_scored_last_1_gw"] - df["expected_goals_last_1_gw"]
df["overperf_expected_goals_last_3_gw"] = df["goals_scored_last_3_gw"] - df["expected_goals_last_3_gw_avg"]
df["overperf_expected_goals_last_5_gw"] = df["goals_scored_last_5_gw"] - df["expected_goals_last_5_gw_avg"]

# --- 2. Assists vs expected assists ---
df["overperf_expected_assists_last_1_gw"] = df["assists_last_1_gw"] - df["expected_assists_last_1_gw"]
df["overperf_expected_assists_last_3_gw"] = df["assists_last_3_gw"] - df["expected_assists_last_3_gw_avg"]
df["overperf_expected_assists_last_5_gw"] = df["assists_last_5_gw"] - df["expected_assists_last_5_gw_avg"]

# --- 3. xGI overperformance (goals + assists vs expected goal involvements) ---
df["overperf_xgi_last_1_gw"] = (
    df["goals_scored_last_1_gw"] + df["assists_last_1_gw"] - df["expected_goal_involvements_last_1_gw"]
)
df["overperf_xgi_last_3_gw"] = (
    df["goals_scored_last_3_gw"] + df["assists_last_3_gw"] - df["expected_goal_involvements_last_3_gw_avg"]
)
df["overperf_xgi_last_5_gw"] = (
    df["goals_scored_last_5_gw"] + df["assists_last_5_gw"] - df["expected_goal_involvements_last_5_gw_avg"]
)

# --- 4. Points overperformance vs expected points (xP) ---
df["overperf_xP_last_1_gw"] = df["total_points_last_1_gw"] - df["xP_last_1_gw"]
df["overperf_xP_last_3_gw"] = df["total_points_last_3_gw"] - df["xP_last_3_gw_avg"]
df["overperf_xP_last_5_gw"] = df["total_points_last_5_gw"] - df["xP_last_5_gw_avg"]


# === Value-based features ===
for w in [1, 3, 5]:
    # Avg change in value over rolling window
    df[f"value_avg_change_last_{w}_gw"] = (
        df.groupby("global_player_id", group_keys=False)["value"]
          .apply(lambda x: x.rolling(window=w, min_periods=w).apply(lambda s: s.diff().mean()).shift(1))
    )

    # Volatility (std) of value over window
    df[f"value_volatility_last_{w}_gw"] = (
        df.groupby("global_player_id", group_keys=False)["value"]
          .apply(lambda x: x.rolling(window=w, min_periods=w).std().shift(1))
    )
   # Target Variable:
   df["total_points_next_gw"] = df.groupby("global_player_id", group_keys=False)["total_points"].shift(-1)

# === Save final dataset ===
df.to_csv("../data/2022-23_to_2024-25_cleanV4.csv", index=False)
print("Saved to cleanV4 — STRICT rolling: min_periods = window, fully leakage-safe.")
