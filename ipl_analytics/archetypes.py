"""
archetypes.py
==============
Classifies every player-season into one of four archetypes:
    Powerplay Aggressor | Anchor | Death Specialist | Middle-Over Squeezer

DESIGN DECISION (documented explicitly because it isn't fully spelled out in
the brief): "Anchor" and "Powerplay Aggressor" are inherently BATTING
concepts, while "Middle-Over Squeezer" is inherently a BOWLING concept
(an economical bowler who strangles the middle overs). "Death Specialist"
can describe either a finisher (batting) or a death-overs bowler. Forcing
every player into one undifferentiated pool would compare apples to oranges
(e.g. comparing a bowler's economy Z-score against a batter's strike-rate
Z-score is meaningless). So:

    1. Each player-season is first assigned a PRIMARY DISCIPLINE (batter or
       bowler) based on which role they had more deliveries in.
    2. Batters are scored only against the batting archetypes:
           {Powerplay Aggressor, Anchor, Death Specialist}
    3. Bowlers are scored only against the bowling archetypes:
           {Middle-Over Squeezer, Death Specialist}
    4. Within each discipline, every metric is Z-SCORE NORMALIZED against
       that SEASON's league average for that discipline (not pooled across
       years) -- this is what makes the archetype comparable across seasons
       even as overall scoring rates shift (e.g. 2022's higher-scoring
       environment doesn't make every batter look like an "Aggressor").
    5. A player's archetype = whichever candidate Z-score is highest, among
       the archetypes valid for their discipline.

EVERY PLAYER-SEASON IS CLASSIFIED -- NO "UNCLASSIFIED" BUCKET.
----------------------------------------------------------------
An earlier version of this module applied a hard minimum-balls
qualification threshold and dropped anyone below it, which downstream
modules then had to bucket into a catch-all "Unclassified" category. That
conflated two genuinely different things: "this player's role couldn't be
determined" and "this player only had a handful of opportunities" -- a
specialist death-overs bowler who only got 3 overs all year because of an
early injury is not "unclassified", he is a Death Specialist with a small
sample.

Instead: the Z-score MEAN and STANDARD DEVIATION used to normalize every
metric are computed from the STABLE population only (players who cleared
MIN_BALLS_FACED / MIN_BALLS_BOWLED that season), so a handful of extreme
small-sample outliers (e.g. a player who faced 2 balls and hit a six off
one of them, an instant 300 strike rate) cannot distort the league
baseline that everyone else is measured against. Every player-season --
stable or thin -- is then scored against that fixed, stable baseline and
assigned the archetype with the highest resulting score, same as before.
A boolean `is_low_sample` column flags anyone below the threshold, so
downstream consumers and the dashboard can show the classification with
an honest confidence caveat instead of hiding the player entirely.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

MIN_BALLS_FACED = 60     # batting "stable sample" threshold per season
MIN_BALLS_BOWLED = 60    # bowling "stable sample" threshold per season (~10 overs)

BATTING_ARCHETYPES = ["Powerplay Aggressor", "Anchor", "Death Specialist"]
BOWLING_ARCHETYPES = ["Middle-Over Squeezer", "Death Specialist"]


# ---------------------------------------------------------------------------
# Player-season batting stats
# ---------------------------------------------------------------------------
def compute_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
    bat = df.copy()
    bat["balls_faced"] = bat["ball_faced_by_batter"].astype(int)  # only wides don't count as a ball faced (no-balls do)

    grp = bat.groupby(["season", "batter", "phase"]).agg(
        runs=("runs_batter", "sum"),
        balls=("balls_faced", "sum"),
        dismissals=("is_wicket", "sum"),  # any wicket on this delivery while this player was on strike
    ).reset_index()

    # only count dismissals of THIS batter (not run-outs of the non-striker etc.)
    dismissed = df[df["is_wicket"] & (df["wicket_player_out"] == df["batter"])]
    dismissals_by_phase = dismissed.groupby(["season", "batter", "phase"]).size().rename("dismissals").reset_index()
    grp = grp.drop(columns="dismissals").merge(dismissals_by_phase, on=["season", "batter", "phase"], how="left")
    grp["dismissals"] = grp["dismissals"].fillna(0)

    grp["strike_rate"] = np.where(grp["balls"] > 0, grp["runs"] / grp["balls"] * 100, np.nan)

    pivot = grp.pivot_table(index=["season", "batter"], columns="phase",
                             values=["runs", "balls", "dismissals", "strike_rate"])
    pivot.columns = [f"{stat}_{phase}" for stat, phase in pivot.columns]
    pivot = pivot.reset_index().fillna(0)

    pivot["total_balls_faced"] = pivot[[c for c in pivot.columns if c.startswith("balls_")]].sum(axis=1)
    pivot["total_runs"] = pivot[[c for c in pivot.columns if c.startswith("runs_")]].sum(axis=1)
    pivot["overall_strike_rate"] = np.where(
        pivot["total_balls_faced"] > 0, pivot["total_runs"] / pivot["total_balls_faced"] * 100, np.nan
    )
    pivot["total_dismissals"] = pivot[[c for c in pivot.columns if c.startswith("dismissals_")]].sum(axis=1)
    pivot["dismissal_rate"] = np.where(
        pivot["total_balls_faced"] > 0, pivot["total_dismissals"] / pivot["total_balls_faced"], np.nan
    )
    for phase in ["Powerplay", "Middle", "Death"]:
        pivot[f"balls_share_{phase}"] = np.where(
            pivot["total_balls_faced"] > 0,
            pivot.get(f"balls_{phase}", 0) / pivot["total_balls_faced"],
            0,
        )

    pivot = pivot.rename(columns={"batter": "player"})
    pivot["is_low_sample"] = pivot["total_balls_faced"] < MIN_BALLS_FACED
    return pivot.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player-season bowling stats
# ---------------------------------------------------------------------------
def compute_bowling_stats(df: pd.DataFrame) -> pd.DataFrame:
    bowl = df.copy()
    bowl["legal_ball_int"] = bowl["legal_ball"].astype(int)
    # bowler concedes all runs off the bat plus wides/no-balls (standard cricket economy definition)
    bowl["runs_conceded"] = bowl["runs_total"] - bowl["extras_byes"] - bowl["extras_legbyes"]
    # bowler credited with a wicket except run-outs (standard convention)
    bowl["bowler_wicket"] = bowl["is_wicket"] & (~bowl["wicket_kind"].isin(["run out", "obstructing the field"]))

    grp = bowl.groupby(["season", "bowler", "phase"]).agg(
        runs_conceded=("runs_conceded", "sum"),
        balls=("legal_ball_int", "sum"),
        wickets=("bowler_wicket", "sum"),
    ).reset_index()
    grp["economy"] = np.where(grp["balls"] > 0, grp["runs_conceded"] / (grp["balls"] / 6), np.nan)

    pivot = grp.pivot_table(index=["season", "bowler"], columns="phase",
                             values=["runs_conceded", "balls", "wickets", "economy"])
    pivot.columns = [f"{stat}_{phase}" for stat, phase in pivot.columns]
    pivot = pivot.reset_index().fillna(0)

    pivot["total_balls_bowled"] = pivot[[c for c in pivot.columns if c.startswith("balls_")]].sum(axis=1)
    pivot["total_runs_conceded"] = pivot[[c for c in pivot.columns if c.startswith("runs_conceded_")]].sum(axis=1)
    pivot["overall_economy"] = np.where(
        pivot["total_balls_bowled"] > 0,
        pivot["total_runs_conceded"] / (pivot["total_balls_bowled"] / 6),
        np.nan,
    )
    for phase in ["Powerplay", "Middle", "Death"]:
        pivot[f"balls_share_{phase}"] = np.where(
            pivot["total_balls_bowled"] > 0,
            pivot.get(f"balls_{phase}", 0) / pivot["total_balls_bowled"],
            0,
        )

    pivot = pivot.rename(columns={"bowler": "player"})
    pivot["is_low_sample"] = pivot["total_balls_bowled"] < MIN_BALLS_BOWLED
    return pivot.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Z-score normalization helper
# ---------------------------------------------------------------------------
def zscore_within_season(frame: pd.DataFrame, col: str, stable_mask: pd.Series) -> pd.Series:
    """Standardizes `col` within each season, using the MEAN and STD of only
    the stable (non-low-sample) rows in that season as the reference, then
    applies that fixed reference to every row (stable or thin-sample) in
    the season. This is what lets thin-sample player-seasons still receive
    a real archetype rather than being dropped, without their own noisy,
    small-n stats distorting the league baseline everyone is compared
    against."""
    out = pd.Series(index=frame.index, dtype=float)
    for season, idx in frame.groupby("season").groups.items():
        season_rows = frame.loc[idx]
        stable_vals = season_rows.loc[stable_mask.loc[idx], col]
        mean = stable_vals.mean()
        std = stable_vals.std(ddof=0)
        if not std or np.isnan(std) or std == 0:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (season_rows[col] - mean) / std
    return out


# ---------------------------------------------------------------------------
# Discipline assignment + archetype classification
# ---------------------------------------------------------------------------
def classify_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    bat_stats = compute_batting_stats(df)
    bowl_stats = compute_bowling_stats(df)

    # --- BATTING Z-SCORES --------------------------------------------------
    bat_stable = ~bat_stats["is_low_sample"]
    bat_stats["z_pp_sr"] = zscore_within_season(bat_stats, "strike_rate_Powerplay", bat_stable)
    bat_stats["z_death_sr"] = zscore_within_season(bat_stats, "strike_rate_Death", bat_stable)
    bat_stats["z_middle_balls_share"] = zscore_within_season(bat_stats, "balls_share_Middle", bat_stable)
    bat_stats["z_dismissal_rate"] = zscore_within_season(bat_stats, "dismissal_rate", bat_stable)

    # Anchor = high crease occupation in the middle overs (high balls-share)
    # AND a low dismissal rate (rarely gets out) -- a composite score, since
    # "anchoring" is about both presence and control, not raw strike rate.
    bat_stats["score_powerplay_aggressor"] = bat_stats["z_pp_sr"]
    bat_stats["score_anchor"] = bat_stats["z_middle_balls_share"] - bat_stats["z_dismissal_rate"]
    bat_stats["score_death_specialist"] = bat_stats["z_death_sr"]

    score_cols_bat = {
        "Powerplay Aggressor": "score_powerplay_aggressor",
        "Anchor": "score_anchor",
        "Death Specialist": "score_death_specialist",
    }
    bat_stats["archetype"] = bat_stats[list(score_cols_bat.values())].idxmax(axis=1).map(
        {v: k for k, v in score_cols_bat.items()}
    )
    bat_stats["archetype_score"] = bat_stats[list(score_cols_bat.values())].max(axis=1)
    bat_stats["discipline"] = "Batter"

    # --- BOWLING Z-SCORES ----------------------------------------------------
    # Economy is "good" when LOW, so we negate the Z-score: a higher
    # score_* value should always mean "more strongly this archetype",
    # consistent with the batting side.
    bat_stats_cols_keep = [
        "season", "player", "discipline", "archetype", "archetype_score", "is_low_sample",
        "total_balls_faced", "overall_strike_rate", "dismissal_rate",
        "strike_rate_Powerplay", "strike_rate_Middle", "strike_rate_Death",
        "balls_share_Powerplay", "balls_share_Middle", "balls_share_Death",
    ]

    bowl_stable = ~bowl_stats["is_low_sample"]
    bowl_stats["z_middle_econ"] = zscore_within_season(bowl_stats, "economy_Middle", bowl_stable)
    bowl_stats["z_death_econ"] = zscore_within_season(bowl_stats, "economy_Death", bowl_stable)
    bowl_stats["score_middle_over_squeezer"] = -bowl_stats["z_middle_econ"]
    bowl_stats["score_death_specialist"] = -bowl_stats["z_death_econ"]

    score_cols_bowl = {
        "Middle-Over Squeezer": "score_middle_over_squeezer",
        "Death Specialist": "score_death_specialist",
    }
    bowl_stats["archetype"] = bowl_stats[list(score_cols_bowl.values())].idxmax(axis=1).map(
        {v: k for k, v in score_cols_bowl.items()}
    )
    bowl_stats["archetype_score"] = bowl_stats[list(score_cols_bowl.values())].max(axis=1)
    bowl_stats["discipline"] = "Bowler"

    bowl_stats_cols_keep = [
        "season", "player", "discipline", "archetype", "archetype_score", "is_low_sample",
        "total_balls_bowled", "overall_economy",
        "economy_Powerplay", "economy_Middle", "economy_Death",
        "balls_share_Powerplay", "balls_share_Middle", "balls_share_Death",
    ]

    batters_out = bat_stats[bat_stats_cols_keep].copy()
    bowlers_out = bowl_stats[bowl_stats_cols_keep].copy()

    # --- resolve players who qualify in BOTH disciplines (genuine all-rounders) -
    # Primary discipline = whichever role had more deliveries that season.
    both = batters_out[["season", "player"]].merge(bowlers_out[["season", "player"]], on=["season", "player"])
    combined = pd.concat([batters_out, bowlers_out], axis=0, ignore_index=True)

    if len(both) > 0:
        balls_lookup = pd.concat([
            bat_stats[["season", "player", "total_balls_faced"]].rename(columns={"total_balls_faced": "balls_bat"}),
            bowl_stats[["season", "player", "total_balls_bowled"]].rename(columns={"total_balls_bowled": "balls_bowl"}),
        ])
        balls_lookup = balls_lookup.groupby(["season", "player"]).sum(min_count=1).reset_index()
        dual = both.merge(balls_lookup, on=["season", "player"], how="left")
        dual["primary"] = np.where(dual["balls_bat"].fillna(0) >= dual["balls_bowl"].fillna(0), "Batter", "Bowler")

        drop_idx = []
        for _, row in dual.iterrows():
            drop_disc = "Bowler" if row["primary"] == "Batter" else "Batter"
            mask = (combined["season"] == row["season"]) & (combined["player"] == row["player"]) & (combined["discipline"] == drop_disc)
            drop_idx.extend(combined[mask].index.tolist())
        combined = combined.drop(index=drop_idx).reset_index(drop=True)

    return combined
