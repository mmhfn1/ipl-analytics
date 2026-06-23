"""
data_loader.py
===============
Loads the raw ball-by-ball CSV and applies all the cleaning / normalization
that every downstream module relies on:

    1. Season label normalization      ('2020/21' -> 2020)
    2. Franchise identity normalization (Delhi Daredevils -> Delhi Capitals, etc.)
    3. Player name normalization        (whitespace + curated alias map)
    4. Derived columns needed everywhere:
         - bowling_team
         - legal_ball     (False for wides/no-balls -> doesn't count toward the over)
         - is_wicket
         - phase           (powerplay / middle / death)
         - innings_legal_ball_no   (1..120 cumulative legal-delivery counter)

Only innings 1 and 2 are kept for the main analytics (super-overs, recorded as
innings 3-6 in this dataset, are a fundamentally different game state and are
dropped here; they could be modeled separately if needed).
"""

from __future__ import annotations
import difflib
import itertools
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. FRANCHISE IDENTITY NORMALIZATION
# ---------------------------------------------------------------------------
# Maps every historical name a franchise has played under onto one canonical
# name. This is what lets Module 6 (Tenure) correctly see a player who was at
# "Delhi Daredevils" in 2018 and "Delhi Capitals" in 2019 as having stayed at
# the SAME franchise, rather than wrongly resetting their tenure to zero.
#
# NOTE: Gujarat Lions / Gujarat Titans and Deccan Chargers / Sunrisers are
# franchise *changes of ownership/identity*, not simple renames, where the
# squad/brand discontinuity is real cricketing history -- they are
# intentionally NOT merged. Only true same-entity renames are merged below.
CANONICAL_TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",  # 2024 rename, no-op pre-2024
    "Rising Pune Supergiants": "Rising Pune Supergiant",            # spelling unification
}


def normalize_team(name: str) -> str:
    if pd.isna(name):
        return name
    return CANONICAL_TEAM_MAP.get(name.strip(), name.strip())


# ---------------------------------------------------------------------------
# 2.5. WINNER RECOVERY FOR TIES / NO-RESULTS
# ---------------------------------------------------------------------------
# The raw `winner` column is not always a team name: a regulation match that
# finishes level is recorded as the literal string "tie" (the actual winner
# is decided by a Super Over, played out as innings 3/4, or 5/6 if the first
# Super Over is ALSO tied -- both occur in this dataset), and a rain-
# abandoned match is recorded as "no result" (no winner exists at all).
# Found via a full audit: 16 of 641 matches in the 2017-2026 window have a
# `winner` that is neither team1 nor team2 -- left unhandled, every
# downstream consumer (win %, standings points, the win-probability model's
# training labels) would silently mistreat genuine Super-Over wins as
# 0-0 non-results, and would train the leverage-index model on a "loss"
# label for a match that was never actually lost. This must run on the RAW
# innings (including 3-6) before the innings-1/2 filter below discards the
# Super Over data needed to recover the true winner.
def _resolve_true_winner(df: pd.DataFrame) -> pd.Series:
    winner = df["winner"].copy()
    is_tie = df["winner"].str.lower() == "tie"
    is_no_result = df["winner"].str.lower().isin(["no result", "no_result"])

    if is_tie.any():
        tie_match_ids = df.loc[is_tie, "match_id"].unique()
        so = df[df["match_id"].isin(tie_match_ids) & df["innings"].isin([3, 4, 5, 6])]
        so_totals = so.groupby(["match_id", "innings"])["runs_total"].sum().reset_index()
        recovered = {}
        for mid, g in so_totals.groupby("match_id"):
            # use the LATEST super-over pair available (5/6 if a second super
            # over was needed because the first one (3/4) also tied)
            pair = [(3, 4), (5, 6)]
            chosen = None
            for a, b in pair:
                ga = g[g["innings"] == a]["runs_total"]
                gb = g[g["innings"] == b]["runs_total"]
                if len(ga) and len(gb):
                    chosen = (a, ga.iloc[0], b, gb.iloc[0])
            if chosen is None:
                continue
            a, runs_a, b, runs_b = chosen
            team_a = df.loc[(df["match_id"] == mid) & (df["innings"] == a), "batting_team"].iloc[0]
            team_b = df.loc[(df["match_id"] == mid) & (df["innings"] == b), "batting_team"].iloc[0]
            recovered[mid] = team_a if runs_a > runs_b else team_b
        winner = winner.where(~is_tie, df["match_id"].map(recovered))

    winner = winner.where(~is_no_result, np.nan)
    return winner


# ---------------------------------------------------------------------------
# 3. SEASON NORMALIZATION
# ---------------------------------------------------------------------------
# BUG FIX: this used to parse the label string directly ('2020/21' -> 2020,
# taking "the first year"). That rule is WRONG for two of the three
# slash-format labels in this dataset: '2007/08' is IPL Season 1, played
# April-June *2008* (not 2007), and '2009/10' is IPL Season 3, played
# March-April *2010* (not 2009) -- there is a SEPARATE, genuine '2009' label
# for the actual 2009 season (played in South Africa). Only '2020/21'
# happens to have its real matches in the first slash-year (2020, played
# Sep-Nov in the UAE due to COVID). There is no single consistent
# first-year/second-year rule that resolves all three correctly by parsing
# the string alone.
#
# The robust fix: don't parse the label at all. Every raw season label's
# matches fall entirely within one calendar year when you look at the
# actual `date` column (verified for all 19 labels in this dataset, no
# season spans a year boundary) -- so the season year is derived from the
# real match dates, which is unambiguous and self-correcting if a future
# data refresh introduces a label this code has never seen before.
def normalize_season(df: pd.DataFrame) -> pd.Series:
    """Returns the normalized season year for every row, derived from each
    raw `season` label's actual match dates (the modal calendar year among
    that label's matches), not from parsing the label string."""
    dates = pd.to_datetime(df["date"])
    label_to_year = (
        pd.DataFrame({"label": df["season"].astype(str).str.strip(), "year": dates.dt.year})
        .groupby("label")["year"]
        .agg(lambda s: int(s.mode().iloc[0]))
    )
    return df["season"].astype(str).str.strip().map(label_to_year)


# ---------------------------------------------------------------------------
# 3. PLAYER NAME NORMALIZATION
# ---------------------------------------------------------------------------
# Curated alias map for known same-player spelling variants. Empty by default
# for this dataset (verified clean - see find_potential_duplicate_names),
# but kept as an explicit extension point: if you re-run this pipeline on a
# different / extended cricsheet-style export, populate this dict with any
# pairs surfaced by find_potential_duplicate_names() after manual review.
PLAYER_ALIAS_MAP: dict = {
    # "Yuvraj Singh ": "Yuvraj Singh",   <- example of the kind of entry this is for
}


def normalize_player(name: str) -> str:
    if pd.isna(name):
        return name
    cleaned = " ".join(name.strip().split())  # collapse internal whitespace too
    return PLAYER_ALIAS_MAP.get(cleaned, cleaned)


def find_potential_duplicate_names(players: list, ratio_threshold: float = 0.90) -> list:
    """QA helper, not used automatically in the pipeline (to avoid silently
    merging two genuinely different players). Run this once against a new
    dataset, eyeball the candidates, and add confirmed pairs to
    PLAYER_ALIAS_MAP above.
    """
    candidates = []
    for a, b in itertools.combinations(sorted(set(players)), 2):
        if abs(len(a) - len(b)) <= 2:
            r = difflib.SequenceMatcher(None, a, b).ratio()
            if r > ratio_threshold:
                candidates.append((a, b, round(r, 3)))
    return candidates


# ---------------------------------------------------------------------------
# 4. PHASE TAGGING
# ---------------------------------------------------------------------------
def assign_phase(over: int) -> str:
    """`over` is 0-indexed in this dataset (overs 0-19 = the 20 overs)."""
    if over <= 5:
        return "Powerplay"      # overs 1-6
    elif over <= 14:
        return "Middle"         # overs 7-15
    else:
        return "Death"          # overs 16-20


# ---------------------------------------------------------------------------
# 5. MAIN LOADER
# ---------------------------------------------------------------------------
def load_and_clean(csv_path: str, season_range: tuple | None = None) -> pd.DataFrame:
    """Load the raw CSV and return a fully cleaned, analysis-ready DataFrame.

    season_range: (start_year, end_year) inclusive, both using the
    normalized "first year" season label. If None (the default), the full
    season range actually present in the file is used -- this is
    deliberate: an earlier version of this pipeline hardcoded (2017, 2022)
    as the default, which silently dropped 2023-2026 data once the source
    CSV was updated to include it. Pass an explicit season_range if you
    want a narrower window.

    NOTE ON 2026: the source CSV's 2026 season is INCOMPLETE (49 matches
    recorded vs. 71-74 for a full season, cut off 2026-05-06, before the
    playoffs). main.py's default season_range is (2017, 2025) specifically
    to exclude it -- a partial season would otherwise produce a wrong
    "champion" (the last chronological match is not the final) and
    distort every season-level stat (standings, streaks, archetype mix)
    for that year. Pass season_range explicitly (e.g. (2017, 2026)) if a
    future, complete version of the 2026 data becomes available."""

    df = pd.read_csv(csv_path, low_memory=False)

    # --- normalize identifiers --------------------------------------------------
    df["season"] = normalize_season(df)
    df["winner"] = _resolve_true_winner(df)
    for col in ["team1", "team2", "toss_winner", "batting_team"]:
        df[col] = df[col].apply(normalize_team)
    df["winner"] = df["winner"].apply(normalize_team)
    for col in ["batter", "bowler", "non_striker", "player_of_match", "wicket_player_out"]:
        df[col] = df[col].apply(normalize_player)

    # --- filter to requested season window & to normal play (drop super overs) -
    if season_range is None:
        season_range = (int(df["season"].min()), int(df["season"].max()))
    lo, hi = season_range
    df = df[(df["season"] >= lo) & (df["season"] <= hi)]
    df = df[df["innings"].isin([1, 2])].copy()

    # --- derived columns ---------------------------------------------------------
    df["bowling_team"] = np.where(df["batting_team"] == df["team1"], df["team2"], df["team1"])

    df["legal_ball"] = (df["extras_wides"] == 0) & (df["extras_noballs"] == 0)

    # BUG FIX: a no-ball IS faced by the batter (they can score off it --
    # including the resulting free hit, which is often a big six) even
    # though it doesn't count as a legal delivery toward the bowler's over.
    # Only a wide is genuinely not faced (it doesn't reach the batter in a
    # hittable way). `legal_ball` above correctly excludes BOTH wides and
    # no-balls for bowler-over-counting / run-rate purposes (that usage is
    # unaffected and unchanged) but was also being reused for a batter's
    # "balls faced" in archetypes.py and tenure_ranking.py, which
    # understated strike rate for the ~0.4% of deliveries that are
    # no-balls -- a small but real inaccuracy, and exactly the kind of
    # no-balls-are-free-hits delivery that tends to go for runs. This
    # separate column is the correct one for any "balls faced" calculation.
    df["ball_faced_by_batter"] = df["extras_wides"] == 0

    df["is_wicket"] = df["wicket_kind"].notna() & (df["wicket_kind"] != "retired hurt") & (df["wicket_kind"] != "retired out")
    df["phase"] = df["over"].apply(assign_phase)

    # cumulative legal-delivery counter within each innings (1..120) -- this is
    # the backbone for every run-rate / "balls remaining" calc downstream,
    # since raw `ball` numbering includes illegal deliveries (wides/no-balls).
    df = df.sort_values(["match_id", "innings", "over", "ball"]).reset_index(drop=True)
    df["innings_legal_ball_no"] = (
        df.groupby(["match_id", "innings"])["legal_ball"].cumsum()
    )

    # running score / wickets within each innings, BEFORE this ball is bowled
    # (i.e. the state the batting side walked into this delivery with) --
    # used heavily by leverage_index.py
    df["cum_runs_before"] = (
        df.groupby(["match_id", "innings"])["runs_total"].cumsum() - df["runs_total"]
    )
    df["cum_wickets_before"] = (
        df.groupby(["match_id", "innings"])["is_wicket"].cumsum().astype(int)
        - df["is_wicket"].astype(int)
    )

    return df


def build_match_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match: teams, winner, season, final score per innings,
    and the actual number of legal deliveries bowled in each innings (needed
    to detect rain-shortened / DLS matches in leverage_index.py)."""

    innings_totals = (
        df.groupby(["match_id", "innings"])
        .agg(runs=("runs_total", "sum"), legal_balls=("legal_ball", "sum"), wickets=("is_wicket", "sum"))
        .reset_index()
    )
    pivot = innings_totals.pivot(index="match_id", columns="innings")
    pivot.columns = [f"{a}_inn{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    match_meta = df.drop_duplicates("match_id")[
        ["match_id", "season", "date", "venue", "city", "team1", "team2", "winner",
         "win_by_runs", "win_by_wickets"]
    ]
    summary = match_meta.merge(pivot, on="match_id", how="left")
    summary["target"] = summary["runs_inn1"] + 1  # standard target, see DLS caveat in leverage_index.py
    return summary
