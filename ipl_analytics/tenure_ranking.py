"""
tenure_ranking.py
===================
Module 6: Tenure & Ranking Impact.

Three sub-analyses:

(A) TENURE
    Continuous years a player has spent with a SINGLE franchise, using the
    normalized team identities from data_loader (so e.g. Delhi Daredevils
    -> Delhi Capitals counts as one continuous spell, not a reset).

(B) TEAM TENURE DENSITY vs STRATEGY CONSISTENCY
    Team Tenure Density = participation-weighted average tenure of a
    team's season squad. Strategy Consistency = how much a team's
    Archetype Mix (from strategy_mapping.py) changes from one season to
    the next (a Euclidean-distance "shift magnitude" between consecutive
    seasons' mix vectors -- LOWER means a more stable strategic identity).
    We correlate the two at team-season granularity.

(C) TOP-5 RANKING STREAKS
    We reconstruct PROGRESSIVE in-season standings (2 points/win, no
    ties/no-results in this date range -- confirmed during data
    exploration) match-by-match in chronological order, determine each
    team's league rank ENTERING each match (i.e. based on results to date,
    not including the match's own outcome -- this avoids leaking the
    match result into the "was this team in contention" label), and find
    maximal consecutive-match windows where a team's rank was Top-5.
    Player strike rate / economy during those windows is then compared to
    their full-season average to see whether players perform differently
    when their team is in genuine title contention.

DESIGN NOTE ON TIE-BREAKING IN STANDINGS
------------------------------------------
The dataset has no Net Run Rate (NRR) column, which the real IPL uses as
the primary tiebreaker. We use standard "competition ranking" (teams with
equal points share the same rank, e.g. 1,2,2,4) rather than fabricating an
NRR-based tiebreak. This is a documented simplification: real-world Top-5
boundaries can differ slightly from ours when several teams are level on
points.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# (A) Tenure
# ---------------------------------------------------------------------------

def compute_tenure(primary_team: pd.DataFrame) -> pd.DataFrame:
    """
    Continuous tenure (in years/seasons) at a single franchise, ending at
    each season a player was active.

    A tenure "spell" resets when either:
      - the player's primary team changes season-over-season, OR
      - there is a gap in the player's active seasons (e.g. they sat out
        a season -- since we cannot distinguish "still contracted but
        injured" from "left and came back", a gap is conservatively
        treated as a break in continuity).

    This is fully vectorized: a "new spell" flag is computed per row, then
    cumsum'd into a spell id, then cumcount'd within each spell to get the
    1-indexed tenure_years for that row.
    """
    df = primary_team.sort_values(['player', 'season']).copy()
    df['prev_team'] = df.groupby('player')['team'].shift(1)
    df['prev_season'] = df.groupby('player')['season'].shift(1)

    is_continuous = (df['team'] == df['prev_team']) & (df['season'] == df['prev_season'] + 1)
    df['new_spell'] = (~is_continuous).astype(int)
    df['spell_id'] = df.groupby('player')['new_spell'].cumsum()
    df['tenure_years'] = df.groupby(['player', 'spell_id']).cumcount() + 1

    return df[['season', 'player', 'team', 'tenure_years']].reset_index(drop=True)


# ---------------------------------------------------------------------------
# (B) Team Tenure Density vs Strategy Consistency
# ---------------------------------------------------------------------------

def compute_team_tenure_density(tenure_df: pd.DataFrame,
                                 player_season_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Participation-weighted average tenure_years across a team's
    season squad -- "how experienced/settled is this team's roster".
    """
    merged = player_season_matches.merge(
        tenure_df, on=['season', 'player', 'team'], how='inner'
    )

    density = (
        merged.groupby(['season', 'team'])
        .apply(lambda g: np.average(g['tenure_years'], weights=g['matches_played']))
        .rename('team_tenure_density')
        .reset_index()
    )
    return density


def compute_strategy_consistency_by_transition(archetype_mix: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, and each consecutive pair of seasons it appears in the
    data, compute the Euclidean-distance "shift magnitude" between the
    archetype mix vectors of the two seasons. A small value means the
    team's strategic identity (its blend of archetypes) barely changed
    year-over-year; a large value means a strategic overhaul.

    Returns one row per (season, team) for every season EXCEPT a team's
    first appearance (which has no prior season to diff against).
    """
    archetype_cols = [c for c in archetype_mix.columns if c not in ('season', 'team')]

    sorted_mix = archetype_mix.sort_values(['team', 'season']).copy()
    diffs = sorted_mix.groupby('team')[archetype_cols].diff()
    sorted_mix['mix_change_magnitude'] = np.sqrt((diffs[archetype_cols] ** 2).sum(axis=1))

    # GAP SAFETY: .diff() compares adjacent ROWS, which is only a valid
    # "year-over-year" comparison if consecutive rows are actually
    # consecutive seasons. No team currently has a gap year in 2017-2026,
    # but if a future data refresh introduces one (e.g. a suspended
    # franchise), this would silently treat a multi-year gap as a single
    # season's change. Guard explicitly rather than relying on that
    # coincidence.
    prev_season = sorted_mix.groupby('team')['season'].shift(1)
    is_consecutive = (sorted_mix['season'] - prev_season) == 1
    sorted_mix.loc[~is_consecutive, 'mix_change_magnitude'] = np.nan

    # The first season for a team has no prior season -> diff is all-NaN
    # for every archetype column -> sqrt(sum of squared NaNs) is NaN too,
    # so this naturally filters out correctly via dropna below.
    result = sorted_mix[['season', 'team', 'mix_change_magnitude']].dropna(
        subset=['mix_change_magnitude']
    )
    return result.reset_index(drop=True)


def correlate_tenure_density_with_consistency(team_tenure_density: pd.DataFrame,
                                               strategy_consistency: pd.DataFrame) -> dict:
    """
    Correlate Team Tenure Density (that season) against the mix-change
    magnitude entering that season (lower magnitude = more consistent
    strategy). A NEGATIVE correlation supports the hypothesis "more
    experienced/settled rosters -> more stable year-over-year strategy".

    Returns a dict of summary stats plus the merged team-season table for
    further inspection.
    """
    from scipy import stats as scipy_stats

    merged = strategy_consistency.merge(
        team_tenure_density, on=['season', 'team'], how='inner'
    )

    if len(merged) < 3 or merged['team_tenure_density'].std() == 0:
        r, p = float('nan'), float('nan')
    else:
        r, p = scipy_stats.pearsonr(merged['team_tenure_density'], merged['mix_change_magnitude'])

    summary = {'pearson_r': r, 'p_value': p, 'n_observations': len(merged)}
    return summary, merged


# ---------------------------------------------------------------------------
# (C) Progressive standings & Top-5 ranking streaks
# ---------------------------------------------------------------------------

def build_progressive_standings(match_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through every match in chronological order within each season,
    and for both participating teams record their league RANK ENTERING
    that match (based on cumulative points from prior matches only -- the
    match's own outcome does not influence its own "entering rank").

    Standard competition ranking is used for ties (equal points share the
    same rank, e.g. 1, 2, 2, 4 -- see module docstring for why we don't
    attempt an NRR-based tiebreak).

    This requires a sequential walk (points after match N affect rank for
    match N+1), so it is implemented as a single pass over matches sorted
    by date rather than a vectorized groupby -- with ~373 matches in the
    2017-2022 range this is fast.
    """
    matches = match_summary.copy()
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values(['season', 'date', 'match_id']).reset_index(drop=True)

    points = {}  # (season, team) -> cumulative points so far
    records = []

    for row in matches.itertuples(index=False):
        season, t1, t2, winner = row.season, row.team1, row.team2, row.winner

        points.setdefault((season, t1), 0)
        points.setdefault((season, t2), 0)

        season_points = {team: pts for (s, team), pts in points.items() if s == season}
        ranked_teams = sorted(season_points.items(), key=lambda kv: -kv[1])

        rank_map = {}
        prev_pts, current_rank = None, 0
        for i, (team, pts) in enumerate(ranked_teams):
            if pts != prev_pts:
                current_rank = i + 1
                prev_pts = pts
            rank_map[team] = current_rank

        records.append({'match_id': row.match_id, 'season': season, 'date': row.date,
                         'team': t1, 'rank_entering_match': rank_map[t1]})
        records.append({'match_id': row.match_id, 'season': season, 'date': row.date,
                         'team': t2, 'rank_entering_match': rank_map[t2]})

        # BUG FIX: a genuine no-result (rain-abandoned, winner is NaN -- see
        # data_loader._resolve_true_winner) awards both teams 1 point each
        # under real IPL rules, not 0-0. The previous version only ever
        # credited 2 points to a literal `winner` match, silently treating
        # every no-result as 0-0, which understates both teams' points
        # tallies for the rest of that season's standings reconstruction.
        if pd.isna(winner):
            points[(season, t1)] += 1
            points[(season, t2)] += 1
        else:
            points[(season, t1)] += 2 if winner == t1 else 0
            points[(season, t2)] += 2 if winner == t2 else 0

    return pd.DataFrame(records)


def identify_top5_streaks(standings: pd.DataFrame, min_streak_length: int = 2) -> pd.DataFrame:
    """
    Find maximal runs of consecutive matches (by date, per team) where a
    team's rank_entering_match was in the Top 5. Short blips of length 1
    are excluded by default (min_streak_length=2) since a single-match
    "streak" doesn't represent sustained contention.
    """
    df = standings.sort_values(['season', 'team', 'date']).copy()
    df['in_top5'] = df['rank_entering_match'] <= 5

    # standard "new block starts when the flag changes" pattern
    flag_changed = df['in_top5'] != df.groupby(['season', 'team'])['in_top5'].shift(1)
    df['block_id'] = flag_changed.groupby([df['season'], df['team']]).cumsum()

    streak_blocks = df[df['in_top5']].groupby(['season', 'team', 'block_id'])
    streaks = streak_blocks.agg(
        streak_start_date=('date', 'min'),
        streak_end_date=('date', 'max'),
        streak_length=('match_id', 'count'),
    ).reset_index()

    match_lists = streak_blocks['match_id'].apply(list).rename('match_ids').reset_index()
    streaks = streaks.merge(match_lists, on=['season', 'team', 'block_id'])

    streaks = streaks[streaks['streak_length'] >= min_streak_length].drop(columns='block_id')
    return streaks.reset_index(drop=True)


def compare_player_performance_during_streaks(df: pd.DataFrame,
                                               streaks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every player, compute strike rate (batting) and economy (bowling)
    during their team's Top-5 streak windows, vs. their full-season
    average, using the SAME formulas as archetypes.py (legal balls faced
    for SR; runs excluding byes/leg-byes per legal over for economy) so
    the two are directly comparable.

    Returns (batting_comparison, bowling_comparison) DataFrames, each with
    season / streak metrics side by side and a `delta` column
    (streak value minus season value -- positive SR delta or negative
    economy delta both indicate a step-UP in performance during contention).
    """
    streak_matches = streaks.explode('match_ids').rename(columns={'match_ids': 'match_id'})
    streak_matches = streak_matches[['season', 'team', 'match_id']].drop_duplicates()

    # ---- Batting: strike rate ----
    bat = df.copy()
    bat['balls_faced'] = bat['ball_faced_by_batter'].astype(int)  # only wides don't count (no-balls do)

    season_bat = bat.groupby(['season', 'batter']).agg(
        season_runs=('runs_batter', 'sum'), season_balls=('balls_faced', 'sum')
    ).reset_index()
    season_bat['season_sr'] = np.where(
        season_bat['season_balls'] > 0, season_bat['season_runs'] / season_bat['season_balls'] * 100, np.nan
    )

    streak_bat_rows = bat.merge(
        streak_matches.rename(columns={'team': 'batting_team'}),
        on=['season', 'match_id', 'batting_team'], how='inner'
    )
    streak_bat = streak_bat_rows.groupby(['season', 'batter']).agg(
        streak_runs=('runs_batter', 'sum'), streak_balls=('balls_faced', 'sum')
    ).reset_index()
    streak_bat['streak_sr'] = np.where(
        streak_bat['streak_balls'] > 0, streak_bat['streak_runs'] / streak_bat['streak_balls'] * 100, np.nan
    )

    batting_compare = season_bat.merge(streak_bat, on=['season', 'batter'], how='inner')
    batting_compare = batting_compare.rename(columns={'batter': 'player'})
    batting_compare['sr_delta'] = batting_compare['streak_sr'] - batting_compare['season_sr']
    # require a minimal sample in BOTH windows to avoid noisy 1-2 ball SR swings
    batting_compare = batting_compare[
        (batting_compare['season_balls'] >= 30) & (batting_compare['streak_balls'] >= 15)
    ].reset_index(drop=True)

    # ---- Bowling: economy ----
    bowl = df.copy()
    bowl['legal_ball_int'] = bowl['legal_ball'].astype(int)
    bowl['runs_conceded'] = bowl['runs_total'] - bowl['extras_byes'] - bowl['extras_legbyes']

    season_bowl = bowl.groupby(['season', 'bowler']).agg(
        season_runs_conceded=('runs_conceded', 'sum'), season_balls=('legal_ball_int', 'sum')
    ).reset_index()
    season_bowl['season_economy'] = np.where(
        season_bowl['season_balls'] > 0,
        season_bowl['season_runs_conceded'] / (season_bowl['season_balls'] / 6), np.nan
    )

    streak_bowl_rows = bowl.merge(
        streak_matches.rename(columns={'team': 'bowling_team'}),
        on=['season', 'match_id', 'bowling_team'], how='inner'
    )
    streak_bowl = streak_bowl_rows.groupby(['season', 'bowler']).agg(
        streak_runs_conceded=('runs_conceded', 'sum'), streak_balls=('legal_ball_int', 'sum')
    ).reset_index()
    streak_bowl['streak_economy'] = np.where(
        streak_bowl['streak_balls'] > 0,
        streak_bowl['streak_runs_conceded'] / (streak_bowl['streak_balls'] / 6), np.nan
    )

    bowling_compare = season_bowl.merge(streak_bowl, on=['season', 'bowler'], how='inner')
    bowling_compare = bowling_compare.rename(columns={'bowler': 'player'})
    bowling_compare['economy_delta'] = bowling_compare['streak_economy'] - bowling_compare['season_economy']
    bowling_compare = bowling_compare[
        (bowling_compare['season_balls'] >= 30) & (bowling_compare['streak_balls'] >= 15)
    ].reset_index(drop=True)

    return batting_compare, bowling_compare
