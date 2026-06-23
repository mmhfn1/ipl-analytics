"""
strategy_mapping.py
====================
Module 3: Strategic Philosophy Mapping.

GOAL
----
For each (season, team), compute the "Archetype Mix" -- the percentage of
playing-XI participation accounted for by each of the four archetypes
(Powerplay Aggressor / Anchor / Death Specialist / Middle-Over Squeezer)
-- and correlate that mix against team win percentage to see which
strategic philosophies actually win matches.

KEY DESIGN DECISIONS
---------------------
1. PARTICIPATION WEIGHT, NOT A FLAT "11 PLAYERS" ASSUMPTION.
   The raw ball-by-ball data has no explicit "playing XI" list -- we only
   know who actually faced a ball / bowled a ball / ran as non-striker in
   a given match. We proxy "XI membership" with the number of matches a
   player appeared in for a team in a season. A player who featured in
   12 of a team's 14 matches contributes more weight to that team's
   season-level archetype mix than a player who featured in 2 matches
   (e.g. an injury replacement). This is more faithful to the team's
   *actual on-field identity* than treating every squad member equally.

2. EVERY PLAYER GETS A REAL ARCHETYPE -- NO "UNCLASSIFIED" BUCKET.
   archetypes.py classifies every player-season against a stable league
   baseline (see that module's docstring), so mix percentages sum to
   100% across exactly the four real archetypes. A separate
   `compute_low_sample_share` below tracks how much of a team's
   participation came from thin-sample players, as a distinct, honestly
   labeled metric rather than folding "low sample" into "couldn't be
   classified".

3. WIN % COMPUTED FROM match_summary, NOT FROM THE BALL-BY-BALL TABLE.
   match_summary already has one row per match with a clean `winner`
   column, so we reshape it to one row per (team, match) and aggregate.
"""

import pandas as pd
from scipy import stats


def build_player_match_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (match_id, season, player, team) appearance table by unioning
    every distinct batter / non_striker / bowler role seen in a match,
    mapped to the team they were representing on that ball (batting_team
    for batter/non_striker, bowling_team for bowler).

    Returns a deduplicated long table: one row per player who appeared for
    a given team in a given match (regardless of how many roles/balls).
    """
    batters = df[['match_id', 'season', 'batting_team', 'batter']].rename(
        columns={'batting_team': 'team', 'batter': 'player'}
    )
    non_strikers = df[['match_id', 'season', 'batting_team', 'non_striker']].rename(
        columns={'batting_team': 'team', 'non_striker': 'player'}
    )
    bowlers = df[['match_id', 'season', 'bowling_team', 'bowler']].rename(
        columns={'bowling_team': 'team', 'bowler': 'player'}
    )

    appearances = pd.concat([batters, non_strikers, bowlers], ignore_index=True)
    appearances = appearances.drop_duplicates(subset=['match_id', 'season', 'team', 'player'])
    return appearances.reset_index(drop=True)


def compute_player_season_matches(appearances: pd.DataFrame) -> pd.DataFrame:
    """
    Count distinct matches played per (season, player, team). This is the
    participation weight used as a proxy for "share of the XI" since the
    dataset has no explicit starting-lineup flag.
    """
    counts = (
        appearances
        .groupby(['season', 'team', 'player'])['match_id']
        .nunique()
        .reset_index(name='matches_played')
    )
    return counts


def compute_archetype_mix(player_season_matches: pd.DataFrame,
                           archetypes: pd.DataFrame) -> pd.DataFrame:
    """
    Merge participation weight with archetype labels and compute, for each
    (season, team), the percentage of total participation-weight belonging
    to each of the four archetypes (Powerplay Aggressor / Anchor / Death
    Specialist / Middle-Over Squeezer). Every player-season now carries a
    real archetype label (see archetypes.py -- the small-sample-only
    "Unclassified" catch-all was removed in favor of classifying everyone
    against a stable league baseline and flagging confidence separately),
    so mix percentages sum to 100% across exactly these four categories
    with no leftover bucket.
    """
    archetype_lookup = archetypes[['season', 'player', 'archetype', 'is_low_sample']].drop_duplicates(
        subset=['season', 'player']
    )

    merged = player_season_matches.merge(
        archetype_lookup, on=['season', 'player'], how='inner'
    )

    weight_by_archetype = (
        merged
        .groupby(['season', 'team', 'archetype'])['matches_played']
        .sum()
        .reset_index(name='weight')
    )

    team_season_total = (
        weight_by_archetype
        .groupby(['season', 'team'])['weight']
        .transform('sum')
    )
    weight_by_archetype['pct'] = 100.0 * weight_by_archetype['weight'] / team_season_total

    mix_wide = weight_by_archetype.pivot_table(
        index=['season', 'team'], columns='archetype', values='pct', fill_value=0.0
    ).reset_index()
    mix_wide.columns.name = None

    return mix_wide


def compute_low_sample_share(player_season_matches: pd.DataFrame,
                              archetypes: pd.DataFrame) -> pd.DataFrame:
    """
    Per (season, team), the percentage of total participation-weight that
    came from low-sample player-seasons (fewer than archetypes.MIN_BALLS_*
    involvements that season -- fringe squad members, injury cover, etc).

    This REPLACES the old "Unclassified %" metric. The earlier version of
    this pipeline correlated "Unclassified share" with win % and reported
    it as the suite's single strongest signal (r ~ -0.47) -- but that
    conflated two different things: "the model couldn't confidently place
    this player" and "this player barely featured". Every player now gets
    a real archetype regardless of sample size (see archetypes.py), so
    that conflation no longer exists. Low-Sample Share is the honest,
    cleanly-defined replacement: how much of a team's season was carried
    by fringe/thin-sample participation, independent of which archetype
    those players were nominally placed in.
    """
    merged = player_season_matches.merge(
        archetypes[['season', 'player', 'is_low_sample']].drop_duplicates(subset=['season', 'player']),
        on=['season', 'player'], how='inner'
    )
    by_team = merged.groupby(['season', 'team']).apply(
        lambda g: pd.Series({
            'low_sample_weight': g.loc[g['is_low_sample'], 'matches_played'].sum(),
            'total_weight': g['matches_played'].sum(),
        }),
        include_groups=False,
    ).reset_index()
    by_team['low_sample_pct'] = 100.0 * by_team['low_sample_weight'] / by_team['total_weight']
    return by_team[['season', 'team', 'low_sample_pct']]


def compute_team_win_pct(match_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape match_summary (one row per match) into one row per team per
    match, then aggregate to win percentage per (season, team).

    Matches with no winner (rain-abandoned "no result" matches, of which
    there are 6 in 2017-2026 -- see data_loader._resolve_true_winner) are
    excluded from both the numerator and denominator: a no-result wasn't
    won OR lost, so counting it as a "played" match would understate win %
    for both teams involved.
    """
    team1 = match_summary[['season', 'match_id', 'team1', 'winner']].rename(
        columns={'team1': 'team'}
    )
    team2 = match_summary[['season', 'match_id', 'team2', 'winner']].rename(
        columns={'team2': 'team'}
    )
    long = pd.concat([team1, team2], ignore_index=True)
    long = long[long['winner'].notna()]  # drop no-result matches
    long['won'] = (long['team'] == long['winner']).astype(int)

    win_pct = (
        long
        .groupby(['season', 'team'])
        .agg(matches_played=('match_id', 'nunique'), matches_won=('won', 'sum'))
        .reset_index()
    )
    win_pct['win_pct'] = 100.0 * win_pct['matches_won'] / win_pct['matches_played']
    return win_pct


def correlate_mix_with_wins(archetype_mix: pd.DataFrame,
                             win_pct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge archetype mix % with team win % per (season, team), then run a
    Pearson correlation between each archetype's mix percentage and win
    percentage across all team-seasons.

    Returns (correlation_summary, merged_data):
      - correlation_summary: one row per archetype, with pearson_r,
        p_value, and n_observations, sorted by pearson_r descending (most
        positively associated with winning first).
      - merged_data: the full team-season table with mix % and win % side
        by side, for further inspection or plotting.
    """
    merged = archetype_mix.merge(win_pct, on=['season', 'team'], how='inner')

    archetype_cols = [c for c in archetype_mix.columns if c not in ('season', 'team')]

    rows = []
    for col in archetype_cols:
        # Need at least 3 paired observations for a meaningful correlation
        valid = merged[[col, 'win_pct']].dropna()
        if len(valid) < 3 or valid[col].std() == 0:
            r, p = float('nan'), float('nan')
        else:
            r, p = stats.pearsonr(valid[col], valid['win_pct'])
        rows.append({
            'archetype': col,
            'pearson_r': r,
            'p_value': p,
            'n_observations': len(valid),
        })

    correlation_summary = pd.DataFrame(rows).sort_values(
        'pearson_r', ascending=False
    ).reset_index(drop=True)

    return correlation_summary, merged
