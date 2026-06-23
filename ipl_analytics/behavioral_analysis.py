"""
behavioral_analysis.py
=======================
Module 4: Behavioral Analysis -- Stable vs. Adaptable players.

GOAL
----
For players who move from one franchise to another between seasons,
determine whether their performance profile is an intrinsic, portable
trait ("Stable" -- they play the same way regardless of team) or a
context-driven one ("Adaptable" -- their numbers shift to resemble
whatever the new team's typical player of that role looks like).

DATA USED
---------
Reuses `archetype_score` from archetypes.py (the same Z-score-derived
composite that drives archetype classification -- e.g. for an Anchor this
is `z_middle_balls_share - z_dismissal_rate`; for a Death Specialist it's
`z_death_sr` or `z_death_econ` depending on discipline). Using the same
score that *defines* the archetype keeps "stability of performance" and
"stability of archetype" conceptually aligned.

KEY DESIGN DECISIONS
---------------------
1. PRIMARY TEAM PER SEASON.
   A player can technically appear for two teams in one season only via
   a mid-season trade (rare in the IPL but not impossible). We resolve
   this by taking whichever team they played the most matches for that
   season as their "primary team" for that year.

2. SAME-ARCHETYPE REQUIREMENT.
   We only compare stability for players who were classified into the
   SAME archetype both before and after the move. If a player's archetype
   itself changed (e.g. Powerplay Aggressor -> Anchor), the "did their
   performance shift" question is confounded by a genuine role change
   rather than the same role performing differently in a new dressing
   room -- those cases are excluded from the comparison entirely (they
   are still visible if you inspect archetypes.py output directly).

3. STABILITY THRESHOLD = 1 LEAGUE-WIDE STD OF THE ARCHETYPE SCORE.
   "Within 1 standard deviation" is judged against the spread of
   archetype_score across ALL qualifying players of that archetype in the
   post-move season -- i.e. is this player's year-over-year shift small
   relative to how much players of this role normally vary across the
   league? This avoids picking an arbitrary fixed cutoff.

4. "ADAPTABLE" = CONVERGENCE TOWARD THE NEW TEAM'S ARCHETYPE MEAN.
   For players whose shift exceeds the stability threshold, we check
   whether the shift moved them CLOSER to the new team's average
   archetype_score for players of that same archetype that season. This
   produces a diagnostic flag (`moved_toward_new_team_mean`) alongside the
   primary Stable/Adaptable label, so you can see not just THAT a player
   changed, but whether the change looks like genuine team-context
   adaptation versus a coincidental/unrelated swing.
"""

import numpy as np
import pandas as pd


def compute_player_primary_team(player_season_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve each player's single "primary team" per season as whichever
    team they played the most matches for (handles the rare mid-season
    trade case by picking the dominant team).
    """
    idx = player_season_matches.groupby(['season', 'player'])['matches_played'].idxmax()
    primary = player_season_matches.loc[idx, ['season', 'player', 'team', 'matches_played']]
    return primary.reset_index(drop=True)


def detect_team_changes(primary_team: pd.DataFrame) -> pd.DataFrame:
    """
    Find every player who switched primary team between two seasons they
    were active in (seasons need not be literally consecutive calendar
    years -- this compares each player's consecutive ACTIVE seasons, so a
    player who skipped a year is still compared correctly across the gap).
    """
    sorted_pt = primary_team.sort_values(['player', 'season']).copy()
    sorted_pt['prev_team'] = sorted_pt.groupby('player')['team'].shift(1)
    sorted_pt['prev_season'] = sorted_pt.groupby('player')['season'].shift(1)

    changed = sorted_pt[
        sorted_pt['prev_team'].notna() & (sorted_pt['team'] != sorted_pt['prev_team'])
    ].copy()

    changed = changed.rename(columns={
        'team': 'new_team',
        'season': 'season_after',
        'prev_team': 'old_team',
        'prev_season': 'season_before',
    })
    changed['season_before'] = changed['season_before'].astype(int)

    return changed[['player', 'old_team', 'season_before', 'new_team', 'season_after']].reset_index(drop=True)


def analyze_behavioral_stability(team_changes: pd.DataFrame,
                                  archetypes_df: pd.DataFrame,
                                  primary_team: pd.DataFrame,
                                  std_threshold: float = 1.0) -> pd.DataFrame:
    """
    For each detected team change, compare the player's archetype_score in
    the season before the move (old team) vs. the season after (new team),
    and classify as Stable or Adaptable.

    Parameters
    ----------
    team_changes : output of detect_team_changes()
    archetypes_df : output of archetypes.classify_archetypes()
    primary_team : output of compute_player_primary_team(), used to
        compute each team's mean archetype_score per archetype/season
        (the "context" a moving player is joining).
    std_threshold : number of league-wide standard deviations of
        archetype_score (within that archetype/season) a player's
        year-over-year shift must exceed to be called "Adaptable" rather
        than "Stable". Default 1.0 per spec.

    Returns
    -------
    DataFrame with one row per qualifying team-change (same archetype
    before and after), including the Stable/Adaptable classification and
    the moved_toward_new_team_mean diagnostic flag.
    """
    arch = archetypes_df[['season', 'player', 'archetype', 'archetype_score']]

    merged = team_changes.merge(
        arch.rename(columns={
            'season': 'season_before', 'archetype': 'archetype_before', 'archetype_score': 'score_before'
        }),
        on=['season_before', 'player'], how='inner'
    )
    merged = merged.merge(
        arch.rename(columns={
            'season': 'season_after', 'archetype': 'archetype_after', 'archetype_score': 'score_after'
        }),
        on=['season_after', 'player'], how='inner'
    )

    # Only meaningful to compare "stability" when the player kept the same
    # archetype across the move -- otherwise this is a role change, not a
    # stability/adaptability question.
    merged = merged[merged['archetype_before'] == merged['archetype_after']].copy()
    merged = merged.rename(columns={'archetype_before': 'archetype'}).drop(columns=['archetype_after'])

    if merged.empty:
        return merged.assign(delta=[], league_std=[], team_archetype_mean=[],
                              classification=[], moved_toward_new_team_mean=[])

    merged['delta'] = merged['score_after'] - merged['score_before']

    # League-wide spread of archetype_score for this archetype, in the
    # post-move season -- the yardstick for "how big a shift is normal".
    league_std = arch.groupby(['season', 'archetype'])['archetype_score'].std()
    league_std = league_std.rename('league_std').reset_index().rename(columns={'season': 'season_after'})
    merged = merged.merge(league_std, on=['season_after', 'archetype'], how='left')

    # New team's mean archetype_score for this archetype/season -- the
    # "context" the player is moving into.
    team_arch = primary_team.merge(arch, on=['season', 'player'], how='inner')
    team_means = (
        team_arch.groupby(['season', 'team', 'archetype'])['archetype_score']
        .mean()
        .rename('team_archetype_mean')
        .reset_index()
        .rename(columns={'season': 'season_after', 'team': 'new_team'})
    )
    merged = merged.merge(team_means, on=['season_after', 'new_team', 'archetype'], how='left')

    merged['classification'] = np.where(
        merged['delta'].abs() <= merged['league_std'], 'Stable', 'Adaptable'
    )
    merged['moved_toward_new_team_mean'] = (
        (merged['score_after'] - merged['team_archetype_mean']).abs()
        < (merged['score_before'] - merged['team_archetype_mean']).abs()
    )

    cols = ['player', 'archetype', 'old_team', 'season_before', 'score_before',
            'new_team', 'season_after', 'score_after', 'delta', 'league_std',
            'team_archetype_mean', 'classification', 'moved_toward_new_team_mean']
    return merged[cols].reset_index(drop=True)


def summarize_behavioral_patterns(behavioral_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up Stable vs Adaptable counts (and adaptation-direction hit rate)
    by archetype, for a quick league-wide view of which roles tend to be
    portable skills vs. team-system-dependent ones.
    """
    if behavioral_df.empty:
        return pd.DataFrame(columns=['archetype', 'n_moves', 'n_stable', 'n_adaptable',
                                      'pct_adaptable', 'pct_adaptable_moved_toward_new_team'])

    summary = behavioral_df.groupby('archetype').apply(
        lambda g: pd.Series({
            'n_moves': len(g),
            'n_stable': (g['classification'] == 'Stable').sum(),
            'n_adaptable': (g['classification'] == 'Adaptable').sum(),
            'pct_adaptable': 100.0 * (g['classification'] == 'Adaptable').mean(),
            'pct_adaptable_moved_toward_new_team': (
                100.0 * g.loc[g['classification'] == 'Adaptable', 'moved_toward_new_team_mean'].mean()
                if (g['classification'] == 'Adaptable').any() else float('nan')
            ),
        }),
        include_groups=False
    ).reset_index()

    return summary.sort_values('n_moves', ascending=False).reset_index(drop=True)
