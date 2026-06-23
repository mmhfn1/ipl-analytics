"""
liability.py
=============
Module 5: Liability Identification.

GOAL
----
1. Flag players in the bottom 25th percentile of archetype_score,
   LEAGUE-WIDE within their own archetype/season (not within-team), as
   "Underperforming" for their role.
2. Project the "Replacement Delta": the estimated change in team win %
   if an underperforming player were swapped out for a league-median
   performer of the same archetype.

CALIBRATION PHILOSOPHY
-----------------------
Rather than picking an arbitrary "1 percentile point = X% win probability"
conversion constant, we calibrate the archetype_score -> win% relationship
empirically, consistent with the same "calibrated heuristic" philosophy
used in leverage_index.py (weights learned from data, not hand-picked).

For each archetype we compute, per (season, team), the PARTICIPATION-
WEIGHTED AVERAGE archetype_score among that team's players of that
archetype (i.e. "how good was this team's collection of Anchors / Death
Specialists / etc that year"), and regress team win % against it via
simple OLS (scipy.stats.linregress). The resulting slope tells us:
"win% moves by `slope` percentage points per 1-unit increase in this
archetype's average quality score for the team."

A liability player's Replacement Delta is then:

    replacement_delta_win_pct = slope[archetype] * weight_share * (median_score - player_score)

where `weight_share` is the player's share of the team's TOTAL
participation-weight within that archetype that season (a player who
featured in 14 of the team's 16 "Anchor" matches moves the team's
weighted-average score far more than one who only played 2 of them) --
i.e. we estimate how much swapping just this one player shifts the
team's weighted-average score for that archetype, then translate that
shift into a win% impact via the regression slope.

LIMITATIONS (documented, not hidden):
- This is a simple-OLS, team-season-level calibration on a modest sample
  (~50 team-seasons for 2017-2022). Slopes for less common archetypes
  will be noisier and should be read with that caveat.
- It assumes the win%-archetype_score relationship is linear and that
  archetypes contribute independently/additively -- a simplification of
  a genuinely multivariate team-composition effect.
- It captures a REGULAR-SEASON statistical association, not a causal
  guarantee that swapping one specific player produces that exact
  outcome in practice.
"""

import numpy as np
import pandas as pd
from scipy import stats

LIABILITY_PERCENTILE = 25.0


def flag_underperformers(archetypes_df: pd.DataFrame,
                          percentile: float = LIABILITY_PERCENTILE) -> pd.DataFrame:
    """
    Flag players whose archetype_score falls at or below the bottom
    `percentile` (default 25th) among LEAGUE-WIDE peers of the same
    archetype and season. Also attaches the league median score for that
    archetype/season -- the benchmark used in the Replacement Delta calc.
    """
    df = archetypes_df.copy()

    # transform() preserves the original row index/columns cleanly (unlike
    # groupby.apply with a DataFrame-returning function, which in recent
    # pandas versions can silently drop the group-key columns).
    grp = df.groupby(['season', 'archetype'])['archetype_score']
    df['percentile_threshold'] = grp.transform(lambda s: np.percentile(s, percentile))
    df['league_median_score'] = grp.transform('median')
    df['is_underperformer'] = df['archetype_score'] <= df['percentile_threshold']

    return df


def compute_team_archetype_quality(player_season_matches: pd.DataFrame,
                                    archetypes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Participation-weighted average archetype_score per (season, team,
    archetype) -- "how strong was this team's collection of players in
    this archetype, that season".
    """
    merged = player_season_matches.merge(
        archetypes_df[['season', 'player', 'archetype', 'archetype_score']],
        on=['season', 'player'], how='inner'  # only classified players carry a score
    )

    def _agg(g):
        wavg = np.average(g['archetype_score'], weights=g['matches_played'])
        return pd.Series({
            'team_archetype_quality': wavg,
            'total_archetype_weight': g['matches_played'].sum(),
        })

    quality = (
        merged.groupby(['season', 'team', 'archetype'])
        .apply(_agg)
        .reset_index()
    )
    return quality


def calibrate_archetype_win_impact(team_archetype_quality: pd.DataFrame,
                                    win_pct: pd.DataFrame) -> pd.DataFrame:
    """
    For each archetype, regress team win% on team_archetype_quality across
    all team-seasons via simple OLS. Returns the slope (win% points per
    1-unit increase in archetype_score) plus fit diagnostics.
    """
    merged = team_archetype_quality.merge(
        win_pct[['season', 'team', 'win_pct']], on=['season', 'team'], how='inner'
    )

    rows = []
    for archetype, g in merged.groupby('archetype'):
        if len(g) < 4 or g['team_archetype_quality'].std() == 0:
            slope, intercept, r, p = float('nan'), float('nan'), float('nan'), float('nan')
        else:
            result = stats.linregress(g['team_archetype_quality'], g['win_pct'])
            slope, intercept, r, p = result.slope, result.intercept, result.rvalue, result.pvalue
        rows.append({
            'archetype': archetype, 'slope': slope, 'intercept': intercept,
            'r_value': r, 'p_value': p, 'n_team_seasons': len(g),
        })
    return pd.DataFrame(rows).sort_values('archetype').reset_index(drop=True)


def compute_replacement_delta(underperformers: pd.DataFrame,
                               team_archetype_quality: pd.DataFrame,
                               slopes: pd.DataFrame,
                               primary_team: pd.DataFrame) -> pd.DataFrame:
    """
    For every flagged underperformer, project the win% impact of replacing
    them with a league-median performer of their archetype.

    Steps:
    1. Determine the player's team that season (primary_team).
    2. Determine the player's weight share within their archetype on that
       team (their matches_played / total_archetype_weight for that team).
    3. score_gap = league_median_score - player_score (>= 0 by
       construction, since these are flagged underperformers).
    4. projected_quality_shift = weight_share * score_gap (how much the
       team's weighted-average archetype_score would rise if this player
       alone were swapped to the league median).
    5. replacement_delta_win_pct = slope[archetype] * projected_quality_shift.
    """
    liabilities = underperformers[underperformers['is_underperformer']].copy()

    liabilities = liabilities.merge(
        primary_team[['season', 'player', 'team', 'matches_played']],
        on=['season', 'player'], how='inner'
    )

    liabilities = liabilities.merge(
        team_archetype_quality[['season', 'team', 'archetype', 'total_archetype_weight']],
        on=['season', 'team', 'archetype'], how='left'
    )

    liabilities['weight_share'] = liabilities['matches_played'] / liabilities['total_archetype_weight']
    liabilities['score_gap'] = liabilities['league_median_score'] - liabilities['archetype_score']
    liabilities['projected_quality_shift'] = liabilities['weight_share'] * liabilities['score_gap']

    liabilities = liabilities.merge(slopes[['archetype', 'slope', 'p_value']], on='archetype', how='left')

    # BUG FIX: replacement_delta_win_pct = slope * projected_quality_shift was
    # producing NEGATIVE "upside" values for flagged underperformers whenever
    # an archetype's league-wide OLS slope came back negative (e.g. Anchor:
    # slope=-3.67, p=0.34 -- not statistically significant). Mechanically that
    # says "replacing this team's worst Anchor with a league-median Anchor
    # would make the team WORSE", which is incoherent: score_gap is already
    # non-negative by construction for a flagged underperformer (they're
    # below the league median by definition), so the projected effect of
    # upgrading them can never be negative. A negative/non-significant team-
    # level slope reflects confounding or noise in the regression, not a
    # real penalty for fielding a better player. We floor the slope at 0
    # specifically for this calculation: archetypes with a non-significant or
    # wrong-signed slope conservatively get a $0 projected upside (rather
    # than a fabricated negative or an overstated abs() magnitude) until a
    # larger sample produces a reliable positive estimate. The raw
    # (possibly negative) slope is still reported as-is in the slopes table.
    # BUG: clip(lower=0) floors negative slopes but still allows MOS
    # (slope=+0.185, p=0.929) to produce non-zero deltas. Fix: also floor
    # any slope that is not statistically significant (p >= 0.05).
    P_THRESHOLD = 0.05
    liabilities['slope_for_replacement'] = liabilities.apply(
        lambda r: float(r['slope']) if (float(r['slope']) > 0 and float(r['p_value']) < P_THRESHOLD) else 0.0,
        axis=1
    )
    liabilities['replacement_delta_win_pct'] = liabilities['slope_for_replacement'] * liabilities['projected_quality_shift']

    cols = ['season', 'team', 'player', 'archetype', 'archetype_score', 'league_median_score',
            'score_gap', 'weight_share', 'projected_quality_shift', 'slope', 'p_value', 'replacement_delta_win_pct']
    return liabilities[cols].sort_values('replacement_delta_win_pct', ascending=False).reset_index(drop=True)
