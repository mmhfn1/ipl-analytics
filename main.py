"""
main.py
========
End-to-end orchestrator for the IPL Performance Analytics Suite.

Runs all six modules in dependency order against a ball-by-ball CSV and
writes a sample CSV output for each major result table to ./output/.

Usage
-----
    python main.py path/to/CRICKET_DATASET.csv [--season-start 2008] [--season-end 2025] [--outdir output]

Pipeline order (later modules depend on earlier ones' outputs):

    1. data_loader        -> cleaned ball-by-ball df, match_summary
    2. leverage_index     -> per-ball Leverage Index
    3. archetypes          -> player-season archetype classification
    4. strategy_mapping    -> team archetype mix, win% correlation
    5. behavioral_analysis -> Stable / Adaptable players who changed teams
    6. liability            -> underperformers + Replacement Delta
    7. tenure_ranking       -> tenure, tenure density vs strategy consistency,
                                Top-5 ranking streaks vs player performance
"""

import argparse
import os

import pandas as pd

from ipl_analytics import (
    data_loader,
    leverage_index,
    archetypes,
    strategy_mapping,
    behavioral_analysis,
    liability,
    tenure_ranking,
)


def run_pipeline(csv_path: str, season_range: tuple[int, int] = (2008, 2025),
                  outdir: str = "output", verbose: bool = True) -> dict:
    os.makedirs(outdir, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg)

    def save(df: pd.DataFrame, name: str):
        path = os.path.join(outdir, f"{name}.csv")
        df.to_csv(path, index=False)
        log(f"  -> saved {path}  ({len(df)} rows)")

    results = {}

    # ---- 1. Load & clean ----
    log("[1/7] Loading and cleaning data...")
    df = data_loader.load_and_clean(csv_path, season_range=season_range)
    match_summary_raw = data_loader.build_match_summary(df)
    log(f"  Cleaned ball-by-ball rows: {len(df)} | Matches: {len(match_summary_raw)}")
    results['cleaned_balls'] = df
    results['match_summary_raw'] = match_summary_raw

    # ---- 2. Leverage Index ----
    log("[2/7] Computing Leverage Index (with DLS detection)...")
    match_summary = leverage_index.detect_dls_matches(match_summary_raw)
    n_dls = int(match_summary['dls_flag'].sum())
    log(f"  Flagged {n_dls} DLS-affected matches")
    df_with_li = leverage_index.compute_leverage_index(df, match_summary)
    log(f"  Mean Leverage Index: {df_with_li['leverage_index'].mean():.4f}")
    results['match_summary'] = match_summary
    results['leverage_index'] = df_with_li
    save(df_with_li[['match_id', 'season', 'innings', 'over', 'ball', 'batting_team',
                      'bowling_team', 'batter', 'bowler', 'phase', 'wp_before', 'wp_after',
                      'leverage_index', 'dls_flag']], "leverage_index")
    top_li = df_with_li.sort_values('leverage_index', ascending=False).head(25)
    save(top_li[['match_id', 'season', 'innings', 'over', 'ball', 'batter', 'bowler',
                  'runs_total', 'leverage_index']], "top25_highest_leverage_balls")

    # ---- 3. Archetype Classification ----
    log("[3/7] Classifying player archetypes...")
    arch_df = archetypes.classify_archetypes(df)
    n_low_sample = int(arch_df['is_low_sample'].sum())
    log(f"  Classified {len(arch_df)} player-seasons "
        f"({(arch_df['discipline'] == 'Batter').sum()} batters, "
        f"{(arch_df['discipline'] == 'Bowler').sum()} bowlers, "
        f"{n_low_sample} flagged low-sample)")
    results['archetypes'] = arch_df
    save(arch_df, "archetypes")

    # ---- 4. Strategic Philosophy Mapping ----
    log("[4/7] Mapping team strategic philosophies...")
    appearances = strategy_mapping.build_player_match_team(df)
    player_season_matches = strategy_mapping.compute_player_season_matches(appearances)
    archetype_mix = strategy_mapping.compute_archetype_mix(player_season_matches, arch_df)
    low_sample_share = strategy_mapping.compute_low_sample_share(player_season_matches, arch_df)
    win_pct = strategy_mapping.compute_team_win_pct(match_summary_raw)
    corr_summary, mix_vs_wins = strategy_mapping.correlate_mix_with_wins(archetype_mix, win_pct)
    low_sample_corr, low_sample_vs_wins = strategy_mapping.correlate_mix_with_wins(
        low_sample_share.rename(columns={'low_sample_pct': 'Low-Sample Share'}), win_pct
    )
    log("  Archetype mix vs win% correlation:")
    for _, r in corr_summary.iterrows():
        log(f"    {r['archetype']:<22} r={r['pearson_r']:+.3f}  p={r['p_value']:.3f}")
    log(f"  Low-Sample Share vs win%: r={low_sample_corr.iloc[0]['pearson_r']:+.3f}  p={low_sample_corr.iloc[0]['p_value']:.3f}")
    results['player_season_matches'] = player_season_matches
    results['archetype_mix'] = archetype_mix
    results['low_sample_share'] = low_sample_share
    results['win_pct'] = win_pct
    results['mix_win_correlation'] = corr_summary
    results['low_sample_correlation'] = low_sample_corr
    save(archetype_mix, "team_archetype_mix")
    save(corr_summary, "archetype_mix_win_correlation")
    save(low_sample_share.merge(win_pct, on=['season', 'team']), "low_sample_share_vs_win_pct")
    save(low_sample_corr, "low_sample_share_win_correlation")

    # ---- 5. Behavioral Analysis ----
    log("[5/7] Analyzing player behavioral stability across team changes...")
    primary_team = behavioral_analysis.compute_player_primary_team(player_season_matches)
    team_changes = behavioral_analysis.detect_team_changes(primary_team)
    behavioral_df = behavioral_analysis.analyze_behavioral_stability(team_changes, arch_df, primary_team)
    behavior_summary = behavioral_analysis.summarize_behavioral_patterns(behavioral_df)
    log(f"  {len(team_changes)} team changes detected, {len(behavioral_df)} same-archetype comparisons")
    if not behavioral_df.empty:
        log(f"  Stable: {(behavioral_df['classification'] == 'Stable').sum()}  "
            f"Adaptable: {(behavioral_df['classification'] == 'Adaptable').sum()}")
    results['primary_team'] = primary_team
    results['behavioral_analysis'] = behavioral_df
    save(behavioral_df, "behavioral_analysis")
    save(behavior_summary, "behavioral_summary_by_archetype")

    # ---- 6. Liability Identification ----
    log("[6/7] Identifying liabilities and projecting Replacement Delta...")
    underperformers = liability.flag_underperformers(arch_df)
    team_quality = liability.compute_team_archetype_quality(player_season_matches, arch_df)
    slopes = liability.calibrate_archetype_win_impact(team_quality, win_pct)
    replacement_delta = liability.compute_replacement_delta(underperformers, team_quality, slopes, primary_team)
    log(f"  Flagged {int(underperformers['is_underperformer'].sum())} underperforming player-seasons")
    log("  Calibrated win%-impact slopes per archetype:")
    for _, r in slopes.iterrows():
        log(f"    {r['archetype']:<22} slope={r['slope']:+.3f}  (n={r['n_team_seasons']})")
    results['underperformers'] = underperformers
    results['archetype_win_slopes'] = slopes
    results['replacement_delta'] = replacement_delta
    save(replacement_delta, "replacement_delta")
    save(slopes, "archetype_win_impact_slopes")

    # ---- 7. Tenure & Ranking Impact ----
    log("[7/7] Computing tenure, tenure density vs strategy consistency, and Top-5 streaks...")
    tenure_df = tenure_ranking.compute_tenure(primary_team)
    density = tenure_ranking.compute_team_tenure_density(tenure_df, player_season_matches)
    consistency = tenure_ranking.compute_strategy_consistency_by_transition(archetype_mix)
    tenure_corr_summary, tenure_vs_consistency = tenure_ranking.correlate_tenure_density_with_consistency(
        density, consistency
    )
    log(f"  Tenure density vs strategy-consistency: r={tenure_corr_summary['pearson_r']:+.3f} "
        f"p={tenure_corr_summary['p_value']:.4f} (n={tenure_corr_summary['n_observations']})")

    standings = tenure_ranking.build_progressive_standings(match_summary_raw)
    streaks = tenure_ranking.identify_top5_streaks(standings)
    batting_streak_compare, bowling_streak_compare = tenure_ranking.compare_player_performance_during_streaks(
        df, streaks
    )
    log(f"  Identified {len(streaks)} Top-5 ranking streaks across {season_range[0]}-{season_range[1]}")
    log(f"  Batting comparisons: {len(batting_streak_compare)}  Bowling comparisons: {len(bowling_streak_compare)}")

    results['tenure'] = tenure_df
    results['team_tenure_density'] = density
    results['strategy_consistency'] = consistency
    results['tenure_consistency_correlation'] = tenure_corr_summary
    results['standings'] = standings
    results['top5_streaks'] = streaks
    results['batting_streak_comparison'] = batting_streak_compare
    results['bowling_streak_comparison'] = bowling_streak_compare

    save(tenure_df, "player_tenure")
    save(density.merge(consistency, on=['season', 'team'], how='outer'), "team_tenure_density_vs_consistency")
    save(streaks.drop(columns='match_ids'), "top5_ranking_streaks")
    save(batting_streak_compare, "batting_performance_during_streaks")
    save(bowling_streak_compare, "bowling_performance_during_streaks")

    log("\nPipeline complete. Outputs written to: " + os.path.abspath(outdir))
    return results


def main():
    parser = argparse.ArgumentParser(description="IPL Performance Analytics Suite")
    parser.add_argument("csv_path", help="Path to the ball-by-ball IPL CSV file")
    parser.add_argument("--season-start", type=int, default=2008)
    parser.add_argument("--season-end", type=int, default=2025)
    parser.add_argument("--outdir", type=str, default="output")
    args = parser.parse_args()

    run_pipeline(
        args.csv_path,
        season_range=(args.season_start, args.season_end),
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
