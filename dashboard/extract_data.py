# Regenerates the data inlined into index.html from the ipl_analytics
# pipeline output. Run main.py first (writes to ../output), then run this
# script from inside this dashboard/ directory. index.html does not read
# these JSON files at runtime -- they are written for transparency and as
# an intermediate step; the dashboard itself is fully self-contained.
import sys, json
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
from ipl_analytics import data_loader, leverage_index, archetypes as arch_mod, strategy_mapping, \
    behavioral_analysis, liability, tenure_ranking

OUT = '../output'
SEASONS = list(range(2008, 2026))

TEAM_META = {
    'Chennai Super Kings':         {'abbr': 'CSK', 'color': 'F9CD05', 'colorDark': '6B5A00'},
    'Mumbai Indians':              {'abbr': 'MI',  'color': '045093', 'colorDark': '0A2A4D'},
    'Royal Challengers Bengaluru': {'abbr': 'RCB', 'color': 'EC1C24', 'colorDark': '6B0A0D'},
    'Kolkata Knight Riders':       {'abbr': 'KKR', 'color': '8B5FBF', 'colorDark': '2E1A47'},
    'Delhi Capitals':              {'abbr': 'DC',  'color': '17479E', 'colorDark': '0E2A5E'},
    'Punjab Kings':                {'abbr': 'PBKS','color': 'ED1B24', 'colorDark': '6B0A0D'},
    'Rajasthan Royals':            {'abbr': 'RR',  'color': 'EA1A85', 'colorDark': '5A0A35'},
    'Sunrisers Hyderabad':         {'abbr': 'SRH', 'color': 'FF822A', 'colorDark': '7A3D00'},
    'Gujarat Titans':              {'abbr': 'GT',  'color': 'B3A123', 'colorDark': '1B2133'},
    'Lucknow Super Giants':        {'abbr': 'LSG', 'color': '00ADEF', 'colorDark': '0E3A52'},
    'Deccan Chargers':             {'abbr': 'DCH', 'color': '041C32', 'colorDark': '000000'},
    'Gujarat Lions':               {'abbr': 'GL',  'color': 'E1471C', 'colorDark': '3D1A0E'},
    'Kochi Tuskers Kerala':        {'abbr': 'KTK', 'color': 'F5821F', 'colorDark': '3D2200'},
    'Pune Warriors':               {'abbr': 'PWI', 'color': '8FBF42', 'colorDark': '2B330E'},
    'Rising Pune Supergiant':      {'abbr': 'RPS', 'color': '651A1F', 'colorDark': '2B2A29'},
}

print("Loading pipeline outputs (2008-2025, full corrected dataset)...")
df = data_loader.load_and_clean('../CRICKET_DATASET.csv', season_range=(2008, 2025))
match_summary_raw = data_loader.build_match_summary(df)
match_summary = leverage_index.detect_dls_matches(match_summary_raw)
li = pd.read_csv(f'{OUT}/leverage_index.csv')
arch = pd.read_csv(f'{OUT}/archetypes.csv')
mix = pd.read_csv(f'{OUT}/team_archetype_mix.csv')
mix_corr = pd.read_csv(f'{OUT}/archetype_mix_win_correlation.csv')
low_sample_vs_win = pd.read_csv(f'{OUT}/low_sample_share_vs_win_pct.csv')
low_sample_corr = pd.read_csv(f'{OUT}/low_sample_share_win_correlation.csv')
repl = pd.read_csv(f'{OUT}/replacement_delta.csv')
slopes = pd.read_csv(f'{OUT}/archetype_win_impact_slopes.csv')
streaks = pd.read_csv(f'{OUT}/top5_ranking_streaks.csv')
behavior_summary = pd.read_csv(f'{OUT}/behavioral_summary_by_archetype.csv')
behavior_df = pd.read_csv(f'{OUT}/behavioral_analysis.csv')

appearances = strategy_mapping.build_player_match_team(df)
player_season_matches = strategy_mapping.compute_player_season_matches(appearances)
win_pct = strategy_mapping.compute_team_win_pct(match_summary_raw)
primary_team = behavioral_analysis.compute_player_primary_team(player_season_matches)
tenure_df = tenure_ranking.compute_tenure(primary_team)
team_tenure_density = tenure_ranking.compute_team_tenure_density(tenure_df, player_season_matches)
strategy_consistency = tenure_ranking.compute_strategy_consistency_by_transition(mix)
tenure_corr_summary, _ = tenure_ranking.correlate_tenure_density_with_consistency(team_tenure_density, strategy_consistency)

print("Rows:", len(df), "Matches:", len(match_summary_raw), "Player-seasons:", len(arch))
print("Tenure-consistency correlation:", tenure_corr_summary)

ARCH_ORDER = ['Powerplay Aggressor', 'Anchor', 'Middle-Over Squeezer', 'Death Specialist']

champions = {}
for season in SEASONS:
    sub = match_summary_raw[match_summary_raw['season'] == season].sort_values('date')
    champions[season] = sub.iloc[-1]['winner']

def phase_avg(frame):
    return frame.groupby('phase')['leverage_index'].mean().to_dict()

leverage_alltime = phase_avg(li)
leverage_by_season = {s: phase_avg(li[li['season'] == s]) for s in SEASONS}

top_moments = li.sort_values('leverage_index', ascending=False).head(15)
top_moments = top_moments.merge(match_summary_raw[['match_id', 'team1', 'team2', 'date']], on='match_id', how='left')
top_moments = top_moments[['match_id', 'season', 'over', 'ball', 'batting_team', 'bowling_team', 'leverage_index', 'date', 'team1', 'team2']]

arch_counts_alltime = arch['archetype'].value_counts().to_dict()
arch_counts_by_season = {s: arch[arch['season'] == s]['archetype'].value_counts().to_dict() for s in SEASONS}
low_sample_rate_alltime = round(100 * arch['is_low_sample'].mean(), 1)
low_sample_rate_by_season = {s: round(100 * arch[arch['season'] == s]['is_low_sample'].mean(), 1) for s in SEASONS if len(arch[arch['season']==s])}

VALIDATION_PLAYERS = ['V Kohli', 'RG Sharma', 'JJ Bumrah', 'MS Dhoni', 'YS Chahal', 'SP Narine']
validation_rows = []
for p in VALIDATION_PLAYERS:
    row = arch[(arch['season'] == 2025) & (arch['player'] == p)]
    if len(row):
        validation_rows.append({'player': p, 'archetype': row.iloc[0]['archetype']})

blueprint_corr = mix_corr.to_dict('records')
low_sample_corr_row = low_sample_corr.iloc[0].to_dict()
mindset_alltime = behavior_summary.to_dict('records')

mindset_by_season = {}
for s in SEASONS:
    sub = behavior_df[behavior_df['season_after'] == s]
    if len(sub) == 0: continue
    rows = []
    for archetype, g in sub.groupby('archetype'):
        rows.append({'archetype': archetype, 'n_moves': len(g),
                      'n_stable': int((g['classification'] == 'Stable').sum()),
                      'n_adaptable': int((g['classification'] == 'Adaptable').sum())})
    mindset_by_season[s] = rows

value_slopes = slopes.to_dict('records')
# PA (reliable delta) first, then others sorted by stat deficit
pa_top    = repl[repl['archetype']=='Powerplay Aggressor'].sort_values('replacement_delta_win_pct', ascending=False).head(5)
other_top = repl[repl['archetype']!='Powerplay Aggressor'].sort_values('key_stat_deficit', ascending=False, key=lambda x: x.fillna(-999)).head(5)
top_repl_alltime = pd.concat([pa_top, other_top]).head(10).to_dict('records')
top_repl_by_season = {s: repl[repl['season'] == s].sort_values('replacement_delta_win_pct', ascending=False).head(5).to_dict('records') for s in SEASONS}

def streak_record(row):
    return {'season': int(row['season']), 'team': row['team'], 'length': int(row['streak_length']),
            'start': str(row['streak_start_date']), 'end': str(row['streak_end_date'])}

streaks_alltime = [streak_record(r) for _, r in streaks.sort_values('streak_length', ascending=False).head(10).iterrows()]
streaks_by_season = {s: [streak_record(r) for _, r in streaks[streaks['season'] == s].sort_values('streak_length', ascending=False).iterrows()] for s in SEASONS}

def get_timeline(season, team, max_matches=18):
    standings = tenure_ranking.build_progressive_standings(match_summary_raw[match_summary_raw['season']==season])
    sub = standings[standings['team']==team].sort_values('date')
    return sub['rank_entering_match'].tolist()[:max_matches]

timelines = {
    'MI 2020 (Champions)': get_timeline(2020, 'Mumbai Indians'),
    'GT 2022 (Debut Champions)': get_timeline(2022, 'Gujarat Titans'),
    'RCB 2025 (Champions)': get_timeline(2025, 'Royal Challengers Bengaluru'),
}

print("League-wide aggregates assembled. Champions:", champions)

# ===========================================================================
# LONGEST-SERVING MEMBERS PER TEAM + ROLE EVOLUTION
# ===========================================================================
def compute_longest_serving(team):
    """Top-5 players by distinct seasons served as this team's PRIMARY
    team, with their archetype + key stat evolution season by season."""
    team_seasons = primary_team[primary_team['team'] == team]
    season_counts = team_seasons.groupby('player')['season'].nunique().sort_values(ascending=False)
    top5 = season_counts.head(5)

    result = []
    for player, n_seasons in top5.items():
        player_seasons = sorted(team_seasons[team_seasons['player'] == player]['season'].tolist())
        timeline = []
        for s in player_seasons:
            arow = arch[(arch['player'] == player) & (arch['season'] == s)]
            if len(arow):
                r = arow.iloc[0]
                if r['discipline'] == 'Batter':
                    stat_val, stat_label = r['overall_strike_rate'], 'SR'
                else:
                    stat_val, stat_label = r['overall_economy'], 'Econ'
                timeline.append({
                    'season': int(s), 'archetype': r['archetype'], 'discipline': r['discipline'],
                    'statLabel': stat_label,
                    'statValue': round(float(stat_val), 1) if pd.notna(stat_val) else None,
                    'isLowSample': bool(r['is_low_sample']),
                })
        if timeline:
            archetypes_seen = sorted(set(t['archetype'] for t in timeline))
            result.append({
                'player': player, 'seasonsServed': int(n_seasons), 'timeline': timeline,
                'roleChanged': len(archetypes_seen) > 1,
                'archetypesSeen': archetypes_seen,
            })
    return result

print("Computing longest-serving cores for each team...")
longest_serving_by_team = {}
all_teams = sorted(df['batting_team'].unique())
for team in all_teams:
    longest_serving_by_team[team] = compute_longest_serving(team)
    n_changed = sum(1 for p in longest_serving_by_team[team] if p['roleChanged'])
    print(f"  {team}: {len(longest_serving_by_team[team])} long-servers, {n_changed} changed role at least once")

print()
print("Sample (Mumbai Indians):")
print(json.dumps(longest_serving_by_team.get('Mumbai Indians', []), indent=2)[:1500])

# ===========================================================================
# PER-TEAM DATA (per-season + all-time aggregate)
# ===========================================================================
teams_data = {}

for team in all_teams:
    meta = TEAM_META.get(team, {'abbr': team[:3].upper(), 'color': '8993B4', 'colorDark': '1B2350'})
    team_matches = match_summary_raw[(match_summary_raw['team1'] == team) | (match_summary_raw['team2'] == team)]
    seasons_active = sorted(team_matches['season'].unique().tolist())

    def season_slice(season_filter):
        if season_filter is None:
            team_mix_rows = mix[mix['team'] == team]
            team_repl = repl[repl['team'] == team]
            team_streaks = streaks[streaks['team'] == team]
            li_bat = li[li['batting_team'] == team]
            li_bowl = li[li['bowling_team'] == team]
            wp = win_pct[win_pct['team'] == team]
        else:
            team_mix_rows = mix[(mix['team'] == team) & (mix['season'] == season_filter)]
            team_repl = repl[(repl['team'] == team) & (repl['season'] == season_filter)]
            team_streaks = streaks[(streaks['team'] == team) & (streaks['season'] == season_filter)]
            li_bat = li[(li['batting_team'] == team) & (li['season'] == season_filter)]
            li_bowl = li[(li['bowling_team'] == team) & (li['season'] == season_filter)]
            wp = win_pct[(win_pct['team'] == team) & (win_pct['season'] == season_filter)]

        total_matches = int(wp['matches_played'].sum()) if len(wp) else 0
        total_wins = int(wp['matches_won'].sum()) if len(wp) else 0
        win_pct_val = round(100 * total_wins / total_matches, 1) if total_matches else 0.0

        if season_filter is None:
            titles = [int(s) for s, w in champions.items() if w == team]
        else:
            titles = [season_filter] if champions.get(season_filter) == team else []

        if len(team_mix_rows):
            mix_avg = {a: round(float(team_mix_rows[a].mean()), 1) if a in team_mix_rows.columns else 0.0 for a in ARCH_ORDER}
        else:
            mix_avg = None

        mix_by_season = []
        for s in seasons_active:
            r = mix[(mix['team'] == team) & (mix['season'] == s)]
            if len(r):
                r = r.iloc[0]
                mix_by_season.append({'season': int(s), 'mix': {a: round(float(r.get(a, 0.0)), 1) for a in ARCH_ORDER}})

        team_tenure_rows = team_tenure_density[team_tenure_density['team'] == team].sort_values('season')
        tenure_trend = [{'season': int(r['season']), 'density': round(float(r['team_tenure_density']), 2)} for _, r in team_tenure_rows.iterrows()]
        if season_filter is None:
            latest_tenure = tenure_trend[-1]['density'] if tenure_trend else None
        else:
            match_row = [t['density'] for t in tenure_trend if t['season'] == season_filter]
            latest_tenure = match_row[0] if match_row else None

        # PA (with reliable delta) first, then others sorted by stat deficit size
        pa_rows    = team_repl[team_repl['archetype'] == 'Powerplay Aggressor'].sort_values('replacement_delta_win_pct', ascending=False)
        other_rows = team_repl[team_repl['archetype'] != 'Powerplay Aggressor'].sort_values('key_stat_deficit', ascending=False, key=lambda x: x.fillna(-999))
        top5       = pd.concat([pa_rows, other_rows]).head(5)
        top_finds  = []
        for _, r in top5.iterrows():
            delta   = float(r['replacement_delta_win_pct'])
            deficit = r.get('key_stat_deficit', None)
            actual  = r.get('key_stat_actual', None)
            median  = r.get('key_stat_median', None)
            label   = r.get('key_stat', None)
            top_finds.append({
                'season':        int(r['season']),
                'player':        r['player'],
                'archetype':     r['archetype'],
                'delta':         round(delta, 2),
                'hasDelta':      delta > 0,
                'keyStatLabel':  str(label)   if label   is not None and str(label)   != 'nan' else None,
                'keyStatActual': round(float(actual), 1) if actual is not None and str(actual) != 'nan' else None,
                'keyStatMedian': round(float(median), 1) if median is not None and str(median) != 'nan' else None,
                'keyStatDeficit':round(float(deficit),1) if deficit is not None and str(deficit) != 'nan' else None,
            })

        best_streak = None
        if len(team_streaks):
            r = team_streaks.sort_values('streak_length', ascending=False).iloc[0]
            best_streak = {'season': int(r['season']), 'length': int(r['streak_length']), 'start': str(r['streak_start_date']), 'end': str(r['streak_end_date'])}

        avg_li_batting = float(li_bat['leverage_index'].mean()) if len(li_bat) else None
        avg_li_bowling = float(li_bowl['leverage_index'].mean()) if len(li_bowl) else None

        if season_filter is None:
            ls_rows = low_sample_vs_win[low_sample_vs_win['team'] == team]
        else:
            ls_rows = low_sample_vs_win[(low_sample_vs_win['team'] == team) & (low_sample_vs_win['season'] == season_filter)]
        low_sample_pct = round(float(ls_rows['low_sample_pct'].mean()), 1) if len(ls_rows) else None

        return {
            'totalMatches': total_matches, 'totalWins': total_wins, 'winPct': win_pct_val,
            'titles': titles, 'latestMix': mix_avg, 'mixBySeason': mix_by_season,
            'latestTenureDensity': latest_tenure, 'tenureTrend': tenure_trend,
            'topFinds': top_finds, 'bestStreak': best_streak, 'nStreaks': len(team_streaks),
            'avgLeverageBatting': round(avg_li_batting, 4) if avg_li_batting else None,
            'avgLeverageBowling': round(avg_li_bowling, 4) if avg_li_bowling else None,
            'lowSamplePct': low_sample_pct,
        }

    by_season = {'all': season_slice(None)}
    for s in seasons_active:
        by_season[str(s)] = season_slice(s)

    teams_data[team] = {
        'name': team, 'abbr': meta['abbr'], 'color': meta['color'], 'colorDark': meta['colorDark'],
        'seasonsActive': seasons_active, 'bySeason': by_season,
        'longestServing': longest_serving_by_team.get(team, []),
    }

hero_match = li[(li['match_id'] == 1304062) & (li['innings'] == 2)].sort_values(['over', 'ball'])
hero_sequence = hero_match['leverage_index'].round(4).tolist()

league_data = {
    'overview': {
        'seasons': SEASONS, 'totalMatches': int(len(match_summary_raw)), 'totalDeliveries': int(len(df)),
        'totalPlayers': int(arch['player'].nunique()), 'totalFranchises': len(all_teams),
        'champions': {str(k): v for k, v in champions.items()},
    },
    'leverage': {
        'meanAllTime': round(float(li['leverage_index'].mean()), 4),
        'phaseAvgAllTime': {k: round(v, 4) for k, v in leverage_alltime.items()},
        'phaseAvgBySeason': {str(s): {k: round(v, 4) for k, v in d.items()} for s, d in leverage_by_season.items()},
        'topMoments': json.loads(top_moments.head(10).to_json(orient='records')),
        'dlsFlaggedCount': int(match_summary['dls_flag'].sum()),
    },
    'archetypes': {
        'countsAllTime': {k: int(v) for k, v in arch_counts_alltime.items()},
        'countsBySeason': {str(s): {k: int(v) for k, v in d.items()} for s, d in arch_counts_by_season.items()},
        'lowSampleRateAllTime': low_sample_rate_alltime,
        'lowSampleRateBySeason': {str(s): v for s, v in low_sample_rate_by_season.items()},
        'totalPlayerSeasons': int(len(arch)), 'validation2025': validation_rows,
    },
    'strategy': {'blueprintCorrelation': blueprint_corr, 'lowSampleShareCorrelation': low_sample_corr_row, 'nTeamSeasons': int(len(mix))},
    'behavioral': {
        'summaryAllTime': mindset_alltime, 'bySeason': {str(s): v for s, v in mindset_by_season.items()},
        'totalChanges': int(len(behavioral_analysis.detect_team_changes(primary_team))), 'totalCompared': int(len(behavior_df)),
    },
    'liability': {
        'slopes': value_slopes,
        'topReplacementsAllTime': json.loads(pd.DataFrame(top_repl_alltime).to_json(orient='records')),
        'topReplacementsBySeason': {str(s): json.loads(pd.DataFrame(v).to_json(orient='records')) if len(v) else [] for s, v in top_repl_by_season.items()},
        'nUnderperformers': int(len(repl)),
    },
    'tenure': {'consistencyCorrelation': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tenure_corr_summary.items()}},
    'streaks': {
        'allTime': streaks_alltime, 'bySeason': {str(s): v for s, v in streaks_by_season.items()},
        'totalStreaks': int(len(streaks)), 'timelines': timelines,
    },
}

with open('./league_data.json', 'w') as f:
    json.dump(league_data, f, default=str)
with open('./team_data.json', 'w') as f:
    json.dump(teams_data, f, default=str)
with open('./hero_sequence.json', 'w') as f:
    json.dump(hero_sequence, f)

import os
print("\nFinal saved files:")
for fn in ['league_data.json', 'team_data.json', 'hero_sequence.json']:
    print(' ', fn, os.path.getsize(f'./{fn}'), 'bytes')
