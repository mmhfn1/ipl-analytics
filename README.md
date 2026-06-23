# IPL Performance Analytics Suite

**Six statistical models applied to 278,034 ball-by-ball deliveries from every complete IPL season on record (2008–2025).**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)
![Status](https://img.shields.io/badge/status-stable-brightgreen)

A Python package and accompanying dashboard for analyzing ball-by-ball
Indian Premier League data: a calibrated win-probability model, a
player-role classifier, and four regression-based studies of team
composition, player consistency after a trade, replacement value, and
roster stability.

Every number this package produces is computed from the data. Nothing in
the pipeline, the sample outputs, or the dashboard is hardcoded, and the
[Methodology](#methodology) and [Known Limitations](#known-limitations)
sections document where each model is an approximation rather than ground
truth, including a full record of corrections made during a data-quality
review (see [Revision History](#revision-history)).

---

## Table of Contents

- [The Six Modules](#the-six-modules)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dashboard](#dashboard)
- [Sample Outputs](#sample-outputs)
- [Methodology](#methodology)
- [Known Limitations](#known-limitations)
- [Revision History](#revision-history)
- [License](#license)

---

## The Six Modules

| # | Module | File | Question it answers |
|---|--------|------|---------|
| 01 | **Leverage Index** ("Pressure Gauge") | `leverage_index.py` | Which deliveries had the largest effect on win probability? |
| 02 | **Archetype Classification** ("Role Scanner") | `archetypes.py` | What is each player's role, based on performance rather than reputation? |
| 03 | **Strategic Philosophy Mapping** ("Blueprint Index") | `strategy_mapping.py` | Does a team's mix of roles predict its win percentage? |
| 04 | **Behavioral Analysis** ("Mindset Monitor") | `behavioral_analysis.py` | Does a player's performance in a role stay stable after changing teams? |
| 05 | **Liability & Replacement Value** ("Value Finder") | `liability.py` | What would replacing an underperforming player be worth, in win%? |
| 06 | **Tenure & Ranking Impact** ("Momentum Meter") | `tenure_ranking.py` | Is roster stability associated with sustained league position? |

All six modules share one data-cleaning foundation (`data_loader.py`) and
reuse each other's outputs — see [Methodology](#methodology) and
`main.py` for the exact dependency graph.

## Key Findings

Headline results from the reference run included in `sample_outputs/`
(2008–2025, full season range). Each is reported with its sample size and
statistical significance, not as a standalone claim — see the relevant
module section in `docs/MODULE_REFERENCE.md` for the full derivation.

- **The Powerplay overs carry the highest average Leverage Index** of the
  three innings phases (0.0261), compared with 0.0194 for the Death overs
  and 0.0120 for the Middle overs. The single highest-leverage delivery on
  record, however, came at the end of a chase: Rahul Tewatia's six off
  Odean Smith, Gujarat Titans vs Punjab Kings, April 2022.
- **Low-Sample Share — the share of a team's participation held by
  below-threshold players — is the strongest single predictor of win
  percentage** found across Models 02–03 (r = −0.362, p < 0.001, n = 156
  team-seasons), stronger than any individual archetype's mix percentage.
- **Anchor is the only archetype with a statistically reliable positive
  relationship to win percentage** at the team-composition level
  (r = +0.245, p = 0.002). Anchor is also the *least* stable archetype
  after a player changes teams (54.2% Stable vs. 66–71% for the other
  three roles), which is consistent with anchoring being shaped
  substantially by what a given lineup needs.
- **Only one of four archetypes' win-impact slopes survives statistical
  testing at full scale.** Powerplay Aggressor (p = 0.043) is the lone
  archetype with a reliable link between team quality and win%; Anchor,
  Death Specialist, and Middle-Over Squeezer are statistically
  indistinguishable from zero once pooled across 18 seasons
  (p = 0.91, 0.997, 0.93). Replacement Delta is floored at 0 for those
  three roles rather than reporting an unreliable figure.
- **A relationship reported in an earlier, smaller-sample version of this
  analysis did not hold up at full scale.** Roster tenure density vs.
  strategic consistency was reported at r = +0.54 on a six-season window;
  recomputed across the complete 18-season dataset, it is r = +0.003
  (p = 0.97, n = 139) — no longer distinguishable from no relationship.
  See [Revision History](#revision-history).

## Dataset

- **Source**: ball-by-ball delivery-level data, one row per ball.
- **Range used**: 2008–2025, all 18 complete IPL seasons. The source file
  also contains a partial 2026 season (49 of an expected ~74 matches,
  cut off before the playoffs); it is excluded by default because a
  partial season produces an incorrect "champion" and distorts every
  season-level statistic. Pass `--season-end 2026` to include it anyway.
- **Scale**: 278,034 deliveries, 1,169 matches, 15 franchise identities
  (10 currently active, 5 defunct), 767 distinct players, 3,137
  player-seasons.
- **Season labeling**: three raw season labels use a "yyyy/yy" format
  (e.g. `2007/08`, `2009/10`, `2020/21`). These are resolved to a single
  calendar year per label by looking up the actual match dates, not by
  parsing the label string — see [Revision History](#revision-history)
  point 4 for why a string-parsing rule cannot resolve all three
  correctly.
- **Franchise identity**: renamed franchises (e.g. Delhi Daredevils →
  Delhi Capitals) are merged to one canonical name so tenure and mix
  statistics aren't reset by a rebrand. Franchises that changed ownership
  entirely (e.g. Deccan Chargers → Sunrisers Hyderabad, Gujarat Lions →
  Gujarat Titans) are intentionally kept separate, since the playing
  squad and franchise identity are not continuous across that change.

## Installation

```bash
git clone <repo-url>
cd ipl-analytics-suite
pip install -r requirements.txt
```

Requires Python 3.10+. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`.

## Quick Start

```bash
python main.py path/to/your_dataset.csv --outdir output
```

This runs all seven pipeline stages (data cleaning → Leverage Index →
Archetype Classification → Strategic Mapping → Behavioral Analysis →
Liability Identification → Tenure & Ranking) and writes 14 CSV files to
`output/`. Default season range is 2008–2025; override with
`--season-start` / `--season-end`.

```python
from ipl_analytics import data_loader, leverage_index, archetypes

df = data_loader.load_and_clean("your_data.csv", season_range=(2008, 2025))
match_summary = leverage_index.detect_dls_matches(data_loader.build_match_summary(df))
df_with_li = leverage_index.compute_leverage_index(df, match_summary)

arch_df = archetypes.classify_archetypes(df)
```

See `main.py` for the full call sequence and how later stages consume
earlier modules' outputs.

## Project Structure

```
ipl-analytics-suite/
├── README.md
├── LICENSE
├── requirements.txt
├── main.py                      # End-to-end pipeline orchestrator
├── ipl_analytics/
│   ├── __init__.py
│   ├── data_loader.py           # Cleaning, normalization, phase tagging
│   ├── leverage_index.py        # Module 01 — Pressure Gauge
│   ├── archetypes.py            # Module 02 — Role Scanner
│   ├── strategy_mapping.py      # Module 03 — Blueprint Index
│   ├── behavioral_analysis.py   # Module 04 — Mindset Monitor
│   ├── liability.py             # Module 05 — Value Finder
│   └── tenure_ranking.py        # Module 06 — Momentum Meter
├── dashboard/
│   └── index.html               # Self-contained interactive dashboard
├── sample_outputs/               # Result tables from one reference run
│   └── *.csv  (15 files)
└── docs/
    └── MODULE_REFERENCE.md       # Full per-module methodology
```

## Dashboard

`dashboard/index.html` is a single self-contained HTML file (no build
step, no server, no external data files — open it directly in a browser)
that presents the results of a 2008–2025 reference run interactively:

- A team selector covering all 15 franchise identities, each with its own
  data view recomputed live for any season or the full multi-season
  record.
- A "longest-serving core" view per team: the five players with the most
  seasons at that franchise, with their archetype and key statistic
  (strike rate or economy) shown season by season, so a role change over
  a career is visible directly rather than asserted.
- The six module results described above, each with its underlying chart
  and a written explanation of the method and finding.
- A Method section listing the data-quality issues found and fixed during
  review (see [Revision History](#revision-history)).

The dashboard's data is generated by `dashboard/extract_data.py`, which
calls the same `ipl_analytics` package functions used by `main.py` and
serializes their output into the two JSON blocks inlined near the bottom
of `index.html`. It is a presentation layer over the pipeline output, not
a separate analysis — running it again after a fresh `main.py` run
regenerates the dashboard's numbers from scratch.

## Sample Outputs

`sample_outputs/` contains the result tables from one reference pipeline
run (2008–2025), generated entirely by `main.py`; none of these values
are hand-edited. `leverage_index_sample.csv` is a representative subset
(one full match plus a random sample of 2,000 deliveries) rather than the
full 278,034-row file, to keep the repository a reasonable size — running
`main.py` regenerates the complete file.

| File | Contents |
|------|----------|
| `leverage_index_sample.csv` | Representative sample of per-ball Leverage Index |
| `top25_highest_leverage_balls.csv` | The 25 highest-leverage deliveries league-wide |
| `archetypes.csv` | All 3,137 classified player-seasons, including `is_low_sample` flag |
| `team_archetype_mix.csv` | Per-team-season archetype mix percentages |
| `archetype_mix_win_correlation.csv` | Pearson r of mix % vs win % per archetype |
| `low_sample_share_vs_win_pct.csv` | Per-team-season Low-Sample Share vs win % |
| `low_sample_share_win_correlation.csv` | Pearson r of Low-Sample Share vs win % |
| `behavioral_analysis.csv` | Stable / Adaptable classification per team-change |
| `behavioral_summary_by_archetype.csv` | Aggregated stability rates by archetype |
| `replacement_delta.csv` | Flagged underperformers and projected win% impact |
| `archetype_win_impact_slopes.csv` | Calibrated OLS slope per archetype |
| `player_tenure.csv` | Consecutive-season tenure spells per player |
| `team_tenure_density_vs_consistency.csv` | Tenure density vs strategy-consistency, by team-season |
| `top5_ranking_streaks.csv` | All reconstructed Top-5 ranking streaks |
| `batting_performance_during_streaks.csv` | Strike rate, in-streak vs season |
| `bowling_performance_during_streaks.csv` | Economy, in-streak vs season |

## Methodology

Each module is built as a **calibrated heuristic rather than a
hand-picked formula** — model weights and conversion factors are learned
from the data itself (e.g. Leverage Index's logistic regression,
Replacement Delta's score-to-win% OLS calibration) rather than chosen
arbitrarily. Every methodology decision, formula, and validation check is
documented in full in **[`docs/MODULE_REFERENCE.md`](docs/MODULE_REFERENCE.md)**,
including:

- The exact feature set and shrinkage correction behind Leverage Index,
  and how its terminal win-probability states (target reached, all out,
  deliveries exhausted, rain-affected) are resolved against the actual
  match winner rather than inferred from score alone.
- The Z-score formulas behind each of the four player archetypes, and how
  the classification baseline is kept stable while still scoring every
  player-season, including low-sample ones.
- How the rain/DLS-affected-match heuristic is defined and validated.
- The OLS calibration behind Replacement Delta, including how a
  negative or statistically unreliable slope is handled.
- The standings-reconstruction algorithm behind Top-5 streaks.

## Known Limitations

This project is explicit about where each model is an approximation
rather than ground truth:

- **No official DLS resource table.** Rain-affected matches are detected
  via an innings-completion heuristic (an innings that ends without
  reaching the target or losing all 10 wickets must have been
  interrupted) and the affected team's actual win/loss is read from the
  match result rather than recomputed from a par score, since the
  official Duckworth-Lewis-Stern table is proprietary and not available
  in the source data.
- **Win probability is a calibrated heuristic, not a ball-tracking or
  player-quality-aware model.** It is fit on three engineered features
  (run-rate gap, wickets in hand, balls remaining) and validated to match
  the actual outcome at the final ball of all 635 decided matches
  checked, but it does not account for venue, opposition bowling
  strength, or pitch conditions.
- **Standings ignore Net Run Rate tiebreaks.** The source data has no NRR
  column; ties on points use standard competition ranking instead.
- **Three of the four Replacement Delta slopes are not statistically
  significant once pooled across all 18 seasons** (Anchor, Death
  Specialist, Middle-Over Squeezer; see Key Findings above) and are
  floored at 0 rather than reported as a precise figure.
- **The roster-tenure / strategic-consistency relationship reported in an
  earlier, smaller-sample version of this analysis does not hold at full
  scale** (see Revision History). This is reported as the correct result
  for the full dataset, not corrected toward the earlier figure.
- **Streak-vs-season performance deltas are conservative.** A player's
  "season average" includes their own streak matches, which biases the
  population-level mean toward zero.

Full detail on each of these is in `docs/MODULE_REFERENCE.md`.

## Revision History

This analysis went through a data-quality review after an initial
version was published on a restricted (2017–2022) season range. The
review was prompted by a question about whether that range was actually
the full extent of the available data — it was not (the source file
covers 2008–2026) — and surfaced eight additional issues, several of
which changed real numbers by a meaningful margin:

1. **Win-probability terminal states now defer to the actual match
   winner.** The most common chase result in T20 — a team falling short
   without being bowled out (e.g. 142/9 chasing 158) — was previously
   left to the regression instead of being forced to a clean loss,
   because the required-run-rate formula collapses to 0 once no balls
   remain. This was producing a win probability around 85–95% on the
   final ball of 127 of 373 matches checked in an earlier run. Final-ball
   win probability now matches the real outcome in all 635 decided
   matches checked in the full dataset.
2. **Rain/DLS detection now uses the correct threshold.** The
   interruption check previously required fewer than 19 overs bowled,
   which missed a match shortened to exactly 19 overs. The correct test
   is "didn't complete all 120 balls, wasn't all out, didn't reach the
   target" — there is no other legitimate reason that combination occurs.
3. **Tied matches now resolve to their real Super Over winner.** 16 of
   1,169 matches have a raw `winner` value of literally `"tie"` or
   `"no result"`, not a team name. Ties are resolved by reading the
   Super Over innings (including the rare case of a second Super Over
   when the first also tied) before that data is filtered out of the
   main ball-by-ball analysis. Genuine no-results are excluded from
   win% and model-training data, and correctly awarded 1 point each in
   the standings reconstruction rather than 0–0.
4. **Season labels are now derived from match dates, not parsed from the
   label string.** Three season labels use a "yyyy/yy" format, and no
   single string-parsing rule resolves all of them correctly:
   `"2007/08"` is IPL Season 1, played in 2008, while `"2020/21"` is the
   pandemic-displaced season, played entirely in 2020 — taking "the
   first year" is correct for one and wrong for the other. Every label's
   true calendar year is now looked up from its matches' actual dates.
5. **A no-ball is now correctly counted as a ball faced** for strike-rate
   purposes. It was previously excluded along with wides, even though a
   no-ball (and its free hit) is faced by the batter and frequently hit
   for runs.
6. **Archetype classification no longer has an "Unclassified" bucket.**
   The Z-score baseline for each metric is computed only from the
   stable, ≥60-ball population so small-sample outliers can't distort
   it, but every player-season is then scored against that fixed
   baseline and assigned a role, flagged `is_low_sample` rather than
   dropped.
7. **Replacement Delta can no longer go negative for a flagged
   underperformer.** When an archetype's league-wide win-impact slope is
   negative or statistically unreliable, the calculation now floors it
   at 0 rather than implying that upgrading a worse player would hurt
   the team.
8. **A previously reported correlation does not hold at full scale.**
   Roster tenure density vs. strategic consistency was r = +0.54 on a
   six-season window; recomputed on the full 18-season dataset it is
   r = +0.003 (p = 0.97). The underlying data was checked for missing
   values, duplicate rows, and outliers — none were found — so this is
   reported as the correct result rather than reverted.

## License

Released under the [MIT License](LICENSE).
