# Module Reference

Full methodology, formulas, and validation notes for each of the six
modules in the IPL Performance Analytics Suite, current as of the
2008–2025 data-quality review (see the main README's
[Revision History](../README.md#revision-history) for a summary of what
changed and why). For a high-level overview, installation, and quick
start, see the [main README](../README.md).

## Table of Contents

- [Data Loading & Normalization](#data-loading--normalization-data_loaderpy)
- [Module 01 — Leverage Index](#module-01--leverage-index-leverage_indexpy)
- [Module 02 — Archetype Classification](#module-02--archetype-classification-archetypespy)
- [Module 03 — Strategic Philosophy Mapping](#module-03--strategic-philosophy-mapping-strategy_mappingpy)
- [Module 04 — Behavioral Analysis](#module-04--behavioral-analysis-behavioral_analysispy)
- [Module 05 — Liability Identification](#module-05--liability-identification-liabilitypy)
- [Module 06 — Tenure & Ranking Impact](#module-06--tenure--ranking-impact-tenure_rankingpy)

---

## Data Loading & Normalization (`data_loader.py`)

- **Season labels are derived from match dates, not parsed from the label
  string.** The source data contains 19 raw season labels; three use a
  "yyyy/yy" format (`2007/08`, `2009/10`, `2020/21`). A naive rule like
  "take the first year" gets two of these three wrong: `2007/08` is IPL
  Season 1, played entirely in April–June 2008, and `2009/10` is IPL
  Season 3, played entirely in March–April 2010 (there is a separate,
  genuine `2009` label for the actual 2009 season, played in South Africa).
  Only `2020/21` happens to have its true matches in the first slash-year.
  `normalize_season(df)` resolves this correctly and robustly by grouping
  on the raw label and taking the modal calendar year of that label's
  actual `date` values — every one of the 19 labels maps unambiguously to
  a single calendar year this way, with no special-casing required.
- **Franchise renames are normalized** so a club counts as one continuous
  entity across name changes: Delhi Daredevils → Delhi Capitals, Kings XI
  Punjab → Punjab Kings, Royal Challengers Bangalore → Royal Challengers
  Bengaluru, and a `Rising Pune Supergiant(s)` spelling-variant fix.
- **Franchises that changed ownership entirely are deliberately NOT
  merged**: Deccan Chargers (2008–2012) and Sunrisers Hyderabad
  (2013–present) are different organizations awarded the same city's
  franchise slot after Deccan Chargers' license was terminated; the same
  applies to Gujarat Lions (2016–2017, a temporary replacement franchise)
  and Gujarat Titans (2022–present). Treating these as one continuous
  entity would incorrectly extend tenure and roster-history statistics
  across a discontinuity in playing squad and ownership.
- `over`/`ball` raw fields don't reliably represent legal deliveries
  (wides and no-balls inflate the `ball` counter without using up an
  over). A derived `legal_ball` flag (excludes both wides and no-balls,
  for over-counting) and a separate `ball_faced_by_batter` flag (excludes
  only wides, since a no-ball is faced by the batter and can be hit for
  runs or a free-hit six) are used everywhere the corresponding
  calculation applies — strike rate uses `ball_faced_by_batter`; economy
  and over-counting use `legal_ball`.
- **Tied and no-result matches are resolved against ground truth.** 16 of
  1,169 matches have a raw `winner` value of `"tie"` or `"no result"`
  rather than a team name. `_resolve_true_winner()` runs on the raw,
  unfiltered data (before Super Over rows are dropped from the main
  ball-by-ball analysis) and recovers the real winner of a tie by
  comparing Super Over runs scored (including the rare case of a second
  Super Over when the first also tied). Genuine no-results (rain
  abandonments with no result possible) are left as `NaN` and excluded
  from win% and model-training calculations downstream.
- Player names were checked for near-duplicates (whitespace variants,
  fuzzy string matching) and came back clean for this dataset;
  `PLAYER_ALIAS_MAP` is provided as an extension point if a future
  dataset needs explicit aliasing.

## Module 01 — Leverage Index (`leverage_index.py`)

**Leverage Index (LI)** = absolute change in Win Probability (WP) caused
by a single delivery:

```
LI = |WP_after - WP_before|
```

Since the dataset has no true ball-by-ball WP labels, WP is estimated
with a **calibrated heuristic**: a `LogisticRegression` trained on real
2nd-innings (chase) ball states, using three engineered features —
`rate_gap` (required run rate minus current run rate), `wickets_in_hand_frac`,
and `balls_remaining_frac` — against the actual match outcome. This keeps
the model's weights *learned from data* rather than hand-picked. The same
fitted model is reused for 1st-innings "competitiveness" scoring (using
the season's par run-rate in place of a fixed target).

**Terminal-state handling.** A chase can end four ways: the target is
reached (WP = 1), the team is bowled out short of the target (WP = 0),
the deliveries run out without either of the above — the most common
result in T20, e.g. 142/9 chasing 158 — or the innings is rain-shortened.
The first two are unambiguous. The third was, until this revision, left
to the regression rather than forced to a clean loss: the required-run-rate
term in `rate_gap` collapses toward 0 as `balls_remaining_frac` approaches
0 regardless of how many runs are actually still needed, which made an
obviously-lost chase look statistically "on pace" right at the end. The
fix is a single rule applied uniformly: whenever an innings has
concluded (all out, or out of deliveries) without reaching the target,
WP is set from the actual match `winner` rather than the model's score.
This one rule also correctly handles ties (via the recovered Super Over
winner — see Data Loading above) and DLS-shortened matches, since all
three cases reduce to "the chase concluded; defer to the real outcome."
**Validation:** final-ball WP now matches the true outcome in all 635
decided matches in the 2008–2025 dataset (prior to this fix, roughly a
third of decided matches showed an incorrect final-ball WP).

A **Bayesian shrinkage** term (`PSEUDO_OVERS=2.0`) pulls early-innings
current-run-rate estimates toward the required/par rate, so the first
ball or two of an innings (a 1–2 ball sample) can't swing LI unrealistically.

**Calibration check:** bucketing every 2nd-innings delivery by its
predicted decile and comparing to the actual win rate in that bucket
shows close agreement across the full range — e.g. the 0–10% predicted
bucket has an actual win rate of 3.3%, and the 90–100% bucket has an
actual win rate of 98.3% — indicating the model is reasonably
well-calibrated rather than systematically over- or under-confident.

**DLS handling:** the dataset has no DLS resource-percentage column, so
an exact DLS par-score recalculation isn't possible — this is documented
as an explicit limitation rather than silently approximated.
`detect_dls_matches()` instead flags rain-shortened innings via an
innings-completion heuristic: an innings that ends without reaching the
target, being bowled out, or facing all 120 legal deliveries must have
been interrupted. The threshold for this check was previously `< 114`
legal balls (under 19 overs), which missed a real match shortened to
exactly 19 overs; it is now the logically correct `< 120` (under a full
20 overs) for both innings. 45 matches are flagged on this basis across
the full dataset.

## Module 02 — Archetype Classification (`archetypes.py`)

Archetypes: **Powerplay Aggressor**, **Anchor**, **Death Specialist**
(batting), and **Middle-Over Squeezer**, **Death Specialist** (bowling).

Players are first assigned a **primary discipline** (batter or bowler,
based on whichever had more deliveries that season), then Z-scored
against same-discipline, same-season peers, then classified via argmax
across the discipline-appropriate archetype candidates:

```
score_anchor                       = z(middle_overs_balls_share) - z(dismissal_rate)
score_powerplay_aggressor          = z(powerplay_strike_rate)
score_death_specialist (batting)   = z(death_overs_strike_rate)
score_middle_over_squeezer         = -z(middle_overs_economy)   # low economy is good
score_death_specialist (bowling)   = -z(death_overs_economy)
```

**Every player-season is classified — there is no "Unclassified"
category.** An earlier version of this model required at least 60 balls
faced or bowled in a season before computing a Z-score at all, and
grouped everyone below that threshold into "Unclassified" (28.1% of all
player-seasons league-wide). This conflated two different things: a
player whose role genuinely couldn't be determined from the data, and a
player who simply had limited opportunities that season but still has a
real, scoreable performance profile. The fix: `zscore_within_season()`
computes the Z-score baseline (mean and standard deviation) using only
the stable, ≥60-ball population, so a handful of small-sample outliers
can't distort the baseline — but every player-season, stable or not, is
then scored against that fixed baseline and assigned a real archetype.
Player-seasons below the 60-ball threshold are flagged with a boolean
`is_low_sample` column instead of being dropped or bucketed separately,
so the distinction between "this player's role is genuinely unclear" and
"this player simply had a small sample" is preserved in the data rather
than collapsed into one undifferentiated category.

**Validation:** the 2025 season's classifications were checked against
well-known player reputations with zero manual overrides — V Kohli
(Anchor), RG Sharma (Powerplay Aggressor), JJ Bumrah (Middle-Over
Squeezer), and MS Dhoni, YS Chahal, and SP Narine (all Death Specialist)
— consistent with real-world cricketing reputation in every case checked.

## Module 03 — Strategic Philosophy Mapping (`strategy_mapping.py`)

**Archetype Mix** per (season, team) = percentage of total
participation-weight (matches played) belonging to each of the four
archetypes; since Module 02 no longer has an Unclassified category, mix
percentages sum to 100% across exactly four archetypes for every
team-season with any qualifying participation.

**Low-Sample Share** per (season, team) = the percentage of a team's
participation held by `is_low_sample` players (see Module 02), computed
separately by `compute_low_sample_share()`. This replaces an earlier,
conflated "Unclassified % vs win%" metric with a cleanly defined one: it
answers "how much of this team's output came from below-threshold
players" without also trying to encode "what role did they play."

Win% is correlated against each archetype's mix percentage, and
separately against Low-Sample Share, via Pearson correlation across all
156 team-seasons in the full 2008–2025 dataset:

| Variable | r | p-value |
|---|---|---|
| Low-Sample Share | −0.362 | < 0.001 |
| Anchor mix % | +0.245 | 0.002 |
| Powerplay Aggressor mix % | −0.159 | 0.047 |
| Death Specialist mix % | −0.081 | 0.317 |
| Middle-Over Squeezer mix % | +0.035 | 0.667 |

Low-Sample Share is the strongest and most reliable signal in this
module. Among the archetypes themselves, only Anchor mix % has its own
statistically reliable relationship with win%; Powerplay Aggressor's
relationship is weak and at the edge of conventional significance, and
Middle-Over Squeezer and Death Specialist show no reliable relationship
in either direction at this sample size.

## Module 04 — Behavioral Analysis (`behavioral_analysis.py`)

For players who change their primary team between active seasons,
compares `archetype_score` (the same Z-score-derived composite from
Module 02) before vs. after the move:

- **Stable**: the year-over-year shift is within 1 league-wide standard
  deviation of that archetype's score distribution.
- **Adaptable**: the shift exceeds that threshold.

Only moves where the player kept the **same archetype** before and after
are compared — a genuine role change (e.g. Powerplay Aggressor → Anchor)
is a different question from "did this player's output stay stable in
the same role," so those moves are excluded from this comparison. Of 824
team changes tracked across the full dataset, 378 qualify on this basis.

| Archetype | Moves compared | % Stable | % Adaptable |
|---|---|---|---|
| Death Specialist | 139 | 69.8% | 30.2% |
| Middle-Over Squeezer | 106 | 70.8% | 29.2% |
| Powerplay Aggressor | 74 | 66.2% | 33.8% |
| Anchor | 59 | 54.2% | 45.8% |

Anchor is the least stable role after a team change by a clear margin. A
diagnostic `moved_toward_new_team_mean` flag additionally records whether
an "Adaptable" player's shift moved them closer to their new team's
typical archetype_score for that role: of Adaptable Anchors, roughly 63%
shifted toward their new team's typical profile, consistent with
anchoring being shaped substantially by what a given lineup needs rather
than being a fixed trait carried by the player.

## Module 05 — Liability Identification (`liability.py`)

Players are flagged as **underperformers** if their `archetype_score`
falls at or below the league-wide bottom 25th percentile for their
archetype/season (a Z-score-based metric, not a raw counting stat). 814
player-seasons are flagged on this basis across the full dataset.

**Replacement Delta** projects the team win% impact of swapping an
underperformer for a league-median performer of the same archetype.
Rather than picking an arbitrary "1 score point = X% win probability"
constant, the score-to-win% relationship is **calibrated empirically**:
for each archetype, team-season "archetype quality" (participation-weighted
mean archetype_score) is regressed against team win% via OLS, and the
resulting slope is applied to `weight_share * (league_median -
player_score)`, where `weight_share` reflects how much of the team's
archetype-participation the player accounts for.

| Archetype | Slope | p-value | Used for Replacement Delta? |
|---|---|---|---|
| Powerplay Aggressor | +4.891 | 0.043 | Yes |
| Middle-Over Squeezer | +0.185 | 0.929 | No — floored at 0 |
| Death Specialist | −0.013 | 0.997 | No — floored at 0 |
| Anchor | −0.296 | 0.907 | No — floored at 0 |

Only Powerplay Aggressor's slope is statistically distinguishable from
zero once computed across the full 18-season dataset. **The calculation
floors `slope_for_replacement` at 0** for the other three archetypes (a
fix made during this revision) so that a negative or unreliable slope
never produces a Replacement Delta implying that upgrading an
underperformer would *hurt* the team — the player is still correctly
flagged as an underperformer, but is not assigned a win% figure the
regression can't support. The unfloored, raw slope is still shown
separately in the output for transparency.

## Module 06 — Tenure & Ranking Impact (`tenure_ranking.py`)

**Tenure** = consecutive seasons at one (normalized) franchise. A tenure
spell resets on either a team change or a gap in active seasons — this
includes, for example, Chennai Super Kings and Rajasthan Royals players
whose tenure spell correctly resets after the franchises' 2016–2017
suspension, since a 2-year absence from the league is not continuous
service even if the player returns to the same franchise afterward.

**Team Tenure Density** = participation-weighted average tenure of a
team's season squad. **Strategy Consistency** is measured as the
Euclidean-distance "shift magnitude" between a team's archetype mix
vector in consecutive seasons (lower = more stable strategic identity);
`compute_strategy_consistency_by_transition()` explicitly checks that the
two seasons being compared are consecutive (`season - prev_season == 1`)
and nulls out the comparison otherwise, so a gap in a team's active
seasons (e.g. the 2016–2017 suspension above) cannot be silently treated
as a one-season transition.

These two measures are correlated at team-season granularity. **This
correlation does not hold at full scale.** An earlier version of this
analysis, run on a six-season window, reported r = +0.54 (p < 0.0001).
Recomputed across the complete 18-season dataset (n = 139 team-season
transitions with both values defined), the relationship is r = +0.003,
p = 0.97 — statistically indistinguishable from no relationship at all.
The underlying tenure-density and mix-change-magnitude values were
checked directly for missing values, duplicated rows, and implausible
outliers; none were found. The most plausible explanation is that
eighteen seasons span several distinct tactical eras of T20 cricket
(powerplay-overs rule changes, a general rise in scoring rates, and the
2023 Impact Player rule among them), and a single Pearson correlation
pooled across all of them does not capture a relationship that may hold
within any one era. This is reported as the correct result for the full
dataset rather than reverted toward the earlier, smaller-sample figure.

**Top-5 Ranking Streaks**: progressive in-season standings are
reconstructed match-by-match in chronological order (2 points per win; a
tie or no-result, where applicable, awards 1 point to each side per the
recovered ground-truth result in Data Loading above). Each team's rank is
recorded as it stood **entering** each match (using only prior results,
so a match's own outcome can't leak into whether it's labeled "in
contention"). Maximal consecutive-match windows where a team's rank was
Top-5 are then extracted as streaks — 192 such streaks were identified
across the full 2008–2025 dataset.

The algorithm reproduces real IPL history well as a sanity check: Mumbai
Indians show a long Top-5 streak through their 2020 title run; Gujarat
Titans, in their maiden 2022 season, surge into the Top-5 early and hold
it for the rest of the season en route to the title; and Royal
Challengers Bengaluru's 2025 title run shows the same pattern.

Player strike rate (batting) and economy (bowling) during these streak
windows are compared against full-season averages, using the same
formulas as Module 02 for direct comparability. **Documented limitation:**
streak windows often comprise a large share of a strong team's matches,
so streak and season figures are naturally correlated — this pulls the
population-level mean delta toward zero even when individual players
show meaningful swings; the per-player deltas in the output are the more
useful signal than the aggregate mean.

**Documented limitation (standings):** the dataset has no Net Run Rate
column, so ties in points use standard competition ranking (equal points
share a rank) rather than the real IPL's NRR tiebreak — Top-5 boundaries
can differ slightly from official standings when several teams are level.

## Longest-Serving Core (dashboard feature, `dashboard/extract_data.py`)

Not a numbered pipeline module, but built on the same `player_tenure`
and `primary_team` data as Module 06. For each team, the five players
with the most distinct seasons as that team's primary team are
identified, and their full season-by-season archetype and key statistic
(strike rate for batters, economy for bowlers) are assembled into a
timeline, with an `is_low_sample`-derived flag per season and a
`roleChanged` boolean (true if the player's archetype varied across
their tenure). This always reflects the player's complete tenure at the
franchise regardless of any season filter applied elsewhere in the
dashboard. Spot-checked against real career history: Mumbai Indians'
longest-serving player is Rohit Sharma (15 seasons), whose archetype
progression in the data — Powerplay Aggressor in his earlier seasons,
shifting toward Anchor and Death Specialist in later ones — matches his
well-documented real-world career arc and captaincy tenure.
