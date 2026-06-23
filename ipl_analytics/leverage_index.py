"""
leverage_index.py
===================
LEVERAGE INDEX (LI): the absolute swing in Win Probability (WP) caused by a
single delivery.  LI[ball_i] = | WP(state AFTER ball_i) - WP(state BEFORE ball_i) |

This is conceptually identical to the "Leverage Index" used in baseball
sabermetrics (Tom Tango) and the "Volatility Index" / WPA frameworks used in
cricket analytics (e.g. ESPNcricinfo's Forecaster) -- a high-leverage ball is
one where a lot of the match outcome hinges on what happens next.

------------------------------------------------------------------------------
WIN PROBABILITY MODEL (the "simplified heuristic")
------------------------------------------------------------------------------
A *true* ball-by-ball WP model (e.g. WinViz/Forecaster-style) needs far more
than 373 matches and far more contextual features (venue scoring history,
opposition bowling attack strength, pitch report, etc.) to be reliable. The
brief explicitly allows a simplified heuristic, so we build one that is more
rigorous than a hand-picked formula but still clearly a heuristic:

  1. For run-chases (innings 2) we have a clean binary label - did the
     chasing team win? - so we fit a small LOGISTIC REGRESSION using three
     interpretable engineered features:
         - rate_gap             = current_run_rate - required_run_rate   (runs/over)
         - wickets_in_hand_frac = wickets_in_hand / 10
         - balls_remaining_frac = balls_remaining / effective_total_balls
     This is exactly the "scale current run rate against required run rate
     and wickets in hand" heuristic the brief asks for, just with the
     scaling WEIGHTS learned from the data instead of guessed.

  2. For innings 1 there is no target yet, so there's no "required rate" to
     compare against. We construct an analogous reference: each ball is
     compared against that SEASON's average par run-rate (mean first-innings
     score / 20), producing a "rate_gap" in the same units. We then reuse the
     SAME fitted logistic model -- i.e. "what is a chasing team's historical
     win probability when it has this rate-gap / wickets-in-hand / balls-
     remaining profile?" is repurposed as "how competitive is this 1st-innings
     score looking, relative to a typical par chase?". This keeps the whole
     suite to one coherent, explainable model rather than two arbitrary ones.

Caveats (documented honestly, not hidden):
  - Sample size is small (one IPL's worth of matches per ~60-75 games/season);
    coefficients should be treated as indicative, not production-grade.
  - No venue / opposition-strength / pitch adjustment.
  - Innings-1 "win probability" is a proxy (competitiveness of the score),
    not a calibrated WP, since the model was trained on chase outcomes only.

------------------------------------------------------------------------------
DLS NORMALIZATION
------------------------------------------------------------------------------
The raw data has no `method`/interruption column (the official DLS resource
table is also proprietary, so it can't be reconstructed exactly). What we
CAN do reliably from ball-by-ball data is detect when an innings was
genuinely shortened by an interruption (as opposed to ending early because a
team was bowled out, or because a chase was completed ahead of schedule) and
then compute run rates against the ACTUAL number of legal deliveries that
innings had available, instead of wrongly assuming a full 20 overs. That
correction is what `effective_total_balls` does below. We flag these matches
explicitly (`dls_flag`) so any downstream consumer can filter/caveat them.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# DLS / rain-shortened match detection
# ---------------------------------------------------------------------------
def detect_dls_matches(match_summary: pd.DataFrame) -> pd.DataFrame:
    """Flags matches where an innings was genuinely curtailed by an
    interruption (rain etc.), as distinct from a team being bowled out or a
    chase finishing early because the target was already reached.

    Heuristic:
        innings 1 short  := legal_balls_inn1 < 120 AND wickets_inn1 < 10
        innings 2 short  := legal_balls_inn2 < 120 AND wickets_inn2 < 10
                             AND runs_inn2 < target
                             (excludes the very common case of a chase being
                              won with overs to spare, which is NOT an
                              interruption)

    BUG FIX: this previously used a threshold of "< 114" (under 19 overs)
    rather than "< 120" (under a full 20 overs), on the assumption that a
    small buffer would avoid false positives. That buffer was wrong: there
    is no legitimate reason for a chase to end before all 120 legal
    deliveries are bowled without either losing all 10 wickets or reaching
    the target -- any such case is, by construction, an interruption. The
    looser threshold missed at least one real match shortened to exactly
    19 overs (114 balls), which fell outside the "< 114" cutoff and was
    left unflagged, causing its win-probability terminal state to fall
    through to the raw (wrong) regression output instead of being
    corrected. "< 120" is not an approximation, it is the exact condition.
    """
    ms = match_summary.copy()
    inn1_short = (ms["legal_balls_inn1"] < 120) & (ms["wickets_inn1"] < 10)
    inn2_short = (
        (ms["legal_balls_inn2"] < 120)
        & (ms["wickets_inn2"] < 10)
        & (ms["runs_inn2"] < ms["target"])
    )
    ms["dls_flag"] = inn1_short | inn2_short
    ms["effective_balls_inn1"] = np.where(inn1_short, ms["legal_balls_inn1"], 120)
    ms["effective_balls_inn2"] = np.where(inn2_short, ms["legal_balls_inn2"], 120)
    return ms


# ---------------------------------------------------------------------------
# Season "par rate" for innings-1 reference (used as the WP-model's analogue
# of "required run rate" before a target exists)
# ---------------------------------------------------------------------------
def compute_season_par_rate(match_summary: pd.DataFrame) -> pd.Series:
    """Average first-innings score per season, expressed as runs/over."""
    par_score = match_summary.groupby("season")["runs_inn1"].mean()
    return par_score / 20.0


# ---------------------------------------------------------------------------
# Feature engineering (shared by training and scoring)
# ---------------------------------------------------------------------------
def _engineer_chase_features(df: pd.DataFrame, match_summary: pd.DataFrame) -> pd.DataFrame:
    """Builds rate_gap / wickets_in_hand_frac / balls_remaining_frac for every
    ball, evaluated at the state BEFORE and the state AFTER that delivery, for
    innings 2 (the chase)."""

    ms = match_summary.set_index("match_id")
    out = df.copy()
    out["target"] = out["match_id"].map(ms["target"])
    out["effective_balls"] = out["match_id"].map(ms["effective_balls_inn2"])

    for suffix in ["before", "after"]:
        if suffix == "before":
            runs = out["cum_runs_before"]
            wkts = out["cum_wickets_before"]
            balls_bowled = out["innings_legal_ball_no"] - out["legal_ball"].astype(int)
        else:
            runs = out["cum_runs_before"] + out["runs_total"]
            wkts = out["cum_wickets_before"] + out["is_wicket"].astype(int)
            balls_bowled = out["innings_legal_ball_no"]

        balls_remaining = (out["effective_balls"] - balls_bowled).clip(lower=0)
        runs_needed = (out["target"] - runs).clip(lower=0)

        rrr_raw = np.where(balls_remaining > 0, runs_needed / (balls_remaining / 6), 0.0)
        rrr = pd.Series(rrr_raw, index=out.index)

        # SHRINKAGE: current run-rate computed from just 1-2 balls is
        # extremely noisy (e.g. a single dot ball makes CRR=0, which the raw
        # formula would read as "this team is miles behind the rate" even
        # though almost no information has actually arrived yet). We apply a
        # simple Bayesian shrinkage, pulling CRR toward the required rate
        # with a prior weight of PSEUDO_OVERS overs worth of deliveries, so
        # early-innings leverage isn't artificially inflated by sampling
        # noise. The shrinkage fades out smoothly as more overs are bowled.
        PSEUDO_OVERS = 2.0
        overs_bowled = balls_bowled / 6
        crr_smoothed = (runs + PSEUDO_OVERS * rrr) / (overs_bowled + PSEUDO_OVERS)

        rate_gap = crr_smoothed - rrr

        out[f"rate_gap_{suffix}"] = rate_gap
        out[f"wickets_in_hand_frac_{suffix}"] = (10 - wkts).clip(lower=0) / 10.0
        out[f"balls_remaining_frac_{suffix}"] = balls_remaining / out["effective_balls"]
        out[f"runs_needed_{suffix}"] = runs_needed
        out[f"balls_remaining_{suffix}"] = balls_remaining

    return out


def _engineer_first_innings_features(df: pd.DataFrame, match_summary: pd.DataFrame,
                                       season_par_rate: pd.Series) -> pd.DataFrame:
    """Analogous features for innings 1, using the season par-rate as the
    'required rate' reference instead of a fixed chase target."""

    ms = match_summary.set_index("match_id")
    out = df.copy()
    out["par_rate"] = out["season"].map(season_par_rate)
    out["effective_balls"] = out["match_id"].map(ms["effective_balls_inn1"])

    for suffix in ["before", "after"]:
        if suffix == "before":
            runs = out["cum_runs_before"]
            wkts = out["cum_wickets_before"]
            balls_bowled = out["innings_legal_ball_no"] - out["legal_ball"].astype(int)
        else:
            runs = out["cum_runs_before"] + out["runs_total"]
            wkts = out["cum_wickets_before"] + out["is_wicket"].astype(int)
            balls_bowled = out["innings_legal_ball_no"]

        balls_remaining = (out["effective_balls"] - balls_bowled).clip(lower=0)

        # Same Bayesian shrinkage rationale as the chase case (see
        # _engineer_chase_features): pull early-innings CRR toward the
        # season's par rate so a single early boundary/dot doesn't generate
        # an inflated leverage reading.
        PSEUDO_OVERS = 2.0
        overs_bowled = balls_bowled / 6
        crr_smoothed = (runs + PSEUDO_OVERS * out["par_rate"]) / (overs_bowled + PSEUDO_OVERS)

        # rate_gap here = how far current scoring pace is above/below the
        # season's typical par rate -- the innings-1 analogue of "ahead of
        # required rate" in a chase.
        out[f"rate_gap_{suffix}"] = crr_smoothed - out["par_rate"]
        out[f"wickets_in_hand_frac_{suffix}"] = (10 - wkts).clip(lower=0) / 10.0
        out[f"balls_remaining_frac_{suffix}"] = balls_remaining / out["effective_balls"]

    return out


# ---------------------------------------------------------------------------
# Model fitting (on chase outcomes only -- the only clean binary label we have)
# ---------------------------------------------------------------------------
def fit_win_probability_model(df: pd.DataFrame, match_summary: pd.DataFrame):
    """Fits the logistic WP model on every ball of every 2nd-innings chase,
    labeled by whether the chasing (batting) team ultimately won the match.
    Returns the fitted sklearn LogisticRegression.

    Matches with no winner (rain-abandoned "no result" matches -- see
    data_loader._resolve_true_winner) are excluded from training: labeling
    every ball of an abandoned, undecided match as a "loss" (the only
    label a naive `== winner` comparison can produce when winner is NaN)
    would inject a clean-looking but false training signal."""

    chase = df[df["innings"] == 2].copy()
    feat = _engineer_chase_features(chase, match_summary)

    ms = match_summary.set_index("match_id")
    feat["winner"] = feat["match_id"].map(ms["winner"])
    feat = feat[feat["winner"].notna()]
    feat["batting_team_won"] = (feat["winner"] == feat["batting_team"]).astype(int)

    X = feat[["rate_gap_before", "wickets_in_hand_frac_before", "balls_remaining_frac_before"]].values
    y = feat["batting_team_won"].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# WP + Leverage Index scoring for the full ball-by-ball dataset
# ---------------------------------------------------------------------------
def compute_leverage_index(df: pd.DataFrame, match_summary_raw: pd.DataFrame) -> pd.DataFrame:
    """Main entry point. Returns `df` with added columns:
        wp_before, wp_after, leverage_index, dls_flag
    """
    match_summary = detect_dls_matches(match_summary_raw)
    season_par_rate = compute_season_par_rate(match_summary)
    model = fit_win_probability_model(df, match_summary)

    def _score(feat: pd.DataFrame):
        Xb = feat[["rate_gap_before", "wickets_in_hand_frac_before", "balls_remaining_frac_before"]].values
        Xa = feat[["rate_gap_after", "wickets_in_hand_frac_after", "balls_remaining_frac_after"]].values
        wp_before = model.predict_proba(Xb)[:, 1]
        wp_after = model.predict_proba(Xa)[:, 1]
        return wp_before, wp_after

    # ---- innings 2 (chase): terminal states are handled explicitly --------
    inn2 = df[df["innings"] == 2].copy()
    feat2 = _engineer_chase_features(inn2, match_summary)
    wp_before, wp_after = _score(feat2)

    # ---- terminal overrides --------------------------------------------------
    # Whatever the regression predicts is overridden once the chase has
    # mathematically concluded, using ground truth rather than re-deriving
    # the outcome from runs/wickets alone:
    #   - target reached/passed  -> the batting side outscored the other
    #     side, an unambiguous win, WP = 1.
    #   - otherwise, once the chase has CONCLUDED (all out, or every
    #     available delivery used) -> defer to the match's actual recorded
    #     winner (data_loader._resolve_true_winner has already resolved
    #     "tie" to the real Super Over winner and "no result" to NaN).
    #     This single rule correctly covers three distinct situations that
    #     earlier, separate patches handled inconsistently:
    #       (a) a normal fall-short loss (most common T20 chase result --
    #           e.g. finishes 142/9 chasing 158): winner != batting team,
    #           so WP=0, exactly as a hand-written "all out or out of
    #           balls -> 0" rule would also give.
    #       (b) a tie that finished level after 20 overs and was then
    #           decided by Super Over: this is NOT simply a loss for the
    #           team that "didn't reach the target" -- either team could
    #           have won the Super Over, so it must defer to the resolved
    #           winner rather than being hard-coded to 0.
    #       (c) a DLS-curtailed chase decided on rain-adjusted par score:
    #           the team can legitimately win without its raw runs column
    #           ever reaching `target`, since no DLS resource table is
    #           available to recompute the true par score (documented
    #           limitation) -- again must defer to the resolved winner.
    #     A genuine no-result has winner=NaN; `chase_won` is then False,
    #     which is an acceptable fallback for the handful of partial,
    #     abandoned innings that reach this branch (no real terminal state
    #     exists for them in reality either).
    winner_map = match_summary.set_index("match_id")["winner"]
    chase_won = feat2["match_id"].map(winner_map) == feat2["batting_team"]

    reached_after = (feat2["cum_runs_before"] + feat2["runs_total"]) >= feat2["target"]
    allout_after = (feat2["cum_wickets_before"] + feat2["is_wicket"].astype(int)) >= 10
    balls_exhausted_after = feat2["balls_remaining_after"] <= 0
    concluded_after = allout_after | balls_exhausted_after
    wp_after = np.where(reached_after, 1.0,
                         np.where(concluded_after, np.where(chase_won, 1.0, 0.0), wp_after))

    reached_before = feat2["cum_runs_before"] >= feat2["target"]
    allout_before = feat2["cum_wickets_before"] >= 10
    balls_exhausted_before = feat2["balls_remaining_before"] <= 0
    concluded_before = allout_before | balls_exhausted_before
    wp_before = np.where(reached_before, 1.0,
                          np.where(concluded_before, np.where(chase_won, 1.0, 0.0), wp_before))

    feat2["wp_before"], feat2["wp_after"] = wp_before, wp_after

    # ---- innings 1: competitiveness proxy, no hard terminal win/loss state -
    inn1 = df[df["innings"] == 1].copy()
    feat1 = _engineer_first_innings_features(inn1, match_summary, season_par_rate)
    wp_before1, wp_after1 = _score(feat1)
    feat1["wp_before"], feat1["wp_after"] = wp_before1, wp_after1

    result = pd.concat([feat1, feat2], axis=0).sort_index()
    result["leverage_index"] = (result["wp_after"] - result["wp_before"]).abs()
    result["dls_flag"] = result["match_id"].map(match_summary.set_index("match_id")["dls_flag"])

    keep_extra = ["wp_before", "wp_after", "leverage_index", "dls_flag"]
    return df.merge(
        result[["match_id", "innings", "over", "ball"] + keep_extra]
        .drop_duplicates(subset=["match_id", "innings", "over", "ball"]),
        on=["match_id", "innings", "over", "ball"], how="left"
    )
