"""
IPL Performance Analytics Suite
================================
A modular toolkit for analyzing IPL ball-by-ball data (2017-2022):
leverage index, player archetypes, team strategy mix, behavioral
adaptation, liability/replacement analysis, and tenure/ranking effects.

Modules:
    data_loader          - load, clean, normalize team/season/player identifiers
    leverage_index        - per-ball Win Probability swing (Leverage Index)
    archetypes             - Z-score based player archetype classification
    strategy_mapping       - team archetype mix vs win % correlation
    behavioral_analysis    - Stable vs Adaptable player classification on team change
    liability               - underperformer detection + replacement delta
    tenure_ranking          - tenure, strategy consistency, ranking-streak performance
"""

__version__ = "1.0.0"
