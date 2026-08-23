"""Canonical method names used in configs, logs, and experiment tables."""

REFERENCE_METHOD = "reference"
TRAINING_METHODS = (
    "reward_ppo",
    "self_play_ppo",
    "nash_md_ppo",
    "nash_rs",
    "mpo_ppo",
    "egpo_ppo",
    "comal_ppo",
)
ALL_METHODS = (REFERENCE_METHOD,) + TRAINING_METHODS

