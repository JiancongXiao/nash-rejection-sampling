"""Core interfaces for unified PPO-based NLHF experiments."""

from .evaluation import PairwiseEvaluation, evaluate_pairwise
from .interfaces import (
    GeometricMixtureSampler,
    PreferenceOracle,
    RewardConstructor,
    ScalarRewardOracle,
    TextSampler,
)
from .nash_rs import NashRSConfig, NashRSRewardConstructor
from .preference_oracles import (
    BTLComponent,
    MixtureBTLPreferenceOracle,
    TransformersScalarRewardOracle,
)
from .runner import IterationResult, PPOBackend, PhaseResult, run_iteration
from .types import Accounting, RewardBatch

__all__ = [
    "Accounting",
    "BTLComponent",
    "NashRSConfig",
    "NashRSRewardConstructor",
    "MixtureBTLPreferenceOracle",
    "IterationResult",
    "PPOBackend",
    "PhaseResult",
    "GeometricMixtureSampler",
    "PreferenceOracle",
    "PairwiseEvaluation",
    "RewardBatch",
    "RewardConstructor",
    "ScalarRewardOracle",
    "TextSampler",
    "TransformersScalarRewardOracle",
    "evaluate_pairwise",
    "run_iteration",
]
