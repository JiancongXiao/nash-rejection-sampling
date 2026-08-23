"""Numerically safe rejection-sampling primitives."""

from __future__ import annotations

import math
import random


def acceptance_probability(g_hat: float, tau: float) -> float:
    """Return exp(-g_hat/tau) for g_hat in [0, 1]."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0.0 <= g_hat <= 1.0:
        raise ValueError(f"g_hat must lie in [0, 1], got {g_hat}")
    return math.exp(-g_hat / tau)


def accept_gibbs_proposal(g_hat: float, tau: float, rng: random.Random) -> bool:
    """Use the correct Gibbs acceptance direction.

    Accept iff u <= exp(-g_hat/tau), equivalently g_hat <= -tau*log(u).
    The log-domain form avoids underflow when tau is small.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0.0 <= g_hat <= 1.0:
        raise ValueError(f"g_hat must lie in [0, 1], got {g_hat}")
    u = rng.random()
    if u == 0.0:
        return True
    return g_hat <= -tau * math.log(u)

