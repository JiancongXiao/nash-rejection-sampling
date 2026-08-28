"""Registry and launch contracts for the method-native Main experiment suite.

The existing :mod:`nashrs.methods` package is the Controlled ablation suite:
all methods there are adapted to one OpenRLHF/PPO backend.  This module keeps
the method-native Main suite separate so logs and tables cannot accidentally
mix the two experiment families.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MAIN_SUITE = "main"
CONTROLLED_SUITE = "controlled_ablation"


@dataclass(frozen=True)
class ExternalSource:
    """A source tree required by a method-native implementation."""

    name: str
    url: str
    revision: str
    subdirectory: str


@dataclass(frozen=True)
class MainMethodSpec:
    """Immutable launch metadata for one Main-table method."""

    name: str
    display_name: str
    backend: str
    entrypoint: str
    fidelity: str
    sources: tuple[ExternalSource, ...] = ()
    min_gpus: int = 1
    smoke_steps: int = 2

    @property
    def experiment_id(self) -> str:
        return f"{MAIN_SUITE}/{self.name}"


COMAL_SOURCE = ExternalSource(
    name="comal",
    url="https://github.com/yale-nlp/COMAL.git",
    revision="30e9446fbc5212d9e6886a27d09d1266e50a45c0",
    subdirectory="COMAL",
)
EGPO_SOURCE = ExternalSource(
    name="egpo",
    url="https://github.com/zhourunlong/EGPO.git",
    revision="f295bf3750627374c7ddb5d4459390321de2836e",
    subdirectory="EGPO",
)
SAFE_RLHF_SOURCE = ExternalSource(
    name="safe_rlhf",
    url="https://github.com/PKU-Alignment/safe-rlhf.git",
    revision="e8cca16665ef2340ac92c6514f05519310251581",
    subdirectory="safe-rlhf",
)


MAIN_METHOD_SPECS = (
    MainMethodSpec(
        name="reward_ppo",
        display_name="Reward-PPO",
        backend="openrlhf-ppo",
        entrypoint="cluster/hopper/main/main_0p5b_smoke.pbs",
        fidelity="native PPO baseline; OpenRLHF is the declared implementation",
        smoke_steps=4,
    ),
    MainMethodSpec(
        name="self_play",
        display_name="Self-play (Nash-MD-PG beta=0)",
        backend="trl-0.13-nash-md",
        entrypoint="cluster/hopper/main/main_0p5b_smoke.pbs",
        fidelity="TRL NashMDTrainer native self-play endpoint with geometric mixture coefficient zero",
        smoke_steps=2,
    ),
    MainMethodSpec(
        name="nash_md",
        display_name="Nash-MD-PG",
        backend="trl-0.13-nash-md",
        entrypoint="cluster/hopper/main/main_0p5b_smoke.pbs",
        fidelity="TRL NashMDTrainer token-level geometric-mixture Nash-MD-PG update",
        smoke_steps=2,
    ),
    MainMethodSpec(
        name="nash_rs",
        display_name="Nash-RS",
        backend="openrlhf-ppo",
        entrypoint="cluster/hopper/main/main_0p5b_smoke.pbs",
        fidelity="native practical Nash-RS: Gibbs rejection sampling plus implicit reward PPO",
        smoke_steps=4,
    ),
    MainMethodSpec(
        name="mpo",
        display_name="MPO",
        backend="safe-rlhf-mpo",
        entrypoint="cluster/hopper/main/main_0p5b_smoke.pbs",
        fidelity="paper Algorithm 1 on the Safe-RLHF base with ReMax and periodic magnet updates",
        sources=(SAFE_RLHF_SOURCE,),
        smoke_steps=2,
    ),
    MainMethodSpec(
        name="egpo",
        display_name="EGPO",
        backend="egpo-official",
        entrypoint="cluster/hopper/main/main_0p5b_smoke.pbs",
        fidelity="official online-IPO extragradient predictor/corrector trainer",
        sources=(EGPO_SOURCE,),
        smoke_steps=2,
    ),
    MainMethodSpec(
        name="comal",
        display_name="COMAL",
        backend="comal-official-inpo",
        entrypoint="cluster/hopper/main/main_0p5b_smoke.pbs",
        fidelity="official sampling/scoring/logprob pipeline with INPO inner updates",
        sources=(COMAL_SOURCE,),
        min_gpus=4,
        smoke_steps=2,
    ),
)

MAIN_METHODS = tuple(spec.name for spec in MAIN_METHOD_SPECS)
_BY_NAME = {spec.name: spec for spec in MAIN_METHOD_SPECS}


def get_main_method(name: str) -> MainMethodSpec:
    try:
        return _BY_NAME[name]
    except KeyError as error:
        choices = ", ".join(MAIN_METHODS)
        raise ValueError(f"unknown Main method {name!r}; choose one of: {choices}") from error


def validate_smoke_steps(steps: int) -> int:
    """Main smoke tests intentionally stop after two to four optimizer steps."""

    if not 2 <= steps <= 4:
        raise ValueError("Main smoke tests must use between 2 and 4 optimizer steps")
    return steps


def validate_optimizer_steps(steps: int) -> int:
    """Validate a positive optimizer-step budget for non-smoke Main runs."""

    if steps <= 0:
        raise ValueError("optimizer steps must be positive")
    return steps


def validate_source_checkout(root: Path, source: ExternalSource) -> list[str]:
    """Return human-readable problems without modifying an external checkout."""

    checkout = root / source.subdirectory
    problems = []
    if not checkout.is_dir():
        return [f"missing checkout: {checkout}"]
    if not (checkout / ".git").exists():
        problems.append(f"not a git checkout: {checkout}")
    return problems


def required_sources() -> tuple[ExternalSource, ...]:
    sources = {source.name: source for spec in MAIN_METHOD_SPECS for source in spec.sources}
    return tuple(sources[name] for name in sorted(sources))
