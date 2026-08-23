# Implemented comparison methods

All methods share the same PPO/OpenRLHF backend. They differ in opponent and
reward construction, KL reference, and the minimum state transitions required
by the original algorithm. The classes here are explicit **PPO adaptations**;
only Nash-RS is the method developed in this repository.

## Shared controller contract

Each controller exposes ordered `PhaseSpec` objects and constructs a scalar
reward batch. `PhaseSpec.kl_reference` tells the trainer which checkpoint must
be used for KL regularization. `start_from="iteration_start"` means that a
phase must restore the checkpoint from the beginning of the iteration before
performing PPO updates.

| Name | Opponent | KL reference | State transition |
|---|---|---|---|
| `reward_ppo` | none | initial reference | none |
| `self_play_ppo` | current policy | initial reference | none |
| `nash_md_ppo` | geometric current/reference mixture | initial reference | none |
| `nash_rs` | Gibbs distribution sampled by RS | initial reference | none |
| `mpo_ppo` | fixed magnetic snapshot | magnetic snapshot | synchronize every interval |
| `egpo_ppo` | current, then predictor | initial reference | prediction/correction |
| `comal_ppo` | current policy | outer anchor | advance anchor after inner solve |

## Reward PPO

The conventional RLHF baseline uses a scalar reward model. It does not handle
general non-transitive preferences, but controls for the shared PPO backend.

## Self-Play PPO

For each current response `y`, sample `y' ~ pi_t` and use
`mean(P(y > y')) - 1/2` as reward. This is also the `beta=0` endpoint of the
practical Nash-MD-PG opponent.

## Nash-MD PPO adaptation

Sample the opponent from the geometric mixture

```text
log pi_mix = (1-beta) log pi_t + beta log pi_ref + constant.
```

For autoregressive LLMs this must be implemented by a token-level logit
mixture. Sampling current or reference policies with Bernoulli probability is
an arithmetic mixture and is not equivalent.

## Nash-RS

Estimate `g_hat(y') = mean_i P(Y_i > y')` using `B1` current-policy samples,
then accept a reference proposal exactly when

```text
u <= exp(-g_hat/tau), equivalently g_hat <= -tau*log(u).
```

Use `B2` accepted opponents to estimate the implicit reward. Finite `B1`
generally biases the Gibbs target; finite `B2` adds variance.

## MPO PPO adaptation

During each interval, both the opponent and KL magnet are one fixed snapshot.
After `update_interval` completed PPO iterations, snapshot the latest policy
and replace both. This implements the practical periodic magnetic-policy
update, not the exact tabular MMD step.

## EGPO PPO adaptation

Every iteration has two phases:

1. prediction: construct rewards against `pi_t` and update to `pi_{t+1/2}`;
2. correction: restore `pi_t`, construct rewards against the frozen predictor,
   and update to `pi_{t+1}`.

Restoring the iteration-start policy before correction is essential: both
updates in predictive EGPO start from `theta_t`.

## COMAL PPO adaptation

Within an outer round, solve a KL-regularized self-play game using PPO while
holding the anchor fixed. After `inner_iterations`, snapshot the latest policy
as the next anchor. The convergence theorem assumes sufficiently accurate
inner solutions; a fixed small number of PPO steps is an approximation.

## Theory disclaimer

Policy-space convergence results in the source papers do not automatically
transfer to neural PPO implementations. Report these implementations as
`*-PPO adaptations`, keep their shared hyperparameters fixed, and evaluate
both Nash quality and compute/query cost.

