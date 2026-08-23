# Minimal implementation plan

## Fair-comparison contract

Every run is derived from one shared experiment manifest. The following fields
must not vary by method: prompt split and order, base/reference checkpoints and
revisions, preference model and revision, random-seed set, decoding settings,
response-token budget, rollout/train batch sizes, PPO epochs, clip range, KL
coefficient, optimizer settings, precision, and total generation/query budget.

The sole method-specific boundary is:

```text
(prompts, current PPO responses)
        -> opponent construction
        -> scalar implicit rewards + accounting
        -> shared OpenRLHF PPO update
```

`RewardConstructor` encodes this boundary. Nash-MD-, MPO-, and COMAL-style PPO
adaptations should implement it without modifying the shared trainer.

## Stage 1: completed CPU core

- Nash-RS finite-`B1` Gibbs opponent construction.
- Correct rejection event: `u <= exp(-g_hat/tau)`.
- Finite-`B2` implicit reward estimation.
- Batched preference-oracle interface.
- Proposal, acceptance, generation, token, PM-call, wall-time, and GPU-hour
  accounting fields.
- Pairwise matrix, average win rate, worst-case win rate, and empirical
  exploitability.
- OpenRLHF custom-reward output adapter.
- Deterministic smoke test and unit tests.

## Stage 2: first GPU vertical slice

1. Pin an OpenRLHF commit and environment lockfile. (completed)
2. Implement a vLLM-backed current/reference `TextSampler`. (completed)
3. Implement a batched local preference-model `PreferenceOracle`. (completed)
4. Expose one configured `reward_func` through `--reward.remote_url`.
5. Run 32 prompts on a small model; persist responses, opponents, rewards,
   acceptance diagnostics, and accounting as JSONL.
6. Verify the same response set can be replayed through the evaluator.

## Stage 3: unified method controllers (completed)

- Reward-PPO and self-play PPO baselines.
- Nash-MD-PPO: geometric-mixture opponent/reward construction.
- MPO-PPO: fixed magnetic opponent/reference with periodic synchronization.
- EGPO-PPO: prediction/correction phases with iteration-start restoration.
- COMAL-PPO: fixed outer anchor and PPO-based regularized inner game.
- Backend-agnostic runner enforcing phase restore/snapshot semantics.

## Stage 3b: official-code checks (pending)

- Run official-code sanity checks separately. Label unified versions as PPO
  adaptations, not official reproductions.

## Stage 4: main LLM experiments

- Primary: Llama-3.1-8B-Instruct.
- Cross-family check: Qwen2.5-7B-Instruct.
- Standard and genuinely non-BT preference settings.
- AlpacaEval 2.0, Arena-Hard, and one open-ended benchmark.
- Quality: benchmark win rate and pairwise matrix.
- Nash quality: worst-case win rate and empirical exploitability.
- Efficiency: PM calls, generated tokens, GPU hours, peak memory, and
  exploitability versus compute/query budget.

## Theory-to-code scope

The exact policy-space mirror update has the linear last-iterate guarantee.
This repository's PPO path is a parametric approximation and should be reported
with a stationarity interpretation unless stronger assumptions are verified.
Finite `B2` contributes stochastic variance; finite `B1` generally biases the
Gibbs target. Experiments must log both separately.
