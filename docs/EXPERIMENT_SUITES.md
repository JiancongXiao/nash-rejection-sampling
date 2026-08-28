# Experiment suite names

The repository uses exactly two experiment-suite names:

1. **Main**: Nash-RS versus method-native faithful baselines.
2. **Controlled ablation**: all methods adapted to one OpenRLHF/PPO backend.

Do not describe Controlled ablation results as reproductions of the source
papers. Their role is to isolate opponent/reward construction under a shared
optimizer. Main is the primary comparison and must be completed first.

## Main implementation boundary

| Method | Main implementation | Source fidelity |
|---|---|---|
| Reward-PPO | OpenRLHF PPO | Native PPO baseline |
| Self-play | TRL 0.13 `NashMDTrainer` with mixture coefficient zero | Native endpoint |
| Nash-MD | TRL 0.13 token-level geometric-mixture `NashMDTrainer` | Native practical algorithm |
| Nash-RS | Gibbs RS, implicit reward, OpenRLHF PPO | Native Nash-RS implementation |
| MPO | Algorithm 1, ReMax baseline, KL magnet, periodic magnet update | Paper-faithful Safe-RLHF contract |
| EGPO | Author-released extragradient trainer | Official implementation |
| COMAL | Author-released generation/scoring/logprob/INPO pipeline | Official implementation |

The full Main runs share model initialization, prompt splits, preference
oracle, compatible decoding settings, evaluation, and resource accounting.
They intentionally do **not** share one optimizer or one set of PPO
hyperparameters.

## Smoke acceptance criteria

A 0.5B Main smoke run is an engineering check, not an experimental result. It
passes only when all of the following hold:

- the trainer performs 2--4 optimizer steps;
- step metrics include loss, learning rate, preference-model calls, generated
  tokens, and GPU-hours;
- the saved actor differs from the initial actor in at least one tensor;
- the run writes `smoke_manifest.json` with `suite: main`;
- the selected trainer/opponent construction matches the method-native path.
