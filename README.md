# Nash-RS: unified PPO experiments for NLHF

This repository starts with a small, testable core for comparing NLHF methods
under one PPO/OpenRLHF training backbone. Methods share prompts, policy and
reference samplers, preference oracle, generation settings, and PPO
hyperparameters. They differ only in opponent and reward construction.

The first implemented method is Nash Rejection Sampling (Nash-RS):

1. draw `B1` current-policy responses for each prompt;
2. propose opponent responses from the reference policy;
3. estimate `g_hat(y') = mean_i P(Y_i > y')`;
4. accept exactly when `u <= exp(-g_hat / tau)`, equivalently
   `g_hat <= -tau log(u)`;
5. estimate the implicit reward of each PPO response from `B2` accepted
   opponents.

The implementation records preference-model calls, generated responses,
generated tokens, proposals, acceptances, and wall-clock time.

## Local smoke test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 examples/toy_nash_rs.py
```

## OpenRLHF boundary

`nashrs.openrlhf.OpenRLHFRewardAdapter` converts a method reward batch into the
custom reward dictionary expected by current OpenRLHF PPO jobs. In production,
provide distributed policy/reference samplers and a batched preference-model
client, then expose the adapter from the file supplied to
`--reward.remote_url`.

The exact policy-space convergence theorem does not automatically apply to
this PPO implementation. Finite `B2` adds variance; finite `B1` generally adds
bias to the Gibbs target.

See `docs/IMPLEMENTATION_PLAN.md` for the staged OpenRLHF integration plan and
the fairness contract shared by all methods.
