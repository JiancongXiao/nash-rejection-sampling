# Main: method-native faithful baselines

`Main` is deliberately separate from the existing `Controlled ablation`
OpenRLHF/PPO adaptations. Main uses each method's native update:

| Method | Native path |
|---|---|
| Reward-PPO | OpenRLHF PPO |
| Self-play | TRL 0.13 `NashMDTrainer` with geometric coefficient 0 |
| Nash-MD | TRL 0.13 token-level geometric `NashMDTrainer` |
| Nash-RS | Gibbs rejection sampling + implicit-reward OpenRLHF PPO |
| MPO | paper Algorithm 1/ReMax with periodic magnet; Safe-RLHF contract |
| EGPO | authors' official extragradient trainer |
| COMAL | authors' official INPO loss and outer pipeline |

Public external sources are pinned by commit and stored under
`/scratch/jiancongxiao/external/nashrs-main`. Existing source trees are never
reset. The setup script only checks out explicitly disposable dependency trees.

```bash
qsub cluster/hopper/main/setup_native_env.pbs
```

After setup succeeds, submit all seven 0.5B smoke jobs:

```bash
bash cluster/hopper/main/submit_0p5b_smokes.sh 47
```

Each smoke job performs 2--4 real optimizer steps, writes `step_metrics.jsonl`,
checks a model tensor against its initial value, and only succeeds after
writing `smoke_manifest.json`. These smoke results are engineering validation,
not paper results.

COMAL's full Main experiment still uses its complete multi-stage official
pipeline and needs a separate four-GPU PBS job. The one-GPU smoke only validates
its published INPO inner update and artifact contract.
