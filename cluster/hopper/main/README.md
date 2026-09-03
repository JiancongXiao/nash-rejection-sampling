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
pipeline and uses two H200 GPUs with FSDP FULL_SHARD in Hopper's `small` queue.
Gradient accumulation is increased from 4 to 16 so the official effective
global batch size of 32 is unchanged. The one-GPU native-entry smoke only validates
its published INPO inner update and artifact contract.

For a schedulable single-H200 execution of the same complete pipeline, use:

```bash
bash cluster/hopper/main/submit_3b_comal_single.sh smoke 47
```

This changes systems parallelism only: generation and log-probability passes
run on one GPU, INPO uses one BF16 process instead of FSDP, and gradient
accumulation is 32 so the effective global batch remains 32.  The candidate
generation, preference ranking, cached log-probabilities, INPO objective, and
24 outer iterations are unchanged.  After the smoke succeeds, the single-GPU
full chain can be submitted with `full` in place of `smoke`.

Before launching all 24 full outer iterations, run the complete two-GPU pipeline
for one isolated outer iteration:

```bash
bash cluster/hopper/main/submit_3b_comal_small.sh smoke 47
```

After its `smoke_manifest.json` verifies a real parameter update, submit the
full chain. It is split into twelve two-iteration jobs for restartability:

```bash
bash cluster/hopper/main/submit_3b_comal_small.sh full 47
```

The six completed non-COMAL 3B checkpoints can be evaluated before the COMAL
chain finishes. The arguments are the completed PBS job numbers in method
order; the job evaluates the shared base plus all six models on the fixed
256-prompt selection split:

```bash
bash cluster/hopper/main/submit_3b_six_eval.sh \
  47 REWARD SELF_PLAY NASH_MD NASH_RS MPO EGPO
```

Generation is greedy with a 512-token cap. The output contains the pairwise
matrix, average and worst-case win rates, empirical exploitability, component
scores, and 95% bootstrap confidence intervals.

After all seven smokes pass, run the 16-step 0.5B native-entry pilot:

```bash
bash cluster/hopper/main/submit_0p5b_pilot.sh 47 16
```

This pilot exercises longer generation and the common accounting/manifest
contract. It is still an engineering pilot, not a Main-table result. In
particular, COMAL's paper result must use the complete multi-stage pipeline and
its four-GPU job; the pilot only checks the scalable inner-update path before
that pipeline is launched.

Once all seven pilot jobs finish, evaluate their final actors on one fixed
held-out prompt set. Pass the seven PBS job-number prefixes in method order:

```bash
bash cluster/hopper/main/submit_0p5b_pilot_eval.sh \
  47 REWARD SELF NASH_MD NASH_RS MPO EGPO COMAL
```

The evaluation includes the shared base model, performs greedy generation,
and exports the full pairwise matrix, average/worst-case win rates, empirical
exploitability, component rewards, and 95% bootstrap confidence intervals.

## 512-prompt, three-seed scaling gate

After the seven native-entry pilots pass, submit the controlled 0.5B stability
gate over one fixed 512-prompt pool and seeds 47, 101, and 211:

```bash
bash cluster/hopper/main/submit_0p5b_512x3.sh
```

This launches 21 one-GPU jobs.  All methods see the same prompt pool and use a
512-token generation cap.  The number of optimizer steps is deliberately
method-native rather than artificially equalized: OpenRLHF consumes two
prompts per global step; TRL/EGPO/COMAL consume one; MPO performs two inner
steps per outer prompt.  The submission TSV is written under
`/scratch/jiancongxiao/results/main-native-0p5b-512-submissions/`, and run
artifacts are separated by seed and method under
`/scratch/jiancongxiao/results/main-native-0p5b-512-seed*/`.

This batch is an implementation and cross-seed stability gate.  It is not a
paper result, and the COMAL entry still exercises the official INPO inner
update rather than the complete four-GPU multi-stage COMAL outer pipeline.

After all 21 training jobs finish successfully, evaluate `base + 7 methods`
for each seed on the fixed 256-prompt selection split:

```bash
bash cluster/hopper/main/submit_0p5b_512_evals.sh
```

The submitter reconstructs actor paths from the immutable training submission
TSV, requires every run manifest and parameter-update check, and then launches
three one-GPU evaluation jobs.  Each uses greedy generation, a 512-token cap,
the same preference oracle, and 10,000 bootstrap replicates.  Selection
results are kept under each seed's `evaluation/` directory and are not final
test results.

## Llama-3.1-8B, 8192-prompt Main run

The Llama family replication uses `meta-llama/Llama-3.1-8B-Instruct`, seed 47,
the same frozen 8192-prompt UltraFeedback subset as the Qwen 7B run, LoRA
rank/alpha 16/32, and the four narrow-NLHF methods `Nash-RS`, `Nash-MD`, `MPO`,
and `EGPO`.  Its result namespace is independent of every Qwen run.

Llama is gated on Hugging Face.  Before the first submission, accept Meta's
license and place a read token at `~/.cache/huggingface/token` (or set
`HF_TOKEN_PATH`).  The submitter automatically launches the cache job when no
complete local snapshot exists:

```bash
bash cluster/hopper/main/submit_llama3p1_8b_lora_smokes.sh 47
```

This first launches the cache job when needed, then queues four dependent
one-GPU engineering smokes.  Nash-RS, Nash-MD, MPO, and EGPO each perform two
real optimizer steps with a 64-token generation cap, LoRA rank/alpha 16/32,
and an isolated result namespace.  Each job must export step metrics and prove
a nonzero parameter update before it succeeds.  These are not paper results.
When all user run slots are occupied, the jobs may be queued before the token
is installed with `NASHRS_DEFER_GATED_TOKEN_CHECK=1`; the token must still exist
before the cache job starts.

After all four smokes pass, submit the full 8192-prompt experiment:

```bash
bash cluster/hopper/main/submit_llama3p1_8b_lora_8192_seed47.sh
```

The Llama Nash-MD entry is mathematically identical to TRL's geometric
mixture, but maintains separate policy/reference KV caches and shares the
frozen base model through PEFT's disabled-adapter reference.  MPO streams
metrics after every outer round and writes a restartable checkpoint every 512
optimizer steps.  All four jobs request one H200 for at most 144 hours.

After all four run manifests verify nonzero parameter updates, submit the
shared-base plus four-method fixed evaluation by passing the completed job IDs:

```bash
bash cluster/hopper/main/submit_llama3p1_8b_8192_eval.sh \
  47 NASH_RS NASH_MD MPO EGPO
```
