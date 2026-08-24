# NUS Hopper

The scripts in this directory target the NUS Hopper PBS project
`CFP05-CF-001`.

Submit the minimal one-GPU smoke test from the repository root:

```bash
module load pbs  # only if qsub is not already available
qsub cluster/hopper/smoke_test.pbs
```

Inspect the queued or running job with:

```bash
qstat -awn1
qstat -fx JOB_ID
qgpu_smi JOB_ID
```

Follow the logs from the repository root with:

```bash
tail -f stdout.JOB_ID
tail -f stderr.JOB_ID
```

The job requests one H200 for at most 15 minutes, enters the NUS PyTorch
2.3/CUDA 12.4 container, verifies CUDA, runs the unit tests, and executes the
toy Nash-RS example. PBS configures the assigned GPU automatically; do not set
`CUDA_VISIBLE_DEVICES` in job scripts or Python code.

## OpenRLHF environment

The reproducible training environment is pinned to OpenRLHF 0.9.3 and vLLM
0.15.0. This combination uses the CUDA 12.8 wheel stack compatible with the
Hopper driver and the cluster's `pytorch_2.7.0_cuda_12.8.sif` image. CUTLASS
DSL is pinned to 4.3.4, the minimum accepted by FlashInfer 0.6.1. Setuptools is
pinned to 80.9.0 to satisfy vLLM's Python 3.12 requirement of `>=77,<81`.
H200 inference is validated through vLLM's bundled FlashAttention 3 extension;
the optional flash-attn FA4/CuTe path targets Blackwell and is not required.

Submit the setup job once from the repository root:

```bash
qsub cluster/hopper/setup_openrlhf.pbs
```

The environment is created under
`/scratch/jiancongxiao/venvs/nashrs-openrlhf-0.9.3-vllm-0.15.0`. Package and
model caches, plus the final `pip freeze` manifest, are also kept under
`/scratch/jiancongxiao`. The setup script is safe to rerun after a successful
installation. If an interrupted first installation leaves an incomplete
environment, move that specific environment directory aside before retrying.

## Real-model vLLM smoke test

After the environment setup succeeds, submit:

```bash
qsub cluster/hopper/vllm_generation_smoke.pbs
```

The job downloads the public `Qwen/Qwen2.5-0.5B-Instruct` model into the shared
Hugging Face scratch cache, records its resolved model revision, and generates
two short responses with vLLM on one H200. JSONL output is written under
`/scratch/jiancongxiao/results/vllm-smoke/`.

## Nash-RS real-generation data path

After the generic vLLM smoke test succeeds, submit:

```bash
qsub cluster/hopper/vllm_nash_rs_smoke.pbs
```

This job uses real vLLM generations for the current-policy samples, reference
proposals, accepted Gibbs opponents, and rollout response. It then constructs
the Nash-RS implicit reward with the corrected acceptance rule
`u <= exp(-g_hat/tau)`. The smoke preference oracle is deterministic and will
be replaced by the experiment preference model in the PPO integration.

## Nash-RS real preference model

Submit the real preference-model vertical slice with:

```bash
qsub cluster/hopper/vllm_nash_rs_pm_smoke.pbs
```

It combines Qwen2.5-0.5B vLLM generations with the MIT-licensed
`OpenAssistant/reward-model-deberta-v3-large-v2` scalar reward model. Pairwise
probabilities use the BTL map `sigmoid(r_left-r_right)`. The same oracle API
also supports weighted mixtures of reward models for later non-BT experiments.
The job disables the Xet transport and retries a pinned model snapshot over the
regular Hub HTTP path because Hopper may reset connections to the Xet CAS
endpoint.

## OpenRLHF Nash-RS PPO update

After the real preference-model smoke test succeeds, submit:

```bash
qsub cluster/hopper/openrlhf_nash_rs_ppo_smoke.pbs
```

This one-H200 vertical slice performs one small OpenRLHF PPO run on two prompts.
The custom agent generates the rollout and `B1` samples from the synchronized
current vLLM policy, samples proposals from a separately loaded fixed reference
checkpoint, scores them with the real preference model, and applies the
correct Gibbs rejection event. The final actor is saved under scratch and a
post-run tensor comparison records a nonzero parameter update.

The cluster exports GPU UUIDs while vLLM 0.15 expects numeric device IDs. The
shared launcher resolves the PBS-assigned UUID to the same physical GPU before
entering the container; it never chooses an unallocated device.

## 32-prompt multi-step pilot

After the one-step PPO smoke test succeeds, submit:

```bash
qsub cluster/hopper/openrlhf_nash_rs_pilot.pbs
```

The pilot uses Qwen2.5-0.5B on 32 prompts, with rollout and train batch sizes of
8, yielding four OpenRLHF global steps. The actor learning rate is `1e-6` and
the critic learning rate is `1e-5`; warmup is disabled because the run is only
four steps. It reuses the validated `B1=B2=2` Gibbs sampling configuration.

Outputs are written to:

```text
/scratch/jiancongxiao/results/openrlhf-nashrs-pilot/JOB_ID/
```

The final actor is stored in `actor/`. Per-step metrics are stored as JSONL in
`step_metrics.jsonl`, including reward, acceptance rate, preference-model
calls, KL, GPU-hours, generated tokens, and cumulative cost fields. OpenRLHF
reports executor accounting as per-prompt batch means, so the exporter also
uses the rollout batch size to write `step_total_*` and
`cumulative_total_*` count fields. `sum_sample_gpu_hours` measures summed
sample work; it is distinct from PBS allocated wall-clock GPU-hours when
sample executions overlap.

Rejection accounting distinguishes two rates. `acceptance_rate` is accepted
samples divided by proposals that actually received a rejection decision;
`proposal_efficiency` is accepted samples divided by all reference responses
generated in fixed-size proposal batches. The latter may be lower because the
last batch can contain unused proposals after `B2` opponents have been found.

## 128-prompt pilot

The next scaling check uses 128 unique prompts and a 128-token response budget:

```bash
qsub cluster/hopper/openrlhf_nash_rs_pilot_128.pbs
```

With rollout and train batch sizes of 8, this produces 16 global optimizer
steps. It keeps the actor learning rate at `1e-6`, critic learning rate at
`1e-5`, and `B1=B2=2`. Results are written under
`/scratch/jiancongxiao/results/openrlhf-nashrs-pilot-128/JOB_ID/`.
