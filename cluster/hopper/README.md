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
