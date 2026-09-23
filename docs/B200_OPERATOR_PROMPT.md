# Prompt for the 8×B200 operator's Codex / Claude Code session

Copy everything below into a new coding-agent session on the B200 server.

---

You are operating a server/cluster with access to one node containing eight
NVIDIA B200 GPUs. Your task is to reproduce and complete a five-job experiment
from Jiancong Xiao's Nash-RS repository. Work autonomously, but never expose or
commit credentials and never change the frozen scientific settings merely to
make a run finish.

Goal: train four policies from `allenai/Llama-3.1-Tulu-3-8B-SFT` on the full
prompt-only PKU-SafeRLHF training set, using seed 47 and one 8×B200 node per
training job. The methods are Nash-RS, method-native Nash-MD, method-native
EGPO, and method-native MPO. Submit a fifth 8-GPU evaluation job that waits for
all four training jobs, compares the initial model plus the four trained
policies on the frozen held-out set, and produces confidence intervals.

Proceed as follows:

1. Inspect this machine first. Identify the scheduler (Slurm, PBS Pro, or no
   scheduler), account/partition or queue requirements, shared high-capacity
   filesystem, Python/module/container availability, CUDA driver, and whether
   eight B200 GPUs can be allocated on one node. Ask me only for information
   that cannot be discovered locally, such as an account name, filesystem path,
   or Hugging Face token. Do not assume paths from the NUS Hopper cluster.

2. Clone `https://github.com/JiancongXiao/nash-rejection-sampling.git`, fetch
   branch `codex/b200-pku-tulu-suite`, check it out, and record the exact commit.
   Read `cluster/b200/README.md` completely before acting.

3. Copy `cluster/b200/config.env.example` to `cluster/b200/config.env`. Change
   only site paths and scheduler submission arguments. Keep all frozen model,
   data, algorithm, seed, batch, LoRA, Nash-RS, and evaluation settings exactly
   as committed. Put cache, environments, datasets, results, and external
   sources on shared persistent storage with enough space. Never write an HF
   token into the config or Git history.

4. Obtain one interactive eight-B200 allocation for setup/validation (or use an
   equivalent site-approved setup job). Authenticate to Hugging Face only if
   necessary. Run `cluster/b200/setup_env.sh`, then
   `cluster/b200/prepare.sh`, then `cluster/b200/doctor.sh`. The setup creates
   the same OpenRLHF 0.9.3 / vLLM 0.15 environment and native TRL 0.13
   environment used in the original experiments. Resolve infrastructure and
   Blackwell compatibility problems without changing algorithmic settings.
   If the optional flash-attn build alone is incompatible, use the documented
   `NASHRS_SKIP_FLASH_ATTN_BUILD=1` fallback and record it.

5. Inspect `cluster/b200/submit_all.sh --scheduler auto --dry-run`. Confirm that
   it will create exactly four independent 8-GPU training jobs and one 8-GPU
   evaluation job with an after-success dependency on all four. Then submit the
   real jobs. Immediately report all five job IDs and the submission-manifest
   path.

6. Monitor efficiently until the evaluation finishes. Check job state, the
   end of each log, GPU utilization, checkpoints, and run manifests. Resume
   recoverable failures from their existing output directories. You may fix
   genuine portability/infrastructure bugs in a new Git commit, but first
   explain the root cause; do not alter the method, model, preference model,
   dataset, seed, LoRA rank, global batch, learning rates, `tau=0.5`, `B1=B2=2`,
   `K=8`, PM batch 8, max proposals 4096, generation length, or evaluation
   protocol. Do not silently fall back to fewer GPUs or a partial dataset.

7. When complete, send me:
   - exact Git commit and any portability patch commit;
   - scheduler and environment/package summary;
   - five job IDs, final states, nodes, elapsed walltime, and B200 GPU-hours;
   - frozen dataset count and SHA-256 from the data manifest;
   - confirmation that every adapter has a nonzero verified update;
   - a compact table for initial, Nash-RS, Nash-MD, EGPO, and MPO containing
     average win rate, worst-case win rate, empirical exploitability, response
     length, truncation rate, and 95% confidence intervals;
   - the complete contents and path of
     `$NASHRS_RESULT_ROOT/seed47/evaluation/summary.json`;
   - an archive or accessible paths for manifests, metrics, logs, adapters, and
     evaluation responses.

Do not stop after submission. Continue monitoring and repairing within these
constraints until the evaluation result exists, or until a blocker genuinely
requires my action.

---
