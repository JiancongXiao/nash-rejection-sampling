# Tulu-3-8B-SFT × full PKU-SafeRLHF on one 8×B200 node

This directory is a portable, restart-aware experiment suite for the paper's
**Main** comparison. It creates exactly five scheduler jobs:

1. Nash-RS training;
2. method-native Nash-MD training;
3. method-native EGPO training;
4. method-native MPO training;
5. a dependent evaluation of the initial model and all four trained models.

Every job requests one exclusive node with eight GPUs. The four training jobs
use synchronous data parallelism across all eight B200s. Nash-RS uses the
validated eight-GPU OpenRLHF/Ray colocated layout; Nash-MD and EGPO retain their
published Trainer update rules under Accelerate DDP; MPO retains its two-inner-
step update in a small DDP wrapper. The evaluation generates the five policies
in parallel on five GPUs and uses the same direct pairwise preference model for
all comparisons.

## Frozen experiment contract

- Policy: `allenai/Llama-3.1-Tulu-3-8B-SFT`, revision
  `f2a0b46b0cfda21003c6141b1ff837b7e165524d`.
- Data: all unique, train/test-disjoint PKU-SafeRLHF training prompts whose
  Tulu-rendered prompt is at most 128 tokens. Original PKU responses and labels
  are not training targets.
- Preference model: the direct two-class EGPO Gemma-2 PKU preference model,
  revision `f57f706...`; it is not converted to a scalar/BT reward.
- Seed 47, LoRA rank 16, global prompt batch 64, maximum response length 512.
- Nash-RS: `tau=0.5`, `B1=B2=2`, proposal batch `K=8`, PM batch 8, and no
  forced acceptance (`max_proposals=4096`).
- Evaluation: 256 disjoint held-out test prompts, greedy generation, all ten
  unordered model pairs, and 10,000 prompt-bootstrap samples.

The preparation manifest records the exact number and SHA-256 of the full
training set. Historically this filter yielded 32,223 training prompts, but the
scripts treat the frozen manifest—not a hard-coded count—as authoritative.

## One-time setup

```bash
git checkout codex/b200-pku-tulu-suite
cp cluster/b200/config.env.example cluster/b200/config.env
# Edit only filesystem paths and scheduler account/partition fields.
source cluster/b200/config.env
bash cluster/b200/setup_env.sh
bash cluster/b200/prepare.sh
bash cluster/b200/doctor.sh
```

Run setup and the doctor inside an allocation exposing all eight B200s. The
machine needs Python 3.12, CUDA 12.8-compatible drivers, Git, and internet
access during setup. If `flash-attn==2.8.3` cannot compile on the site's
Blackwell toolchain, set `NASHRS_SKIP_FLASH_ATTN_BUILD=1`; vLLM 0.15 retains its
own supported attention backend. Do not change package versions silently.

Tulu is public, but Hugging Face rate limits may require `HF_TOKEN`. Never put
the token in `config.env` or commit it.

## Submit and monitor

First inspect the exact commands:

```bash
bash cluster/b200/submit_all.sh --scheduler auto --dry-run
```

Then create the five jobs:

```bash
bash cluster/b200/submit_all.sh --scheduler auto
```

`auto` detects Slurm, PBS Pro, or a scheduler-free single node. In local mode
the four training runs execute sequentially because only eight GPUs exist. On
Slurm/PBS the four training jobs may queue independently, and the evaluation is
submitted with an `afterok` dependency on all four. Site-specific scheduler
arguments belong in `config.env`.

Final results are written to:

```text
$NASHRS_RESULT_ROOT/seed47/evaluation/summary.json
```

The submission IDs are also stored under
`$NASHRS_RESULT_ROOT/submissions/`. A completed method is not overwritten.

## What must be reported

Report the exact Git commit, five job IDs, node/GPU inventory, data manifest,
run manifests, adapter verification files, scheduler walltime/GPU-hours, and
the full evaluation `summary.json`. A failure must be diagnosed and resumed;
do not replace the model, data, preference oracle, method, seed, or frozen
hyperparameters to make a job pass.
