# Foundation Models HW2 - LLM Alignment

## Overview
Building and aligning an assistant from a base language model, under a single-GPU Colab budget. Starting from Qwen2.5-0.5B (base), the notebook runs the full post-training pipeline: supervised fine-tuning, parameter-efficient fine-tuning, a preference dataset and reward model, Direct Preference Optimization (DPO), and Best-of-N sampling, with every step evaluated against the base model using an LLM-as-judge.

## Key Features
- **SFT:** full fine-tuning of Qwen2.5-0.5B on 3,000 examples from databricks-dolly-15k, with 12 fixed prompts compared before and after (length, termination, repetition, held-out loss).
- **Full FT vs LoRA vs QLoRA:** the same training run in three modes, comparing trainable parameters, peak memory, time and judged quality (PEFT + bitsandbytes 4-bit).
- **Data size and quality ablation:** LoRA on 200 examples, and on 3,000 examples with 30% of the answers corrupted.
- **Preference data and reward model:** 300 new prompts, two SFT samples each, pairwise judging, and a Bradley-Terry reward model on top of embeddings.
- **DPO:** DPO on the SFT model with LoRA (beta = 0.1), compared with SFT on a fixed set and a judged set, including a second judge model to check judge circularity.
- **Best-of-N:** 8 SFT candidates per prompt re-ranked by the reward model (N = 4 and N = 8), compared with single sampling and with DPO, including cost per query.
- **Bonus:** LoRA placement ablation - attention projections only (q, k, v, o) vs all linear layers.

## Repository Content
- `Foundation_Models_HW2_v7.ipynb`: the full notebook - training, evaluation and analysis for all parts.
- `Assignment2.pdf`: the assignment specification.

---
Foundation Models course, M.Sc. in Data Science, Holon Institute of Technology (HIT).
