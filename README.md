# Foundation Models HW2 - LLM Alignment

**Home Assignment 2 (M.Sc. Data Science, HIT). Building and aligning an assistant from Qwen2.5-0.5B (base) on a single Colab GPU — supervised fine-tuning, parameter-efficient fine-tuning, preference data and a reward model, DPO and Best-of-N, each judged against the base model. How far can post-training move a small base model, and what does each step cost?**

## Headline Results
- **SFT did most of the work:** the base model never stopped on its own; after SFT **100%** of answers terminate, about **5×** shorter, with no junk text. Measured directly, no judge needed.
- **LoRA matches Full FT:** judged **0.50** head to head, while training **1.8%** of the parameters with **43%** less peak memory (8.7 vs. 15.2 GiB).
- **QLoRA did not pay off at 0.5B:** same quality as LoRA, and the slowest run (411 s vs. 323 s).
- **DPO was the only decisive win:** **0.66** preference over SFT, same direction under a second judge (Granite) and on both seeds.
- **Best-of-N did not help:** the reward model stayed at chance (held-out 15/30 from 190 decisive pairs), so Best-of-8 cost **7×** the tokens for no gain (0.44).
- **Robustness check:** five comparisons flipped between evaluation seeds; only results that held across seeds, judges and runs are treated as conclusions.

## Key Features
- **Supervised Fine-Tuning:** full fine-tuning on 3,000 databricks-dolly-15k examples, with 12 fixed prompts compared before and after (length, termination, repetition, held-out loss).
- **Full FT vs. LoRA vs. QLoRA:** the same run in three modes, compared by trainable parameters, peak memory, time and judged quality.
- **Data Size & Quality:** LoRA on 200 examples, and on 3,000 examples with 30% of the answers replaced.
- **Preference Data & Reward Model:** 300 new prompts, two SFT samples each, pairwise LLM judging and a Bradley-Terry reward model over embeddings.
- **DPO:** DPO with LoRA on top of the SFT model (beta = 0.1), against SFT on a fixed and a judged set, with a second judge model to check judge circularity.
- **Best-of-N:** 8 SFT candidates per prompt re-ranked by the reward model (N = 4 and N = 8), against single sampling and DPO, including cost per query.
- **Bonus - LoRA Placement:** attention projections only (q, k, v, o) vs. all linear layers under the same conditions.

## Repository Structure
- `Foundation_Models_HW2.ipynb`: Full solution notebook: all six parts, the bonus and the training-and-alignment profile (explanations in Hebrew).
- `Assignment2.pdf`: Original assignment instructions.
