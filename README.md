# Foundation Models HW2 - LLM Alignment

**Home Assignment 2 (M.Sc. Data Science, HIT). Building and aligning an assistant from Qwen2.5-0.5B (base) on a single Colab GPU — supervised fine-tuning, parameter-efficient fine-tuning, preference data and a reward model, DPO and Best-of-N, each judged against the base model. How far can post-training move a small base model, and what does each step cost?**

## Key Features
- **Supervised Fine-Tuning:** full fine-tuning on 3,000 databricks-dolly-15k examples, with 12 fixed prompts compared before and after (length, termination, repetition, held-out loss).
- **Full FT vs. LoRA vs. QLoRA:** the same run in three modes, compared by trainable parameters, peak memory, time and judged quality.
- **Data Size & Quality:** LoRA on 200 examples, and on 3,000 examples with 30% of the answers replaced.
- **Preference Data & Reward Model:** 300 new prompts, two SFT samples each, pairwise LLM judging and a Bradley-Terry reward model over embeddings.
- **DPO:** DPO with LoRA on top of the SFT model (beta = 0.1), against SFT on a fixed and a judged set, with a second judge model to check judge circularity.
- **Best-of-N:** 8 SFT candidates per prompt re-ranked by the reward model (N = 4 and N = 8), against single sampling and DPO, including cost per query.
- **Bonus - LoRA Placement:** attention projections only (q, k, v, o) vs. all linear layers under the same conditions.

## Repository Structure
- `Foundation_Models_HW2_v7.ipynb`: Full solution notebook — all six parts and the bonus (explanations in Hebrew).
- `Assignment2.pdf`: Original assignment instructions.
