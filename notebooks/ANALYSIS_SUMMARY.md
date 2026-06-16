# RAGiCamp Analysis Summary

**Study:** `smart_retrieval_slm` | **Date:** 2026-02-20 | **Primary Metric:** F1

---

## Study at a Glance

| Stat | Value |
|------|-------|
| Total experiments | 1,035 (990 complete, 6 failed, 32 in-progress) |
| After dedup + broken-model filter | 710 |
| Models | 7 (Phi-3-mini excluded, mean F1 = 0.017) |
| Datasets | 3 (NQ, TriviaQA, HotpotQA) |
| Agent types | 4 (direct_llm, fixed_rag, iterative_rag, self_rag) |
| RAG experiments | 668 |
| Direct experiments | 42 |
| Overall F1 | mean = 0.233, std = 0.177 |
| Fail rate | 0.9% (7/756) |

---

## Notebook Status

| # | Notebook | Status | Key Finding |
|---|----------|--------|-------------|
| 01 | Study Overview | OK | Gemma2-9B leads (F1=0.40), 7 viable models, 100% core metric coverage |
| 02 | RAG vs Direct | OK | Overall Cohen's d = -0.08 (negligible); RAG helps 49% / hurts 50% of configs |
| 03 | Component Analysis | OK | model+dataset explain 83% of variance; universal recipe identified |
| 04 | Agent Strategies | OK | Advanced agents beat fixed RAG in 86% of scenarios (+0.028 mean delta) |
| 05 | Model Scaling | OK | Small+RAG matches Medium+Direct; tiny+RAG achieves 1.19x compensation |
| 06 | When RAG Fails | OK | 15 questions where RAG consistently hurts (>80% configs) |
| 07 | Error Analysis | OK | 53.6% of questions are "hard"; Qwen2.5-7B avg 101 words/answer |
| 08 | TPE Optimization | OK | 751 trials converged; dataset(42%) + model(41%) dominate importance |
| 09 | Efficiency | BLOCKED | No timing data in results.json |
| 10 | Statistical Tests | OK | Most component differences are negligible; only model tier is significant |
| 11 | Retrieval Diagnostics | OK | 53,852 question-level retrieval records analyzed |

---

## Key Findings

### 1. Model Leaderboard (NB01)

| Model | Tier | Params | Mean F1 | Best F1 |
|-------|------|--------|---------|---------|
| Gemma2-9B | medium | 9B | 0.397 | 0.678 |
| Mistral-7B | medium | 7B | 0.340 | 0.652 |
| Gemma2-2B | tiny | 2B | 0.296 | 0.571 |
| Llama-3.2-3B | small | 3B | 0.272 | 0.599 |
| Qwen2.5-3B | small | 3B | 0.147 | 0.390 |
| Qwen2.5-1.5B | tiny | 1.5B | 0.133 | 0.387 |
| Qwen2.5-7B | medium | 7B | 0.066 | 0.251 |

Qwen2.5-7B underperforms its tier badly (worse than 1.5B Qwen). Likely a prompt/instruction-following issue.

### 2. RAG Effectiveness (NB02, NB05, NB10)

**Overall RAG vs Direct (NB02, dataset-stratified):**
- Overall Cohen's d = **-0.081 (negligible)** -- RAG does not reliably help on average
- Per dataset: hotpotqa d=-0.04, nq d=+0.23, triviaqa d=-0.08
- RAG helps in **49%** of configs, hurts in **50%**, neutral 0%
- When RAG helps: mean uplift = +0.049 F1
- When RAG hurts: mean loss = -0.030 F1
- Best RAG uplift: +0.293 | Worst RAG penalty: -0.579

**RAG success factors (NB02):**
- Best prompts: fewshot_3 (70% helps), concise_strict (65%), fewshot_1 (64%)
- Worst prompts: cot (0% helps, -0.181 mean), cot_final (0%, -0.190), extractive (0%, -0.103)
- Best retriever: hybrid (69% helps), hierarchical (60%), dense (45%)
- Top-K sweet spot: 15 (58% helps), 10 (49%)

**Best-RAG vs Best-Direct (NB10, section 6):**
- RAG beats direct in **20/21** model+dataset combos
- Mean RAG advantage: **+0.072 F1 (+44.3%)**
- By tier: tiny +0.118, small +0.083, medium +0.034
- Smaller models benefit proportionally more from RAG

**Statistical significance (NB10, section 1):**
- RAG significantly helps in only **2/21** combos (after Bonferroni correction)
- RAG significantly hurts in **2/21** combos
- 17/21 show no significant difference (high variance within configs)

**Study question: Can Small+RAG match Medium+Direct? (NB05)**

| Comparison | Compensation Ratio | Verdict |
|------------|-------------------|---------|
| Tiny+RAG vs Small+Direct | 1.19x | YES |
| Small+RAG vs Medium+Direct | 1.53x | YES |

Both tiers exceed the next tier's direct baseline when given good retrieval.

### 3. What Drives Performance? (NB03, NB08)

**Variance decomposition (NB08 TPE importance):**

| Factor | Importance |
|--------|-----------|
| Dataset | 41.9% |
| Model | 40.6% |
| Prompt | 14.1% |
| Retriever | 1.8% |
| Agent type | 0.8% |
| Top-K | 0.3% |
| Reranker | 0.3% |
| Query transform | 0.2% |

Model and dataset dominate. RAG component choices (retriever, reranker, query transform) collectively explain only ~2.3% of variance.

**Universal best recipe (NB03, 3/3 datasets agree):**
- Retriever: dense (BGE-large)
- Reranker: bge-v2
- Prompt: fewshot_1
- Query transform: iterative
- Top-K: 10
- Agent: iterative_rag

### 4. Component Significance Tests (NB10)

Almost no component choice reaches statistical significance:

| Comparison | Cohen's d | Significant? |
|------------|-----------|-------------|
| Reranker: bge-v2 vs bge | 0.05 (negligible) | No |
| Reranker: none vs bge | 0.13 (negligible) | No |
| Agent: iterative vs fixed | 0.01 (negligible) | No |
| Agent: self_rag vs fixed | -0.11 (negligible) | No |
| Query: multiquery vs hyde | 0.09 (negligible) | No |
| Tier: medium vs tiny | 0.37 (small) | Yes* |
| Tier: medium vs small | 0.38 (small) | Yes** |

Only model tier comparisons reach significance, and even those are "small" effect sizes.

### 5. Agent Strategies (NB04)

Advanced agents beat best fixed_rag in **18/21** (86%) model+dataset scenarios.
- Mean delta when wins: +0.028
- Mean delta when loses: -0.042
- iterative_rag is the most frequent winner

Notable: Qwen2.5-1.5B on TriviaQA is the biggest loss (-0.109), where fixed_rag substantially outperforms iterative_rag.

### 6. Synergistic Component Combinations (NB03)

**Best synergies (positive interaction effects):**
- concise + hyde: +0.103
- extractive_quoted + hyde: +0.100
- hybrid + bge reranker: +0.083
- no_reranker + multiquery: +0.069
- hierarchical + no_reranker: +0.057

**Worst redundancies (negative interaction):**
- bge reranker + iterative query: -0.043
- hybrid + GTE-Qwen2-1.5B embedding: -0.055

### 7. Error Taxonomy (NB07)

| Error Type | % of All Predictions |
|------------|---------------------|
| Wrong answer | 28.0% |
| Correct | 24.6% |
| Partial match | 24.4% |
| Over verbose | 12.9% |
| Refusal/empty | 9.4% |
| Refusal/hedging | 0.8% |

**Answer verbosity varies wildly by model:**
- Qwen2.5-7B: 101 words avg (over-generates, hurts F1)
- Gemma2-9B: 5.2 words avg (concise, best F1)

Verbosity anti-correlates with F1 for Qwen models (rho = -0.19) but positively correlates for Gemma (rho = +0.27), suggesting Gemma benefits from slightly longer answers while Qwen over-generates noise.

### 8. Question Difficulty (NB07)

- Easy: 0% (no question is trivially answered by all configs)
- Discriminating: 46.4% (answered correctly by some configs)
- Hard: 53.6% (answered incorrectly by most configs)
- Inter-model agreement on difficulty: rho = 0.688

### 9. When RAG Fails (NB06)

- 15 questions where RAG consistently hurts (>80% of configs)
- 13 questions where RAG consistently helps (>80% of configs)
- Ceiling effect test: no significant correlation (rho = -0.28, p = 0.22)
- Worst case: TriviaQA Q963 (seat belt invention date), RAG delta = -0.549

RAG failure is not simply a "hard question" issue -- some questions are misleading when context is provided (retrieved docs contain plausible but incorrect information).

### 10. TPE Convergence (NB08)

- 751 trials completed
- Best F1: 0.678 (Gemma2-9B, TriviaQA, self_rag, fewshot_1, bge-v2)
- 6/7 models converged; 1 still improving
- Study is converged overall (last 25% no better than first 75% after normalization)

---

## Recommendations

### High Priority (strengthen the analysis)

1. **NB02 reveals a narrative tension -- resolve it.** NB02 shows overall Cohen's d = -0.08 (RAG barely helps on average, 49% helps / 50% hurts). But NB10 shows best-RAG beats best-direct in 20/21 combos (+44%). The difference: *average* RAG config is no better, but *optimized* RAG is much better. This distinction is the core story -- make it explicit with a dedicated comparison cell.

2. **Prompt choice is the hidden driver.** NB02 success factors reveal that CoT and extractive prompts have 0% RAG help rate (mean penalties of -0.10 to -0.19). Meanwhile fewshot prompts help 64-70% of the time. This is likely the biggest actionable finding -- prompt matters more than retriever/reranker -- but it's buried in the success factors table. Elevate it.

3. **Qwen2.5-7B investigation.** This 7B model performs worse than the 1.5B Qwen. The error analysis (NB07) shows it averages 101 words/answer. Hypostudy: it fails to follow concise answer instructions. Consider:
   - Adding a dedicated cell in NB07 showing Qwen2.5-7B's error type distribution vs other models
   - Testing if a stricter prompt (concise_strict) helps it specifically

4. **Address the "nothing is significant" story.** NB10 shows almost no component comparison reaches significance. This is actually a key finding: *model choice and dataset matter; RAG component tuning barely matters*. Frame this explicitly rather than letting readers wonder if the study was underpowered.

5. **Add timing instrumentation** to unblock NB09. Without cost/latency data, you can't argue that small+RAG is *practically* better than large+direct (it could be slower/more expensive).

### Medium Priority (improve presentation)

6. **NB01 metric correlations are suspicious.** faithfulness and hallucination show rho=nan (too few data points: only 62/749 experiments). bertscore shows rho=0.500 for all variants, which looks like a computation artifact (possibly only 3 data points). Investigate whether the Spearman correlation is computed on per-model aggregated values (too few points) rather than per-experiment values.

7. **Consolidate the "universal recipe" message.** NB03 cell-22 and NB08 cell-14 both identify optimal configs but show slightly different answers (NB03 says "iterative" query transform is universal, NB08 shows mode=none at 38%). Reconcile or explain the difference (NB03 uses grid data, NB08 uses TPE trials).

8. **NB11 outputs are image-only.** The retrieval diagnostics tables render as images, making it hard to extract numbers programmatically. Consider adding print() statements for key summary statistics alongside the plots.

### Low Priority (nice to have)

9. **Cross-notebook navigation.** Add a "See also" footer to each notebook linking to related notebooks (e.g., NB05 -> NB10 for statistical backing of scaling claims).

10. **NB06 case studies.** The seat belt question case study is compelling. Add 1-2 more case studies showing different failure modes (e.g., a question where retrieved context is correct but the model ignores it).

11. **Confidence intervals on the study claim.** NB05 says small+RAG beats medium+direct, but doesn't show uncertainty. Use bootstrap CIs from NB10 to put error bars on the compensation ratios.

---

## Headline Takeaways (for presentation)

1. **Average RAG is a coin flip; optimized RAG is transformative.** Across all configs, RAG helps 49% and hurts 50% (Cohen's d = -0.08). But best-RAG beats best-direct in 20/21 model+dataset combos at +44% F1. The gap is configuration quality, not RAG itself.

2. **Small+RAG > Large+Direct.** A 1.5-3B model with optimized RAG exceeds a 7-9B model without retrieval (compensation ratio 1.2-1.5x). Tiny models gain +118% from RAG vs +34% for medium models.

3. **Model choice >> RAG tuning.** Model and dataset explain 83% of variance. All RAG components combined (retriever, reranker, query transform, top-K) explain ~2.3%.

4. **Prompt is the silent killer.** CoT and extractive prompts have a 0% RAG help rate (penalties of -0.10 to -0.19 F1). Fewshot prompts help 64-70% of the time. Getting the prompt right matters more than any retrieval component.

5. **The practical recipe is simple.** Dense retriever (BGE-large) + bge-v2 reranker + fewshot prompt + top-K=10. Advanced components (HyDE, multi-query, hierarchical retrieval) don't reliably help. No component comparison reaches statistical significance after correction.

6. **RAG can hurt.** 15 questions are consistently harmed by retrieval. Verbose models (Qwen2.5-7B) are especially vulnerable -- over-generation combined with noisy context compounds errors.
