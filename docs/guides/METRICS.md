# Metrics Recommendations for QA Evaluation

## 📊 Overview

Choosing the right metrics is crucial for understanding your QA system's performance. Here's a comprehensive guide.

---

## ✅ Currently Implemented Metrics

### 1. **Exact Match (EM)** ⚡ Fast
**What it measures**: Binary - does the answer exactly match (after normalization)?

**Strengths:**
- Simple, interpretable
- Strict evaluation
- Fast computation

**Weaknesses:**
- Too strict - doesn't reward partial correctness
- Misses semantically correct paraphrases

**When to use:**
- Factoid questions with short answers
- When you need strict correctness
- Initial baseline evaluation

**Example:**
```python
Question: "When did WWII end?"
Reference: "1945"
Prediction: "The war ended in 1945" → EM = 0.0 ❌
Prediction: "1945"                    → EM = 1.0 ✅
```

---

### 2. **F1 Score** ⚡ Fast
**What it measures**: Token-level overlap between prediction and reference

**Strengths:**
- More lenient than EM
- Rewards partial matches
- Standard in QA evaluation

**Weaknesses:**
- Doesn't understand semantics
- Can give credit for wrong but overlapping answers

**When to use:**
- Longer answers
- When partial credit makes sense
- Always use alongside EM

**Example:**
```python
Question: "Who was the first US president?"
Reference: "George Washington"
Prediction: "George Washington was the first president" 
→ F1 = 0.67 (2 tokens match / 3 tokens total)
```

---

### 3. **BERTScore** 🐢 Slow
**What it measures**: Semantic similarity using contextualized embeddings

**Strengths:**
- Understands paraphrases
- Semantic evaluation
- Better for descriptive answers

**Weaknesses:**
- Slow (uses neural network)
- Can be overly lenient
- Not great for factoid answers

**When to use:**
- Evaluating longer, descriptive answers
- When paraphrasing is acceptable
- Research/academic evaluation

**Example:**
```python
Question: "What is photosynthesis?"
Reference: "Process where plants convert light to energy"
Prediction: "Plants use sunlight to make energy"
→ BERTScore F1 = 0.92 (semantically similar)
```

---

### 4. **BLEURT** 🐢 Very Slow
**What it measures**: Learned metric trained on human judgments

**Strengths:**
- Correlates well with human evaluation
- Trained on diverse text
- Catches nuanced quality differences

**Weaknesses:**
- Very slow
- Requires checkpoint download
- Black box (hard to interpret)

**When to use:**
- Research/publication
- When you need human-like judgments
- Final evaluation of best models

---

### 5. **Faithfulness** 🎯 RAG-Specific
> **Note:** Faithfulness is fully wired in the metrics pipeline — contexts are automatically extracted from `retrieved_docs` in predictions.json.

**What it measures**: Whether the answer is grounded in retrieved documents

**Strengths:**
- Detects hallucinations
- Critical for RAG evaluation
- Multiple methods (NLI, token-based, LLM)
- Fast with NLI models

**Weaknesses:**
- Requires retrieved context (RAG only)
- NLI models can make mistakes
- Doesn't measure correctness

**When to use:**
- ✅ **Evaluating RAG systems**
- ✅ **Detecting hallucinations**
- When you need to verify context usage
- Comparing parametric vs retrieved knowledge

**Example:**
```python
Question: "When was the Eiffel Tower built?"
Retrieved Context: "The Eiffel Tower was completed in 1889 in Paris."
Answer: "1889"  → Faithfulness = 1.0 ✅ (supported by context)
Answer: "It was built by Gustave Eiffel in Paris"  → Faithfulness = 0.3 ❌ (name not in context)
```

**Methods:**
- `nli`: Natural Language Inference (default, best quality)
- `token_overlap`: Simple token matching (fast baseline)
- `llm`: LLM-based judgment (highest quality, expensive)

**Config:**
```python
FaithfulnessMetric(
    method="nli",  # or "token_overlap", "llm"
    nli_model="microsoft/deberta-base-mnli",
    threshold=0.5
)
```

---

### 6. **Hallucination Detection** 🎯 RAG-Specific
> **Note:** Hallucination detection is fully wired in the metrics pipeline — contexts are automatically extracted from `retrieved_docs` in predictions.json.

**What it measures**: Inverse of faithfulness - detects unsupported claims

**Strengths:**
- Identifies problematic outputs
- Complementary to faithfulness
- Includes contradiction detection
- Multiple detection methods

**Weaknesses:**
- Requires retrieved context (RAG only)
- May have false positives
- Doesn't measure correctness

**When to use:**
- ✅ **Quality control for RAG systems**
- ✅ **Identifying model hallucinations**
- Production monitoring
- Comparing retrieval strategies

**Example:**
```python
Question: "What is Python?"
Retrieved Context: "Python is a programming language."
Answer: "Python is a snake"  → Hallucination = 0.8 ❌ (contradicted by context)
Answer: "Python is a programming language"  → Hallucination = 0.0 ✅ (supported)
```

**Methods:**
- `nli`: NLI-based detection (checks contradiction + lack of entailment)
- `simple`: Token-based heuristic (fast baseline)

**Config:**
```python
HallucinationMetric(
    method="nli",  # or "simple"
    nli_model="microsoft/deberta-base-mnli",
    threshold=0.5
)
```

**Output:**
```json
{
  "hallucination_rate": 0.15,        // Average score
  "hallucinated_ratio": 0.12,        // % above threshold
  "hallucinated_count": 12,          // Number of cases
  "total": 100
}
```

---

### 7. **Answer In Context** ⚡ Fast, RAG-Specific
**What it measures**: Binary check — does the gold answer appear in retrieved context?

**Strengths:**
- No model required (pure text matching)
- Fast, deterministic
- Cheap proxy for retrieval quality

**Weaknesses:**
- Binary (no partial credit)
- Sensitive to answer normalization
- Doesn't verify semantic match

**When to use:**
- Quick retrieval quality diagnostic
- Debugging retrieval failures
- Comparing retrieval strategies

**Example:**
```python
Question: "When did WWII end?"
Retrieved Context: "The war ended in 1945."
Gold Answer: "1945" → answer_in_context = 1.0 ✅
Gold Answer: "September 2, 1945" → answer_in_context = 0.0 ❌ (exact phrase not found)
```

**Config:**
```python
AnswerInContextMetric(min_answer_len=2)  # Skip trivially short answers
```

---

### 8. **Context Recall** ⚡ Fast, RAG-Specific
**What it measures**: Fraction of reference answer sentences found in retrieved context

**Strengths:**
- Graded signal (not just binary)
- No model required
- Word-overlap fallback for paraphrases

**Weaknesses:**
- Sentence-level granularity may miss fine-grained info
- Word overlap threshold is configurable but heuristic

**When to use:**
- Graded retrieval quality assessment
- Diagnosing partial retrieval success
- Works best with longer reference answers

**Example:**
```python
Reference: "The war ended in 1945. It was devastating."
Context: "The war ended in 1945."
→ context_recall = 0.5 (1/2 sentences found)
```

**Config:**
```python
ContextRecallMetric(
    min_sentence_words=3,    # Skip trivial fragments
    overlap_threshold=0.8,   # Word-overlap fallback threshold
)
```

---

### 9. **LLM-as-a-Judge** 💰 Expensive
**What it measures**: Categorical judgment by GPT-4 (correct/partial/incorrect)

**Strengths:**
- Most semantic understanding
- Flexible, adapts to question type
- Can explain reasoning
- **Binary classification for performance**

**Weaknesses:**
- Expensive (API costs)
- Slow
- Requires API key
- Non-deterministic (even with temp=0)

**When to use:**
- ✅ **When you need binary correctness labels**
- ✅ **For building performance metrics**
- When standard metrics disagree
- For analyzing edge cases
- When you have API budget

**Cost estimate:** ~$0.01-0.03 per evaluation (GPT-4)

---

## 🎯 Recommended Metric Combinations

### For RAG Systems 🔥

```python
metrics = [
    ExactMatchMetric(),           # Correctness
    F1Metric(),                   # Partial credit
    BERTScoreMetric(),            # Semantic similarity
    AnswerInContextMetric(),      # Retrieval proxy (fast)
    ContextRecallMetric(),        # Graded retrieval quality (fast)
    FaithfulnessMetric(           # Groundedness
        method="nli"
    ),
    HallucinationMetric(          # Hallucination detection
        method="nli"
    )
]
```

**Why this combination:**
- EM/F1/BERTScore: Measure correctness
- AnswerInContext: Quick check if retrieval found the answer
- ContextRecall: Graded retrieval quality diagnostic
- Faithfulness: Verify answers use retrieved context
- Hallucination: Catch unsupported claims
- **Complete RAG evaluation**: Correctness + Retrieval Quality + Groundedness

**Use for:**
- Comparing retrieval strategies
- Tuning RAG parameters
- Production RAG monitoring
- Research on grounded generation

### For Binary Performance Evaluation

```python
metrics = [
    ExactMatchMetric(),           # Strict baseline
    F1Metric(),                   # Partial credit
    LLMJudgeQAMetric(             # Binary judgment
        judge_model=gpt4,
        judgment_type="binary"    # correct/incorrect
    )
]
```

**Why this combination:**
- EM/F1: Fast, standard baselines
- LLM Judge: High-quality binary labels for performance metrics
- Can compare automated metrics against LLM judgments

### For Quick Iteration (Development)

```python
metrics = [
    ExactMatchMetric(),
    F1Metric(),
]
```

**Why:** Fast, cheap, good enough for rapid experimentation

### For Publication/Research

```python
metrics = [
    ExactMatchMetric(),
    F1Metric(),
    BERTScoreMetric(),
    BLEURTMetric(),
    LLMJudgeQAMetric(judge_model=gpt4, judgment_type="ternary")
]
```

**Why:** Comprehensive, covers all aspects, publishable

### For Production Monitoring

```python
metrics = [
    ExactMatchMetric(),
    F1Metric(),
    LLMJudgeQAMetric(           # Sample 10% of traffic
        judge_model=gpt4,
        judgment_type="binary"
    )
]
```

**Why:** Fast automated metrics + sampled LLM judgments for quality control

---

## 🔧 Adding Metrics to an Existing Study

After running a study, you can compute additional metrics without re-running generation. The `compute-metrics` command iterates all experiments in a study directory, checks which ones are missing the requested metrics, and appends scores to each experiment's `predictions.json` (per-item) and `results.json` (aggregate).

### Usage

```bash
# Add new metrics to all experiments in a study
uv run ragicamp compute-metrics outputs/your_study -m answer_in_context,context_recall

# Preview which experiments need metrics (no computation)
uv run ragicamp compute-metrics outputs/your_study -m answer_in_context,context_recall --dry-run

# Force recompute even if metric already exists
uv run ragicamp compute-metrics outputs/your_study -m faithfulness --force

# Add LLM judge (requires API key)
uv run ragicamp compute-metrics outputs/your_study -m llm_judge_qa \
    --judge-model gpt-4o-mini

# Single experiment only
uv run ragicamp metrics outputs/your_study/exp_name -m answer_in_context,context_recall
```

### How it works

1. Scans all experiment directories under the study output
2. For each experiment, checks `state.json` to see which metrics are already computed
3. Skips experiments that already have all requested metrics (unless `--force`)
4. Loads `predictions.json`, extracts `retrieved_docs` as contexts
5. Calls `compute_metrics_batched()` with only the missing metrics
6. Appends per-item scores to each prediction and merges aggregate scores into `results.json`

### Context-aware metrics

Metrics that evaluate retrieval quality (`faithfulness`, `hallucination`, `answer_in_context`, `context_recall`) automatically extract context from the `retrieved_docs` field in predictions. For `direct_llm` experiments (no retrieval), contexts are empty and these metrics return 0.0.

### Typical workflow

```bash
# 1. Run study with fast metrics only
#    (study YAML has: metrics: [f1, exact_match])
uv run ragicamp run conf/study/my_study.yaml

# 2. Add retrieval quality metrics (no GPU needed, fast)
uv run ragicamp compute-metrics outputs/my_study -m answer_in_context,context_recall

# 3. Add NLI-based metrics (needs GPU for deberta model)
uv run ragicamp compute-metrics outputs/my_study -m faithfulness,hallucination

# 4. Add LLM judge (needs API key, costs money)
uv run ragicamp compute-metrics outputs/my_study -m llm_judge_qa \
    --judge-model gpt-4o-mini

# 5. Add expensive model-based metrics
uv run ragicamp compute-metrics outputs/my_study -m bertscore,bleurt
```

Each step appends to the existing results — no data is lost or overwritten.

---

## 📈 Additional Metrics to Consider

### 6. **ROUGE-L** (Not yet implemented)
**What:** Longest common subsequence

**Good for:** Summarization, longer answers

**Implementation:**
```python
from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(predictions, references)
```

### 7. **Answer Relevance** (Not yet implemented)
**What:** Is the answer relevant to the question?

**Good for:** Detecting hallucinations

**Implementation:** Use semantic similarity between question and answer

### 8. **Answer Length** (Easy to add)
**What:** Token count of answer

**Good for:** Detecting overly verbose or terse answers

### 10. **Confidence Score** (Model-specific)
**What:** Model's confidence in its answer

**Good for:** Calibration, uncertainty estimation

---

## 💡 Best Practices

### 1. Always Use Multiple Metrics
```python
# ✅ Good - Multiple perspectives
metrics = [EM, F1, BERTScore]

# ❌ Bad - Only one metric
metrics = [EM]
```

### 2. Match Metrics to Answer Type

| Answer Type | Best Metrics |
|-------------|-------------|
| Short factoid (dates, names) | EM, F1, **LLM Judge** |
| Long descriptive | F1, BERTScore, BLEURT, **LLM Judge** |
| Yes/No | EM, **LLM Judge** |
| Numerical | EM with normalization |
| Lists | F1, custom list metrics |
| **RAG outputs** | **EM/F1 + AnswerInContext + ContextRecall + Faithfulness + Hallucination** |

### 3. Understand Trade-offs

| Priority | Use This |
|----------|----------|
| **Speed** | EM, F1 |
| **Accuracy** | BERTScore, BLEURT, **LLM Judge** |
| **Cost** | EM, F1 (free) |
| **Interpretability** | EM, **LLM Judge** (explains) |
| **Binary classification** | **LLM Judge** (best) |
| **RAG evaluation** | **Faithfulness + Hallucination** (essential) |
| **Grounded generation** | **Faithfulness** (NLI method) |

### 4. Validate Metrics Against Human Judgment

```python
# Compute correlations
human_scores = [...]
metric_scores = [...]

from scipy.stats import pearsonr
correlation, p_value = pearsonr(human_scores, metric_scores)
print(f"Correlation: {correlation:.3f} (p={p_value:.4f})")
```

---

## 🚀 Using LLM-as-a-Judge (Your Use Case)

### Setup

```bash
# Set API key
export OPENAI_API_KEY='your-key-here'

# Run example
uv run python examples/llm_judge_evaluation.py
```

### Code Example

```python
from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric
from ragicamp.models.openai import OpenAIModel

# Create GPT-4 judge
judge_model = OpenAIModel("gpt-4o", temperature=0.0)

# Binary judge (correct/incorrect)
binary_judge = LLMJudgeQAMetric(
    judge_model=judge_model,
    judgment_type="binary"
)

# Use in evaluation
metrics = [
    ExactMatchMetric(),
    F1Metric(),
    binary_judge  # Adds binary correctness labels
]

evaluator = Evaluator(agent, dataset, metrics)
results = evaluator.evaluate()

# Get binary performance
print(f"LLM Judge Accuracy: {results['llm_judge_qa']:.2%}")
print(f"Correct: {results['llm_judge_qa_correct']:.2%}")
print(f"Incorrect: {results['llm_judge_qa_incorrect']:.2%}")
```

### Ternary Judge (correct/partial/incorrect)

```python
ternary_judge = LLMJudgeQAMetric(
    judge_model=judge_model,
    judgment_type="ternary"
)

# Results include:
# - llm_judge_qa_correct
# - llm_judge_qa_partial
# - llm_judge_qa_incorrect
```

---

## 💰 Cost Considerations

### Per-Question Cost Estimates

| Metric | Cost | Speed |
|--------|------|-------|
| EM, F1 | Free | ⚡ Instant |
| BERTScore | Free | 🐢 ~0.5s/q |
| BLEURT | Free | 🐢 ~1s/q |
| **LLM Judge (GPT-4)** | **~$0.01-0.03/q** | 🐢 ~2s/q |
| **LLM Judge (GPT-4-mini)** | **~$0.001/q** | ⚡ ~1s/q |

### Budget Examples

**100 questions:**
- EM/F1: $0 (free)
- BERTScore: $0 (free, ~1 min)
- LLM Judge (GPT-4): $1-3
- LLM Judge (GPT-4-mini): $0.10

**Strategy:** Use LLM judge on subset for validation
```python
# Evaluate 100 questions with EM/F1
# Sample 20 for LLM judge validation
import random
validation_indices = random.sample(range(100), 20)
```

---

## 📊 Interpreting Results

### Good Performance Indicators

```
Exact Match:     0.35-0.45  (Natural Questions benchmark)
F1:              0.50-0.60  
BERTScore F1:    0.85-0.90
LLM Judge:       0.70-0.85  (binary correct rate)
```

### Red Flags

```
⚠️ EM high, F1 low       → Answers too short
⚠️ F1 high, EM low       → Close but not exact
⚠️ BERTScore high, EM/F1 low → Semantic match but wrong facts
⚠️ LLM Judge disagrees with all → Review edge cases
```

---

## ✅ Summary: Your Binary Performance Metric

**Recommended Setup:**

1. **Primary metrics** (fast, cheap):
   - Exact Match
   - F1 Score

2. **Binary performance metric** (your goal):
   - LLM Judge (GPT-4) with `judgment_type="binary"`
   - Returns: correct (1.0) or incorrect (0.0)
   - Use for: Performance classification, error analysis

3. **Optional** (for deeper analysis):
   - BERTScore (semantic similarity)
   - Ternary LLM Judge (correct/partial/incorrect)

**Next steps:**
1. Run: `uv run python examples/llm_judge_evaluation.py`
2. Check outputs for per-question binary judgments
3. Build your performance metrics from binary labels

🎯 **This gives you high-quality binary labels for building performance evaluation!**

