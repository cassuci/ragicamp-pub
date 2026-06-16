# Troubleshooting Guide

Common issues and solutions for RAGiCamp.

## Quick Fixes

| Issue | Solution |
|-------|----------|
| Experiment crashed | Just re-run - experiments auto-resume |
| Check status | `ragicamp health outputs/my_study` |
| CUDA/GPU not found | Use `load_in_4bit: true` or OpenAI models |
| Out of memory | Use 4-bit quantization or reduce batch size |
| Metrics missing | `ragicamp metrics <exp_dir> -m f1,llm_judge` |

---

## Experiment Issues

### Experiment Crashed Mid-Run

**Symptom:** Experiment failed during generation or metrics computation.

**Solution:** Just run again - experiments automatically resume:
```bash
# Check status first
uv run ragicamp health outputs/my_study

# Re-run (will resume from checkpoint)
uv run ragicamp run conf/study/my_study.yaml
```

### Need to Recompute Metrics Only

**Symptom:** Predictions are done but metrics need to be recomputed.

**Solution:**  
```bash
# Recompute specific metrics
uv run ragicamp metrics outputs/my_study/exp_name -m f1,llm_judge
```

### Experiment Shows "Failed" Status

**Symptom:** `ragicamp health` shows ✗ Failed

**Solution:**
```bash
# Check the error
cat outputs/my_study/exp_name/state.json

# Use --force to retry (if using study runner)
# Or delete state.json and re-run
rm outputs/my_study/exp_name/state.json
uv run ragicamp run conf/study/my_study.yaml
```

---

## OpenAI API Issues

### Error: "max_tokens is not supported with this model"

**Symptom:**
```
Unsupported parameter: 'max_tokens' is not supported with this model.
Use 'max_completion_tokens' instead.
```

**Solution:** Already fixed in v0.3.0. Update to latest version:
```bash
git pull
uv sync
```

### Error: "temperature does not support 0.0 with this model"

**Symptom:** Newer OpenAI models (o1, o3, gpt-5) don't support temperature parameter.

**Solution:** Already fixed in v0.3.0. The code automatically detects these models and skips unsupported parameters.

### Error: "OPENAI_API_KEY not set"

**Solution:**
```bash
export OPENAI_API_KEY='your-api-key'
```

---

## Memory Issues

### CUDA Out of Memory

**Solutions:**

1. Use 4-bit quantization:
```yaml
# In study config
quantization: [4bit]
```

2. Reduce batch size:
```yaml
batch_size: 4  # or even 1
```

3. Use a smaller model:
```yaml
models:
  - hf:google/gemma-2b-it  # Instead of larger models
```

4. Clear GPU memory between experiments:
```python
from ragicamp.utils.resource_manager import ResourceManager
ResourceManager.clear_gpu_memory()
```

### BERTScore: "CUDA out of memory"

**Solutions:**
```bash
# Use CPU for BERTScore (slower but works)
# Or reduce dataset size
num_questions: 50
```

---

## Model Issues

### Error: "You need to accept the Gemma license"

**Solution:**
1. Visit: https://huggingface.co/google/gemma-2b-it
2. Click "Agree and access repository"
3. Login: `uv run huggingface-cli login`

### Error: "We couldn't connect to huggingface.co"

**Solutions:**
```bash
# 1. Check internet
ping huggingface.co

# 2. Login to HuggingFace
uv run huggingface-cli login

# 3. Check if you accepted the license
```

### HuggingFace Padding Warning

**Symptom:**
```
A decoder-only architecture is being used, but right-padding was detected!
```

**Solution:** Already fixed in v0.3.0. The code sets `padding_side='left'` automatically.

---

## Metrics Issues

### Error: "No module named 'bert_score'"

**Solution:**
```bash
uv sync --extra metrics
```

### Error: "No module named 'bleurt'"

**Solution:**
```bash
uv sync --extra metrics
```

### BLEURT is extremely slow

**This is normal!** BLEURT is the slowest metric.

**Solutions:**
- Use fewer examples: `num_questions: 50`
- Skip BLEURT: `metrics: [f1, exact_match, bertscore]`
- Compute BLEURT later: `ragicamp metrics <dir> -m bleurt`

### LLM Judge API Errors

**Solution:** Check your OpenAI API key and model name:
```yaml
llm_judge:
  model: openai:gpt-4o-mini  # Make sure model exists
```

---

## Installation Issues

### Error: "uv: command not found"

**Solution:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart terminal or source profile
```

### Error: "No module named 'ragicamp'"

**Solution:** Always use `uv run`:
```bash
# Correct
uv run python script.py
uv run ragicamp run config.yaml

# Wrong
python script.py
ragicamp run config.yaml
```

### NumPy Version Compatibility

**Symptom:**
```
AttributeError: _ARRAY_API not found
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Solution:**
```bash
# Resync dependencies (numpy is pinned in pyproject.toml)
uv sync
```

---

## Performance Issues

### Evaluation is Very Slow

**Solutions:**

1. Use batch processing:
```yaml
batch_size: 8  # or 16
```

2. Use 4-bit quantization:
```yaml
quantization: [4bit]
```

3. Skip completed experiments:
```bash
uv run ragicamp run config.yaml --skip-existing
```

4. Reduce dataset size for testing:
```yaml
num_questions: 50
```

### Dry-run is Slow

**Cause:** Loading TensorFlow/PyTorch for health checks.

**Solution:** This is normal for the first run. Subsequent runs are faster.

---

## Configuration Issues

### Error: "Config missing required field"

**Solution:**
```bash
# Validate config first
uv run ragicamp run config.yaml --validate

# Check the error message for missing fields
```

### Error: "Invalid model spec"

**Solution:** Use correct format:
- HuggingFace: `hf:google/gemma-2b-it`
- OpenAI: `openai:gpt-4o-mini`

### Error: "Unknown dataset"

**Solution:** Valid datasets are:
- `nq` (Natural Questions)
- `triviaqa`
- `hotpotqa`

---

## Quick Diagnosis

```bash
# Check UV
uv --version

# Check Python
uv run python --version

# Check ragicamp
uv run python -c "import ragicamp; print(f'v{ragicamp.__version__}')"

# Check experiment health
uv run ragicamp health outputs/my_study

# Test basic run
uv run ragicamp run conf/study/simple_hf.yaml --dry-run
```

## Start Fresh

If nothing works:
```bash
# Remove virtual environment
rm -rf .venv

# Re-install
uv sync

# Test import
uv run python -c "import ragicamp; print('✓ Works')"
```

## Still Having Issues?

1. Check the documentation:
   - [Optuna Study Design](OPTUNA_STUDY_DESIGN.md) — How the optimization loop works
   - [Metrics Guide](guides/METRICS.md) — Evaluation metrics reference

2. Check experiment health:
```bash
   uv run ragicamp health outputs/my_study
   ```

3. Look at state.json for errors:
```bash
   cat outputs/my_study/exp_name/state.json
   ```

4. Try the simplest possible workflow:
   ```bash
   uv run ragicamp run conf/study/simple_hf.yaml --dry-run
   ```
