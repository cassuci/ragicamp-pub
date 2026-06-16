# RAGiCamp Makefile

.PHONY: help install index-simple index-full run-baseline-simple run-baseline-full evaluate compute-metrics compare

# ============================================================================
# HELP
# ============================================================================

help:
	@echo ""
	@echo "RAGiCamp"
	@echo "========"
	@echo ""
	@echo "Setup:"
	@echo "  make install              Install dependencies"
	@echo ""
	@echo "Quick Development (simple):"
	@echo "  make index-simple            Build 1 small index (500 docs)"
	@echo "  make run-baseline-simple     Run quick baseline (10 questions)"
	@echo "  make run-baseline-simple-hf  HF models only, no OpenAI costs"
	@echo ""
	@echo "Full Experiments:"
	@echo "  make index-full           Build all indexes for baseline"
	@echo "  make run-baseline-full    Run full baseline (100+ questions)"
	@echo ""
	@echo "Re-evaluate and compare:"
	@echo "  make evaluate DIR=outputs/simple METRICS=bertscore,bleurt"
	@echo "  make compare DIR=outputs/simple              # Show comparison table"
	@echo "  make compare DIR=outputs/simple SORT=f1      # Sort by metric"
	@echo ""
	@echo "Other:"
	@echo "  make download             Download datasets"
	@echo "  make test                 Run tests"
	@echo "  make clean                Clean outputs"
	@echo ""

# ============================================================================
# SETUP
# ============================================================================

install:
	uv sync

download:
	@echo "Use 'uv run ragicamp index' to build indexes (downloads corpus automatically)"

# ============================================================================
# SIMPLE (quick dev/validation)
# ============================================================================

index-simple:
	@echo "📚 Building simple index (1 index, 500 docs)..."
	uv run ragicamp index --corpus simple --embedding minilm --max-docs 500

run-baseline-simple:
	@echo "🚀 Running simple baseline (10 questions)..."
	uv run ragicamp run conf/study/simple.yaml

run-baseline-simple-hf:
	@echo "🚀 Running simple baseline - HF only, no OpenAI (10 questions)..."
	uv run ragicamp run conf/study/simple_hf.yaml

# ============================================================================
# FULL (production experiments)
# ============================================================================

index-full:
	@echo "📚 Building full indexes (3 embeddings)..."
	uv run ragicamp index --corpus simple --embedding minilm
	uv run ragicamp index --corpus simple --embedding mpnet
	uv run ragicamp index --corpus simple --embedding e5

run-baseline-full:
	@echo "🚀 Running full baseline (100 questions, all variations)..."
	uv run ragicamp run conf/study/full.yaml

# ============================================================================
# COMPREHENSIVE BASELINE (all models, all datasets)
# ============================================================================

# Build indexes for comprehensive study (simple + en corpora, 3 embeddings)
index-comprehensive:
	@echo "📚 Building comprehensive indexes (6 variations)..."
	@echo "This will take 2-4 hours for English Wikipedia..."
	uv run ragicamp index --corpus simple --embedding minilm
	uv run ragicamp index --corpus simple --embedding mpnet
	uv run ragicamp index --corpus simple --embedding e5
	uv run ragicamp index --corpus en --embedding minilm
	uv run ragicamp index --corpus en --embedding mpnet
	uv run ragicamp index --corpus en --embedding e5

# Build just Simple Wikipedia indexes (faster, for testing)
index-simple-wiki:
	@echo "📚 Building Simple Wikipedia indexes (3 variations)..."
	uv run ragicamp index --corpus simple --embedding minilm
	uv run ragicamp index --corpus simple --embedding mpnet
	uv run ragicamp index --corpus simple --embedding e5

# Run comprehensive baseline study
run-comprehensive:
	@echo "🚀 Running comprehensive baseline (full study)..."
	@echo "This will take 10-20 hours with all models..."
	@echo "Using --skip-existing to resume safely..."
	uv run ragicamp run conf/study/comprehensive_baseline.yaml --skip-existing

# Run with timeout per experiment (e.g. 2 hours max)
run-comprehensive-safe:
	@echo "🚀 Running comprehensive baseline with 2h timeout per experiment..."
	uv run ragicamp run conf/study/comprehensive_baseline.yaml --skip-existing

# ============================================================================
# EVALUATION (re-run metrics on existing predictions)
# ============================================================================

# Re-evaluate with specific metrics: make evaluate DIR=outputs/simple METRICS=bertscore bleurt
evaluate:
	@echo "📊 Re-evaluating predictions..."
	uv run ragicamp evaluate $(DIR) --metrics $(or $(METRICS),all)

# Batch compute metrics: make compute-metrics DIR=outputs/simple METRICS=llm_judge_qa
# With judge model:      make compute-metrics DIR=outputs/simple METRICS=llm_judge_qa JUDGE_MODEL=deepseek-ai/DeepSeek-V3.2
# Dry run:               make compute-metrics DIR=outputs/simple METRICS=llm_judge_qa DRY_RUN=1
# Force recompute:       make compute-metrics DIR=outputs/simple METRICS=llm_judge_qa FORCE=1
compute-metrics:
	uv run ragicamp compute-metrics $(DIR) --metrics $(METRICS) $(if $(JUDGE_MODEL),--judge-model $(JUDGE_MODEL)) $(if $(DRY_RUN),--dry-run) $(if $(FORCE),--force)

# Compare results in a table: make compare DIR=outputs/simple
compare:
	uv run ragicamp compare $(or $(DIR),outputs/simple)

# Examples:
#   make evaluate DIR=outputs/simple_hf METRICS=bertscore
#   make evaluate DIR=outputs/simple_hf/direct_hf_google_gemma2bit_default_nq METRICS=llm_judge
#   make evaluate DIR=outputs/simple METRICS=all

# ============================================================================
# DEV
# ============================================================================

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src/ragicamp --cov-report=term-missing

lint:
	uv run ruff format --check src/ tests/ scripts/
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

validate-configs:
	@echo "Validating Hydra configs..."
	uv run python -c "from hydra import compose, initialize_config_dir; from pathlib import Path; conf_dir = Path('conf').absolute(); initialize_config_dir(version_base=None, config_dir=str(conf_dir)).__enter__(); cfg = compose(config_name='config'); print('✅ All configs valid!')"

# Pre-push check (run before pushing to CI)
pre-push: format lint test
	@echo "✅ All checks passed - ready to push!"

clean:
	rm -rf outputs/
	@echo "✓ Cleaned"
