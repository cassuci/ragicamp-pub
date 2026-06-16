#!/bin/bash
# === RAGiCamp Results Discovery Script ===
# Run this on the server to show what experiments exist and their metrics.
#
# Usage:
#   cd ~/repos/ragicamp && bash scripts/discover_results.sh
#   REPO_DIR=/path/to/ragicamp bash scripts/discover_results.sh

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "========================================"
echo " RAGiCamp Results Discovery"
echo "========================================"
echo ""

# 1. Find all output directories
echo "--- Output directories ---"
for d in "$REPO_DIR"/outputs/*/; do
    [ -d "$d" ] || continue
    study=$(basename "$d")
    n_exps=$(find "$d" -maxdepth 1 -mindepth 1 -type d | wc -l)
    n_complete=$(find "$d" -name "results.json" -maxdepth 2 | wc -l)
    n_predictions=$(find "$d" -name "predictions.json" -maxdepth 2 | wc -l)
    echo "  $study: $n_exps experiments ($n_complete complete, $n_predictions with predictions)"
done
echo ""

# 2. Show per-experiment file structure for one example
echo "--- Example experiment file listing (first complete one) ---"
first_result=$(find "$REPO_DIR"/outputs -name "results.json" -maxdepth 3 2>/dev/null | head -1)
if [ -n "$first_result" ]; then
    exp_dir=$(dirname "$first_result")
    echo "  Dir: $exp_dir"
    ls -lh "$exp_dir"
else
    echo "  No completed experiments found."
fi
echo ""

# 3. Show results.json structure (first experiment)
echo "--- Example results.json (first found) ---"
if [ -n "$first_result" ]; then
    python3 -c "
import json
with open('$first_result') as f:
    d = json.load(f)
print('Keys:', list(d.keys()))
if 'metrics' in d:
    print('Metrics:', json.dumps(d['metrics'], indent=2))
if 'name' in d:
    print('Name:', d['name'])
if 'num_examples' in d:
    print('Num examples:', d['num_examples'])
if 'duration_seconds' in d:
    print('Duration:', round(d['duration_seconds'], 1), 'seconds')
"
fi
echo ""

# 4. Summary table of all completed experiments and their metrics
echo "--- All completed experiments + metrics ---"
python3 << 'PYEOF'
import json, os
from pathlib import Path

repo = Path(os.environ.get("REPO_DIR", "."))
rows = []
for results_file in sorted(repo.glob("outputs/**/results.json")):
    try:
        with open(results_file) as f:
            d = json.load(f)
        m = d.get("metrics", {})
        rows.append({
            "study": results_file.parent.parent.name,
            "experiment": d.get("name", results_file.parent.name),
            "f1": m.get("f1", ""),
            "exact_match": m.get("exact_match", ""),
            "bertscore": m.get("bertscore", ""),
            "bleurt": m.get("bleurt", ""),
            "llm_judge": m.get("llm_judge", ""),
            "n": d.get("num_examples", ""),
            "duration_s": round(d.get("duration_seconds", 0), 1),
        })
    except Exception as e:
        print(f"  WARN: {results_file}: {e}")

if not rows:
    print("  No completed experiments found.")
else:
    hdr = f"{'study':<30} {'experiment':<50} {'f1':>6} {'em':>6} {'bert':>6} {'bleurt':>6} {'judge':>6} {'n':>4} {'dur_s':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and v != "" else str(v)[:6]
        print(f"{r['study']:<30} {r['experiment']:<50} {fmt(r['f1']):>6} {fmt(r['exact_match']):>6} {fmt(r['bertscore']):>6} {fmt(r['bleurt']):>6} {fmt(r['llm_judge']):>6} {str(r['n']):>4} {r['duration_s']:>7}")

    print(f"\nTotal: {len(rows)} completed experiments")
PYEOF

echo ""
echo "--- Incomplete experiments (have predictions but no results) ---"
python3 << 'PYEOF'
import json, os
from pathlib import Path

repo = Path(os.environ.get("REPO_DIR", "."))
found = False
for pred_file in sorted(repo.glob("outputs/**/predictions.json")):
    results_file = pred_file.parent / "results.json"
    if not results_file.exists():
        found = True
        try:
            with open(pred_file) as f:
                d = json.load(f)
            n = len(d.get("predictions", []))
            state_file = pred_file.parent / "state.json"
            phase = "?"
            if state_file.exists():
                with open(state_file) as f:
                    phase = json.load(f).get("phase", "?")
            print(f"  {pred_file.parent.name}: {n} predictions, phase={phase}")
        except Exception as e:
            print(f"  {pred_file.parent.name}: ERROR {e}")
if not found:
    print("  None found.")
PYEOF

echo ""
echo "--- Disk usage ---"
du -sh "$REPO_DIR"/outputs/*/ 2>/dev/null || echo "  No output directories"
