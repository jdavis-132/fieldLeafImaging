#!/bin/bash
# Comprehensive Model Comparison Runner Script
# Usage: ./run_model_comparison.sh [num_samples]

set -e  # Exit on error

echo "=========================================="
echo "Model Comparison Framework"
echo "=========================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Default number of samples (all validation samples)
NUM_SAMPLES=${1:-"all"}

# Prepare command
if [ "$NUM_SAMPLES" = "all" ]; then
    echo "Running comparison on ALL validation samples (~814 images)"
    echo "This may take 30-60 minutes depending on your GPU..."
    echo ""
    CMD="python3 comprehensive_model_comparison.py"
else
    echo "Running comparison on $NUM_SAMPLES validation samples"
    echo ""
    CMD="python3 comprehensive_model_comparison.py --num-samples $NUM_SAMPLES"
fi

# Create output directory
mkdir -p comparison_results

# Run comparison
echo "Starting comparison..."
echo "Command: $CMD"
echo ""

$CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Comparison Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: comparison_results/"
    echo ""
    echo "Key files:"
    echo "  - comparison_results/summary_table.csv"
    echo "  - comparison_results/detailed_metrics.json"
    echo "  - comparison_results/visualizations/"
    echo ""
    echo "View results:"
    echo "  cat comparison_results/summary_table.csv"
    echo "  firefox comparison_results/visualizations/*.png"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Error occurred during comparison"
    echo "=========================================="
    echo ""
    echo "Please check error messages above"
    exit 1
fi
