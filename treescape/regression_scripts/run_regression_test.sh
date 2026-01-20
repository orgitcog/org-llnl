#!/bin/bash

# Regression Test Runner for performance_CaliReader.py
# This script ensures the test runs from the correct directory
#
# Usage:
#   ./run_regression_test.sh [dataset_name]
#
# Examples:
#   ./run_regression_test.sh test_plus_6
#   ./run_regression_test.sh test_plus_24c
#   ./run_regression_test.sh  # Uses default dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREESCAPE_DIR="$(dirname "$SCRIPT_DIR")"

# Check if dataset parameter is provided
if [ $# -eq 1 ]; then
    DATASET_NAME="$1"
    DATASET_PATH="/Users/aschwanden1/datasets/newdemo/$DATASET_NAME"
	DATASET_PATH="/usr/gapps/spot/datasets/newdemo/$DATASET_NAME"

    echo "Running regression test for performance_CaliReader.py..."
    echo "Dataset: $DATASET_NAME"
    echo "Dataset path: $DATASET_PATH"
    echo "Working directory: $TREESCAPE_DIR"
    echo

    # Check if dataset directory exists
    if [ ! -d "$DATASET_PATH" ]; then
        echo "❌ Error: Dataset directory does not exist: $DATASET_PATH"
        echo
        echo "Available datasets:"
        ls -1 /Users/aschwanden1/datasets/newdemo/ | grep -E '^(test|mpi|user)' | head -10
        exit 1
    fi

    cd "$TREESCAPE_DIR"
    python3 regression_scripts/regression_test_performance_CaliReader.py --dataset "$DATASET_PATH"
elif [ $# -eq 0 ]; then
    echo "Running regression test for performance_CaliReader.py..."
    echo "Using default dataset"
    echo "Working directory: $TREESCAPE_DIR"
    echo

    cd "$TREESCAPE_DIR"
    python3 regression_scripts/regression_test_performance_CaliReader.py
else
    echo "Usage: $0 [dataset_name]"
    echo
    echo "Examples:"
    echo "  $0 test_plus_6"
    echo "  $0 test_plus_24c"
    echo "  $0  # Uses default dataset"
    echo
    echo "Available datasets:"
    ls -1 /Users/aschwanden1/datasets/newdemo/ | grep -E '^(test|mpi|user)' | head -10
    exit 1
fi

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo
    echo "✅ Regression test PASSED"
else
    echo
    echo "❌ Regression test FAILED"
fi

exit $exit_code
