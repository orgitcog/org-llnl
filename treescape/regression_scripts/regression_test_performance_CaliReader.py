#!/Users/aschwanden1/min-venv/bin/python
#
# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

"""
Regression Test for performance_CaliReader.py

This script captures the current state of alltests as a baseline and
detects major differences in future runs.
"""

import sys
import json
import hashlib
import os
from datetime import datetime
from collections import defaultdict, Counter


sys.path.append("/usr/gapps/spot/treescape-ven/lib/python3.9/site-packages")
sys.path.append("/usr/gapps/spot/treescape")

# this is just for local, not used for other environments.
# sys.path.append("/Users/aschwanden1/min-venv-local/lib/python3.9/site-packages")
sys.path.append("/")

import treescape as tr

# Configuration (matching performance_CaliReader.py)
# set cali file loc just in case the parameter is missing.
cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test_plus_6"
xaxis = "launchday"
metadata_key = "test"
processes_for_parallel_read = 15
initial_regions = ["main"]

# Regression test configuration
BASELINE_DIR = "regression_scripts/baseline_results"
TOLERANCE_PERCENT = 5.0  # Allow 5% variance in numeric values
MAX_MISSING_TESTS = 10  # Allow up to 10 missing tests
MAX_EXTRA_TESTS = 10  # Allow up to 10 extra tests


def get_baseline_filename(dataset_path):
    """Generate a baseline filename based on the dataset path"""
    # Extract the dataset name from the path
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    # Create a safe filename
    safe_name = "".join(
        c for c in dataset_name if c.isalnum() or c in ("-", "_")
    ).rstrip()
    return os.path.join(BASELINE_DIR, f"{safe_name}.json")


def extract_test_signature(test):
    """Extract key characteristics of a test for comparison"""
    signature = {}

    # Basic metadata
    if hasattr(test, "metadata"):
        signature["metadata"] = dict(test.metadata)

    # Performance tree summary
    if hasattr(test, "perftree") and isinstance(test.perftree, dict):
        perf_summary = {}
        for node_name, metrics in test.perftree.items():
            if isinstance(metrics, dict):
                node_summary = {}
                for metric_name, value in metrics.items():
                    try:
                        # Convert to float for numeric comparison
                        node_summary[metric_name] = float(value)
                    except (ValueError, TypeError):
                        # Keep as string for non-numeric values
                        node_summary[metric_name] = str(value)
                perf_summary[node_name] = node_summary
        signature["perftree_summary"] = perf_summary

    return signature


def generate_dataset_fingerprint(alltests):
    """Generate a comprehensive fingerprint of the dataset"""
    fingerprint = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(alltests),
        "test_signatures": [],
        "metadata_summary": {},
        "perftree_summary": {},
        "data_hash": None,
    }

    # Extract signatures for each test
    for i, test in enumerate(alltests):
        signature = extract_test_signature(test)
        signature["test_index"] = i
        fingerprint["test_signatures"].append(signature)

    # Metadata summary
    metadata_counts = defaultdict(Counter)
    for test in alltests:
        if hasattr(test, "metadata"):
            for key, value in test.metadata.items():
                metadata_counts[key][str(value)] += 1

    fingerprint["metadata_summary"] = {
        key: dict(counter) for key, counter in metadata_counts.items()
    }

    # Performance tree summary
    node_metrics = defaultdict(lambda: defaultdict(list))
    for test in alltests:
        if hasattr(test, "perftree") and isinstance(test.perftree, dict):
            for node_name, metrics in test.perftree.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        try:
                            node_metrics[node_name][metric_name].append(float(value))
                        except (ValueError, TypeError):
                            pass

    # Calculate statistics for numeric metrics
    perf_stats = {}
    for node_name, metrics in node_metrics.items():
        node_stats = {}
        for metric_name, values in metrics.items():
            if values:
                node_stats[metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "sum": sum(values),
                }
        perf_stats[node_name] = node_stats

    fingerprint["perftree_summary"] = perf_stats

    # Generate hash of the entire dataset for quick comparison
    dataset_str = json.dumps(fingerprint["test_signatures"], sort_keys=True)
    fingerprint["data_hash"] = hashlib.md5(dataset_str.encode()).hexdigest()

    return fingerprint


def save_baseline(fingerprint, dataset_path):
    """Save the current fingerprint as the baseline"""
    baseline_file = get_baseline_filename(dataset_path)

    # Ensure the baseline directory exists
    os.makedirs(BASELINE_DIR, exist_ok=True)

    with open(baseline_file, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"✅ Baseline saved to {baseline_file}")


def load_baseline(dataset_path):
    """Load the baseline fingerprint for a specific dataset"""
    baseline_file = get_baseline_filename(dataset_path)

    if not os.path.exists(baseline_file):
        return None

    with open(baseline_file, "r") as f:
        return json.load(f)


def compare_fingerprints(baseline, current):
    """Compare current fingerprint against baseline"""
    issues = []
    warnings = []

    # Check total test count
    baseline_count = baseline["total_tests"]
    current_count = current["total_tests"]

    if abs(current_count - baseline_count) > MAX_MISSING_TESTS:
        issues.append(
            f"Test count changed significantly: {baseline_count} -> {current_count}"
        )
    elif current_count != baseline_count:
        warnings.append(f"Test count changed: {baseline_count} -> {current_count}")

    # Check data hash for quick comparison
    if baseline["data_hash"] != current["data_hash"]:
        warnings.append("Data hash changed - detailed comparison needed")

    # Always compare performance tree statistics (not just when hash changes)
    baseline_perf = baseline.get("perftree_summary", {})
    current_perf = current.get("perftree_summary", {})

    bpf = baseline_perf.keys()
    cpf = current_perf.keys()

    for node_name in bpf | cpf:
        if node_name not in baseline_perf:
            warnings.append(f"New performance node: {node_name}")
            continue
        if node_name not in current_perf:
            issues.append(f"Missing performance node: {node_name}")
            continue

        baseline_node = baseline_perf[node_name]
        current_node = current_perf[node_name]

        for metric_name in set(baseline_node.keys()) | set(current_node.keys()):
            if metric_name not in baseline_node:
                warnings.append(f"New metric {node_name}.{metric_name}")
                continue
            if metric_name not in current_node:
                issues.append(f"Missing metric {node_name}.{metric_name}")
                continue

            baseline_stats = baseline_node[metric_name]
            current_stats = current_node[metric_name]

            # Compare key statistics
            for stat_name in ["count", "min", "max", "avg", "sum"]:
                if stat_name in baseline_stats and stat_name in current_stats:
                    baseline_val = baseline_stats[stat_name]
                    current_val = current_stats[stat_name]

                    if baseline_val != 0:
                        percent_change = (
                            abs(current_val - baseline_val) / abs(baseline_val) * 100
                        )
                        if percent_change > TOLERANCE_PERCENT:
                            issues.append(
                                f"Significant change in {node_name}.{metric_name}.{stat_name}: "
                                f"{baseline_val:.3f} -> {current_val:.3f} ({percent_change:.1f}% change)"
                            )

    return issues, warnings


def run_regression_test(update_baseline=False, dataset_path=None):
    """Run the regression test"""

    # Use provided dataset path or default
    test_dataset = dataset_path if dataset_path else cali_file_loc

    print("=" * 60)
    print("REGRESSION TEST: performance_CaliReader.py")
    print("=" * 60)
    print(f"Dataset: {test_dataset}")

    if update_baseline:
        print("🔄 UPDATE BASELINE MODE")
    else:
        print("🧪 COMPARISON MODE")
    print("=" * 60)

    # Load data
    print("Loading data...")

    if __name__ == "__main__":
        from multiprocessing import freeze_support

        freeze_support()

        inclusive_strs = [
            "min#inclusive#sum#time.duration",
            "max#inclusive#sum#time.duration",
            "avg#inclusive#sum#time.duration",
            "sum#inclusive#sum#time.duration",
        ]

        caliReader = tr.CaliReader(
            test_dataset, processes_for_parallel_read, inclusive_strs
        )
        tsm = tr.TreeScapeModel(caliReader)
        alltests = sorted(tsm, key=lambda x: x.metadata[xaxis])

        print(f"Loaded {len(alltests)} tests")

        # Generate current fingerprint
        print("Generating dataset fingerprint...")
        current_fingerprint = generate_dataset_fingerprint(alltests)
        current_fingerprint["dataset_path"] = (
            test_dataset  # Track which dataset was used
        )

        if update_baseline:
            # Update baseline mode
            print("📝 Updating baseline...")
            save_baseline(current_fingerprint, test_dataset)
            print("✅ Baseline updated successfully!")
            print(f"📊 New baseline: {len(alltests)} tests from {test_dataset}")
            return True

        # Normal comparison mode
        baseline_fingerprint = load_baseline(test_dataset)

        if baseline_fingerprint is None:
            print("📝 No baseline found. Creating baseline...")
            save_baseline(current_fingerprint, test_dataset)
            print("✅ Baseline created successfully!")
            return True

        # Show baseline info
        baseline_dataset = baseline_fingerprint.get("dataset_path", "Unknown")
        baseline_count = baseline_fingerprint.get("total_tests", "Unknown")
        print(f"📋 Baseline: {baseline_count} tests from {baseline_dataset}")

        # Compare against baseline
        print("Comparing against baseline...")
        issues, warnings = compare_fingerprints(
            baseline_fingerprint, current_fingerprint
        )

        # Report results
        print("\n" + "=" * 60)
        print("REGRESSION TEST RESULTS")
        print("=" * 60)

        if not issues and not warnings:
            print("✅ PASS: No significant changes detected")
            return True

        if warnings:
            print(f"⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"   • {warning}")
            print()

        if issues:
            print(f"❌ ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"   • {issue}")
            print()
            print("❌ FAIL: Significant changes detected!")
            return False
        else:
            print("✅ PASS: Only minor changes detected")
            return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regression test for performance_CaliReader.py"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update the baseline with current dataset",
    )
    parser.add_argument(
        "--dataset", type=str, help="Path to dataset (default: test_plus_80)"
    )
    parser.add_argument(
        "--list-info", action="store_true", help="Show current baseline information"
    )

    args = parser.parse_args()

    if args.list_info:
        # Show baseline info
        if args.dataset:
            # Show info for specific dataset
            baseline = load_baseline(args.dataset)
            if baseline:
                print("=" * 60)
                print("BASELINE INFORMATION")
                print("=" * 60)
                print(f"Created: {baseline['timestamp']}")
                print(f"Dataset: {baseline.get('dataset_path', 'Unknown')}")
                print(f"Total tests: {baseline['total_tests']}")
                print(f"Data hash: {baseline['data_hash']}")
                print(f"Baseline file: {get_baseline_filename(args.dataset)}")
            else:
                print(f"❌ No baseline found for dataset: {args.dataset}")
        else:
            # Show all available baselines
            print("=" * 60)
            print("AVAILABLE BASELINES")
            print("=" * 60)
            if os.path.exists(BASELINE_DIR):
                baseline_files = [
                    f for f in os.listdir(BASELINE_DIR) if f.endswith(".json")
                ]
                if baseline_files:
                    for baseline_file in sorted(baseline_files):
                        baseline_path = os.path.join(BASELINE_DIR, baseline_file)
                        try:
                            with open(baseline_path, "r") as f:
                                baseline = json.load(f)
                            dataset_name = baseline_file.replace(".json", "")
                            print(
                                f"• {dataset_name}: {baseline['total_tests']} tests, created {baseline['timestamp']}"
                            )
                        except Exception as e:
                            print(f"• {baseline_file}: Error reading file ({e})")
                else:
                    print("No baseline files found")
            else:
                print("Baseline directory does not exist")
        sys.exit(0)

    success = run_regression_test(
        update_baseline=args.update_baseline, dataset_path=args.dataset
    )
    sys.exit(0 if success else 1)
