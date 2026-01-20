#!/usr/bin/env python3
#
# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

"""
Convenient script to add a regression test baseline for a specific dataset.
This creates a separate baseline file for each test input directory,
allowing multiple test directories to coexist without overwriting baselines.
"""

import sys
import os

# Add paths
sys.path.append("/Users/aschwanden1/min-venv-local/lib/python3.9/site-packages")
sys.path.append("/")


def add_baseline_for_dataset(dataset_path):
    """Add a regression baseline for a specific dataset"""

    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path does not exist: {dataset_path}")
        return False

    if not os.path.isdir(dataset_path):
        print(f"❌ Error: Dataset path is not a directory: {dataset_path}")
        return False

    # Check for .cali files
    import glob

    cali_files = glob.glob(os.path.join(dataset_path, "**", "*.cali"), recursive=True)
    if not cali_files:
        print(f"❌ Error: No .cali files found in {dataset_path}")
        return False

    print(f"📁 Found {len(cali_files)} .cali files in {dataset_path}")

    # Import and run the regression test in update mode
    import subprocess

    print(f"🔄 Adding baseline for dataset: {dataset_path}")

    # Run the regression test script with update baseline flag
    cmd = [
        sys.executable,
        "regression_scripts/regression_test_performance_CaliReader.py",
        "--update-baseline",
        "--dataset",
        dataset_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd="/Users/aschwanden1/treescape"
        )

        if result.returncode == 0:
            print("✅ Baseline added successfully!")
            print(
                "💡 You can now run regression tests against this baseline for this specific dataset."
            )
            success = True
        else:
            print("❌ Failed to add baseline")
            print("Error output:")
            print(result.stderr)
            success = False

        # Show any output from the regression test
        if result.stdout:
            print("\nRegression test output:")
            print(result.stdout)

    except Exception as e:
        print(f"❌ Error running regression test: {e}")
        success = False

    return success


def show_usage():
    """Show usage information"""
    print("Add Regression Baseline Tool")
    print("=" * 40)
    print()
    print("Usage:")
    print("  python regression_scripts/add_regression_baseline.py <dataset_path>")
    print()
    print("Examples:")
    print("  # Add baseline for test_plus_24c dataset")
    print(
        "  python regression_scripts/add_regression_baseline.py /Users/aschwanden1/datasets/newdemo/test_plus_24c"
    )
    print()
    print("  # Add baseline for test dataset")
    print(
        "  python regression_scripts/add_regression_baseline.py /Users/aschwanden1/datasets/newdemo/test"
    )
    print()
    print("This will create a separate baseline file for the specified dataset")
    print("in the regression_scripts/baseline_results/ directory.")
    print()
    print("Available datasets:")
    base_path = "/Users/aschwanden1/datasets/newdemo/"
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Count .cali files
                import glob

                cali_count = len(
                    glob.glob(os.path.join(item_path, "**", "*.cali"), recursive=True)
                )
                if cali_count > 0:
                    print(f"  • {item} ({cali_count} .cali files)")
    print()
    print("To see existing baselines:")
    print(
        "  python regression_scripts/regression_test_performance_CaliReader.py --list-info"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        show_usage()
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Handle relative paths
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(dataset_path)

    success = add_baseline_for_dataset(dataset_path)
    sys.exit(0 if success else 1)
