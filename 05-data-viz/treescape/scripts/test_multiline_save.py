#!/Users/aschwanden1/min-venv/bin/python

cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test_plus_24a"
xaxis = "launchday"
metadata_key = "test"
processes_for_parallel_read = 15
initial_regions = ["main"]

import sys
import warnings
import os
from contextlib import redirect_stderr
from io import StringIO

# Filter out the specific Roundtrip warning
warnings.filterwarnings("ignore", message=".*Roundtrip module could not be loaded.*")

sys.path.append("/Users/aschwanden1/min-venv-local/lib/python3.9/site-packages")
sys.path.append("/Users/aschwanden1/treescape")


# Custom stderr filter to suppress Roundtrip warnings
class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, text):
        if "Roundtrip module could not be loaded" not in text:
            self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()


# Replace stderr with filtered version
sys.stderr = FilteredStderr(sys.stderr)

import treescape as tr


# Custom MultiLine class that saves plots instead of showing them
class MultiLineSave(tr.MultiLine):
    def __init__(self, all_tests, save_directory="/Users/aschwanden1/svg_imgs/"):
        super().__init__(all_tests)
        self.save_directory = save_directory
        self.plot_counter = 0

    def plot_one_sum(self, tests, xaxis, node_name, line_metadata_name):
        import matplotlib.pyplot as plt
        from collections import defaultdict

        plt.figure(figsize=(15, 8))  # Larger figure for better readability

        test_data = defaultdict(lambda: ([], []))

        # Loop through each test in tests
        for test in tests:
            if hasattr(test, "perftree") and isinstance(test.perftree, dict):
                myx = test.metadata[xaxis]
                myx = self.convert_to_number(myx)
                test_name = test.metadata.get(line_metadata_name, "Unknown")

                for key, metrics in test.perftree.items():
                    if key == node_name and "sum" in metrics:

                        if xaxis == "launchday" or xaxis == "launchdate":
                            myx = self.launchday_to_date(myx)

                        test_data[test_name][0].append(myx)
                        test_data[test_name][1].append(float(metrics["sum"]))

        # Plot each unique test type separately
        for test_name, (labels, sums) in test_data.items():
            labels, sums = self.make_x_uniq(labels, sums)
            print(f"Plotting {test_name}: {len(labels)} data points")
            plt.plot(labels, sums, label=f"{test_name}", linewidth=1.5, alpha=0.8)

        # Configure plot
        plt.title(
            f"Sum Values from Node: {node_name}\n(Improved X-axis Formatting)",
            fontsize=14,
        )
        plt.xlabel(xaxis, fontsize=12)
        plt.ylabel("Sum Values", fontsize=12)

        # Improve x-axis readability for date data
        if xaxis == "launchday" or xaxis == "launchdate":
            self.configure_date_xaxis(plt, labels)
        else:
            plt.xticks(rotation=45, ha="right")

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save the plot
        filename = f"{self.save_directory}multiline_{node_name}_{xaxis}_{self.plot_counter}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {filename}")

        plt.close()  # Close to free memory
        self.plot_counter += 1


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
        cali_file_loc, processes_for_parallel_read, inclusive_strs
    )
    tsm = tr.TreeScapeModel(caliReader)
    alltests = sorted(tsm, key=lambda x: x.metadata[xaxis])

    # Use the custom MultiLine class that saves plots
    grapher = MultiLineSave(alltests)
    for region in initial_regions:
        grapher.plot_sums(xaxis, region, metadata_key)

    print("All plots saved successfully!")
