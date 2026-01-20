# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import matplotlib
import matplotlib.pyplot as plt

# Configure matplotlib for Jupyter after importing pyplot
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        # Enable inline plotting for Jupyter
        ipython.run_line_magic('matplotlib', 'inline')
except:
    # Not in Jupyter or magic command failed, continue anyway
    pass

from collections import defaultdict
from datetime import datetime

# Example:
# tsm3 = TreeScapeModel( caliReader )
#
# alltests = sorted(tsm3, key=lambda x: x.metadata["launchday"])
#
# grapher = MultiLine(alltests)
# grapher.plot_sums( "launchday", "main" )
# grapher.plot_sums( "figure_of_merit", "main" )


class MultiLine:
    def __init__(self, all_tests):
        self.all_tests = all_tests

    def plot_sums(self, xaxis, node_name, line_metadata_name):
        # Create a figure

        self.plot_one_sum(self.all_tests, xaxis, node_name, line_metadata_name)
        # [t for t in self.all_tests if t.metadata["test"] == testname])

    def launchday_to_date(self, epoch_timestamp):
        # Ensure the timestamp is an integer
        epoch_timestamp = int(epoch_timestamp)

        # Create a datetime object from the epoch timestamp
        date = datetime.fromtimestamp(epoch_timestamp)

        # Define month abbreviations
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        # Extract date components
        year = date.year
        month = months[date.month - 1]  # Months are 1-based in Python
        day = f"{date.day:02d}"
        hours = f"{date.hour:02d}"
        minutes = f"{date.minute:02d}"
        seconds = f"{date.second:02d}"

        # Create the readable string
        readable_string = f"{year}-{month}-{day} {hours}:{minutes}:{seconds}"

        return readable_string

    def configure_date_xaxis(self, plt, labels):
        """Configure x-axis for better date readability"""
        from datetime import datetime

        # Convert string dates back to datetime objects for better handling
        try:
            # Parse the date strings back to datetime objects
            date_objects = []
            for label in labels:
                if isinstance(label, str):
                    # Parse format like "2021-Apr-01 17:00:00"
                    date_obj = datetime.strptime(label, "%Y-%b-%d %H:%M:%S")
                    date_objects.append(date_obj)
                else:
                    date_objects.append(label)

            # Determine appropriate tick spacing based on data range
            total_days = (max(date_objects) - min(date_objects)).days

            if total_days > 1000:  # More than ~3 years
                # Show every 6 months
                tick_spacing = max(1, len(labels) // 8)
                date_format = "%Y-%b"
            elif total_days > 365:  # More than 1 year
                # Show every 2 months
                tick_spacing = max(1, len(labels) // 12)
                date_format = "%Y-%b"
            elif total_days > 90:  # More than 3 months
                # Show every 2 weeks
                tick_spacing = max(1, len(labels) // 20)
                date_format = "%b-%d"
            else:
                # Show every few days
                tick_spacing = max(1, len(labels) // 15)
                date_format = "%b-%d"

            # Select subset of labels and positions
            tick_positions = range(0, len(labels), tick_spacing)
            tick_labels = []

            for i in tick_positions:
                if i < len(date_objects):
                    formatted_date = date_objects[i].strftime(date_format)
                    tick_labels.append(formatted_date)

            # Set the ticks
            plt.xticks(
                ticks=[labels[i] for i in tick_positions if i < len(labels)],
                labels=tick_labels,
                rotation=45,
                ha="right",
            )

        except Exception as e:
            # Fallback to original behavior if date parsing fails
            print(f"Warning: Could not parse dates for better formatting: {e}")
            # Show every 20th label to reduce crowding
            tick_spacing = max(1, len(labels) // 20)
            tick_positions = range(0, len(labels), tick_spacing)
            plt.xticks(
                ticks=[labels[i] for i in tick_positions if i < len(labels)],
                labels=[labels[i] for i in tick_positions if i < len(labels)],
                rotation=45,
                ha="right",
            )

    def plot_one_sum(self, tests, xaxis, node_name, line_metadata_name):

        plt.figure(figsize=(12, 6))

        test_data = defaultdict(
            lambda: ([], [])
        )  # Dictionary to store labels and sums per test type

        # Loop through each test in tests
        for test in tests:
            # going through all the Runs
            # print(test)
            # exit()
            if hasattr(test, "perftree") and isinstance(test.perftree, dict):
                myx = test.metadata[xaxis]
                # print(f"{metavar} for test {test.metadata.get('test', 'Unknown')}: {myx}")  # Debug line
                myx = self.convert_to_number(myx)
                test_name = test.metadata.get(
                    line_metadata_name, "Unknown"
                )  # Use 'Unknown' if metadata is missing

                for key, metrics in test.perftree.items():
                    if key == node_name and "sum" in metrics:

                        if xaxis == "launchday" or xaxis == "launchdate":
                            myx = self.launchday_to_date(myx)

                        # Handle corrupted duplicate values from old make_date_test2.py
                        try:
                            sum_value = float(metrics["sum"])
                            test_data[test_name][0].append(
                                myx
                            )  # Store job size (x-axis)
                            test_data[test_name][1].append(
                                sum_value
                            )  # Store sum value (y-axis)
                        except ValueError as e:
                            # Skip corrupted values
                            print(
                                f"Warning--{e}: Skipping corrupted value '{metrics['sum']}' for test {test_name}"
                            )
                            continue

        # Plot each unique test type separately
        for test_name, (labels, sums) in test_data.items():
            labels, sums = self.make_x_uniq(labels, sums)
            plt.plot(labels, sums, label=f"{test_name}")

        # Configure plot
        plt.title("Sum Values from Node: " + node_name)
        plt.xlabel(xaxis)
        plt.ylabel("Sum Values")

        # Improve x-axis readability for date data
        if xaxis == "launchday" or xaxis == "launchdate":
            self.configure_date_xaxis(plt, labels)
        else:
            plt.xticks(rotation=45, ha="right")

        plt.legend()
        plt.tight_layout()

        # In Jupyter with %matplotlib inline, the figure displays automatically
        # We don't need plt.show() - it actually causes the text representation to appear
        # Just let the figure object be returned and Jupyter will display it
        try:
            from IPython import get_ipython
            if get_ipython() is None:
                # Not in Jupyter, use show()
                plt.show()
        except:
            # IPython not available, use show()
            plt.show()

    def make_x_uniq(self, a, b):
        import numpy as np

        # Create a dictionary to store the sum of corresponding b values for each unique a
        unique_a = sorted(set(a))  # Get unique values of a, sorted ascending
        summed_b = []

        # Sum the corresponding b values for each unique value in a
        for value in unique_a:
            summed_b.append(sum(b[i] for i in range(len(a)) if a[i] == value))

        # Convert the results into arrays (if needed)
        a = np.array(unique_a)
        b = np.array(summed_b)

        return a, b

    def convert_to_number(self, s):
        try:
            # Try converting to an integer
            return int(s)
        except ValueError:
            return float(s)
