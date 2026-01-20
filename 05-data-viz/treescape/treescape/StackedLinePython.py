# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend for image generation
import matplotlib.pyplot as plt

from collections import defaultdict
from datetime import datetime
import numpy as np
import hashlib


class StackedLinePython:
    def __init__(self, directory, all_tests, testname):
        self.all_tests = all_tests
        self.directory = directory
        self.testname = testname

        self.yMax = None
        self.yMin = None

        self.xaxis_agg = "sum"
        self.yaxis = "avg"

    def setYAxis(self, yaxis):
        self.yaxis = yaxis

    def setYMax(self, yMax):
        self.yMax = yMax

    def setYMin(self, yMin):
        self.yMin = yMin

    def plot_sums(self, metavar, node_names):
        """
        Plots sums for multiple nodes as a stacked area chart.
        :param metavar: The metadata key to use for the x-axis.
        :param node_names: A list of node names to include in the plot.
        """
        self.plot_stacked_sums(self.all_tests, metavar, node_names)

    def launchday_to_date(self, epoch_timestamp):
        """
        Converts an epoch timestamp to a human-readable date string.
        """
        epoch_timestamp = int(epoch_timestamp)
        date = datetime.fromtimestamp(epoch_timestamp)
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
        return f"{date.year}-{months[date.month - 1]}-{date.day:02d} {date.hour:02d}:{date.minute:02d}:{date.second:02d}"

    def plot_stacked_sums(self, tests, metavar, node_names):
        """
        Processes and plots data as a stacked line (area) chart.
        :param tests: The list of tests to process.
        :param metavar: The metadata key used for the x-axis.
        :param node_names: A list of node names to plot separately.
        """
        plt.figure(figsize=(12, 6))

        # Capture the original tests parameter to avoid any modifications
        original_tests = tests

        test_data = {node: defaultdict(lambda: ([], [])) for node in node_names}

        # Step 1: Collect data for each node
        for test in tests:
            if hasattr(test, "perftree") and isinstance(test.perftree, dict):
                myx = test.metadata[metavar]
                myx = self.convert_to_number(myx)
                if metavar == "launchday":
                    myx = self.convert_to_number(myx)

                test_name = test.metadata.get("test", "Unknown")

                for node in node_names:
                    if node in test.perftree and self.yaxis in test.perftree[node]:
                        test_data[node][test_name][0].append(myx)
                        test_data[node][test_name][1].append(
                            float(test.perftree[node][self.yaxis])
                        )

        # Step 2: Collect all global x-values and map each node's data to them
        global_x_vals = set()
        node_to_xy = {node: [] for node in node_names}

        for node, node_tests in test_data.items():
            for test_name, (x_vals, y_vals) in node_tests.items():
                for x, y in zip(x_vals, y_vals):
                    global_x_vals.add(x)
                    node_to_xy[node].append((x, y))

        global_x_sorted = sorted(global_x_vals)
        x_index = {x: i for i, x in enumerate(global_x_sorted)}

        # Create a structure to collect all y values per (i, idx)
        temp = defaultdict(list)

        for i, node in enumerate(node_names):
            for x, y in node_to_xy[node]:
                idx = x_index[x]
                temp[(i, idx)].append(y)

        # For topmax aggregation, we need to collect "main" node data to determine the index
        # Use the same data collection loop as the main data to ensure consistency
        main_temp = defaultdict(list)
        if self.xaxis_agg == "topmax":
            # Use the original tests parameter to ensure consistency
            for test in original_tests:
                if hasattr(test, "perftree") and isinstance(test.perftree, dict):
                    myx = test.metadata[metavar]
                    myx = self.convert_to_number(myx)
                    if metavar == "launchday":
                        myx = self.convert_to_number(myx)

                    if myx in x_index:  # Make sure this x value is in our index
                        idx = x_index[myx]

                        if (
                            "main" in test.perftree
                            and self.yaxis in test.perftree["main"]
                        ):
                            main_temp[idx].append(
                                float(test.perftree["main"][self.yaxis])
                            )

        # Initialize y_matrix
        y_matrix = np.zeros((len(node_names), len(x_index)))

        # Choose aggregation function
        agg_func = self.xaxis_agg  # 'avg' or 'sum', 'min', 'max'

        # For topmax, pre-calculate the topmax indices based on main node
        topmax_indices = {}
        if agg_func == "topmax":
            for idx, main_values in main_temp.items():
                if main_values:  # Make sure we have values
                    topmax_indices[idx] = main_values.index(max(main_values))

        # Apply aggregation
        for (i, idx), values in temp.items():
            if agg_func == "sum":
                y_matrix[i][idx] = sum(values)
            elif agg_func == "avg":
                y_matrix[i][idx] = sum(values) / len(values)
            elif agg_func == "min":
                y_matrix[i][idx] = min(values)
            elif agg_func == "max":
                y_matrix[i][idx] = max(values)
            elif agg_func == "topmax":
                # Use the index determined by the main node's maximum value
                if idx in topmax_indices and topmax_indices[idx] < len(values):
                    y_matrix[i][idx] = values[topmax_indices[idx]]
                else:
                    # Fallback to max if main data is not available
                    y_matrix[i][idx] = max(values) if values else 0

        # Step 4: Plot the stacked area
        prev_sums = np.zeros(len(global_x_sorted))
        colors = [self.spot2_color_hash(str(node)) for node in node_names]

        for i, node in enumerate(node_names):
            plt.fill_between(
                global_x_sorted,
                prev_sums,
                prev_sums + y_matrix[i],
                label=f"Node {node}",
                color=colors[i],
                alpha=0.6,
            )
            prev_sums += y_matrix[i]

        # Step 5: Format x-axis
        formatted_labels = [self.launchday_to_date(x) for x in global_x_sorted]
        plt.xticks(
            ticks=global_x_sorted, labels=formatted_labels, rotation=45, ha="right"
        )

        # Save to file
        pstr = "_".join(node_names)
        filename = (
            self.directory
            + "yaxis="
            + self.yaxis
            + " xagg="
            + self.xaxis_agg
            + " test="
            + self.testname
            + "_"
            + pstr
            + ".svg"
        )

        plt.xlabel(metavar)
        plt.ylabel("Summed Values")
        plt.legend()
        plt.title("Stacked Line (Area) Graph for Multiple Nodes")

        if self.yMax is not None:
            plt.ylim(top=self.yMax)
        if self.yMin is not None:
            plt.ylim(bottom=self.yMin)

        plt.tight_layout()
        plt.savefig(filename, format="svg")
        plt.close()

    def setXAggregation(self, xagg):
        self.xaxis_agg = xagg

    def spot2_color_hash(self, text, alpha=0.6):
        reverse_string = text[::-1]
        hash_obj = hashlib.md5(reverse_string.encode())
        hash_hex = hash_obj.hexdigest()
        r = int(hash_hex[12:14], 16) / 255.0
        g = int(hash_hex[14:16], 16) / 255.0
        b = int(hash_hex[16:18], 16) / 255.0
        return (r, g, b, alpha)

    def make_x_uniq(self, a, b):
        """
        Groups and sums values for duplicate x-axis labels.
        """
        unique_a = sorted(set(a))
        summed_b = [
            sum(b[i] for i in range(len(a)) if a[i] == value) for value in unique_a
        ]
        return np.array(unique_a), np.array(summed_b)

    def convert_to_number(self, s):
        """
        Attempts to convert a value to an integer, then float if necessary.
        """
        try:
            return int(s)
        except ValueError:
            return float(s)
