# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
import datetime
import time

# Only import IPython if available (for Jupyter environments)
try:
    from IPython.display import Javascript, display

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


class PythonPull:

    def __init__(self):
        self.json_result = None

    # Function to execute JavaScript and pass JSON back to Python
    def extract_json(self):
        if not IPYTHON_AVAILABLE:
            print(
                "Warning: IPython not available. Cannot extract JSON from JavaScript."
            )
            return

        js_code = """
        (function() {
            var data = window.plots || {};  // Ensure data exists
            var jsonString = JSON.stringify(data);
            console.log("Extracted JSON:", jsonString);

            // Use display.JSON to send it back to Python
            IPython.notebook.kernel.execute("json_result = " + jsonString);
        })();
        """

        display(Javascript(js_code))

    def exe(self):
        # Call the function to execute JS and extract the JSON
        self.extract_json()
        # print(json_result)

        # Wait for the JavaScript execution to complete
        time.sleep(1)

        # Ensure json_result is set
        if self.json_result is None:
            raise ValueError("Failed to retrieve JSON data from JavaScript.")

        # Given dataset
        data = self.json_result

        # Extract data
        x_timestamps = list(
            map(int, data[0]["nodes"]["xaxis"])
        )  # Convert timestamps to integers
        y_values = data[0]["nodes"]["ydata"]
        series_name = data[0]["nodes"]["name"]

        # Convert timestamps to datetime
        x_dates = [datetime.datetime.utcfromtimestamp(ts) for ts in x_timestamps]

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.fill_between(
            x_dates, y_values, alpha=0.4, color="blue"
        )  # Stacked effect with fill
        plt.plot(
            x_dates,
            y_values,
            marker="o",
            linestyle="-",
            color="blue",
            label=series_name,
        )

        # Formatting
        plt.xlabel("Date")
        plt.ylabel("Y Values")
        plt.title("Stacked Line Plot - " + series_name)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # Show the plot
        plt.show()
