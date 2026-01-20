#!/Users/aschwanden1/min-venv/bin/python

cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test_plus_24a"
xaxis = "launchday"
metadata_key = "test"
processes_for_parallel_read = 15
node_name = "main"

import sys
import warnings
from collections import defaultdict
from datetime import datetime

# Filter warnings
warnings.filterwarnings("ignore", message=".*Roundtrip module could not be loaded.*")

sys.path.append("/Users/aschwanden1/min-venv-local/lib/python3.9/site-packages")
sys.path.append("/Users/aschwanden1/treescape")

import treescape as tr


def launchday_to_date(epoch_timestamp):
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
    year = date.year
    month = months[date.month - 1]
    day = f"{date.day:02d}"
    hours = f"{date.hour:02d}"
    minutes = f"{date.minute:02d}"
    seconds = f"{date.second:02d}"
    return f"{year}-{month}-{day} {hours}:{minutes}:{seconds}"


def convert_to_number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def analyze_data_distribution(tests, xaxis, node_name, line_metadata_name):
    print("=== DATA ANALYSIS FOR SPIKE INVESTIGATION ===")
    print(f"Analyzing node: {node_name}")
    print(f"X-axis: {xaxis}")
    print(f"Line metadata: {line_metadata_name}")
    print()

    # Collect raw data before aggregation
    raw_data = defaultdict(list)  # x_value -> list of (test_name, sum_value)
    test_data = defaultdict(lambda: ([], []))  # Same as MultiLine logic

    for test in tests:
        if hasattr(test, "perftree") and isinstance(test.perftree, dict):
            myx = test.metadata[xaxis]
            myx = convert_to_number(myx)
            test_name = test.metadata.get(line_metadata_name, "Unknown")

            for key, metrics in test.perftree.items():
                if key == node_name and "sum" in metrics:
                    if xaxis == "launchday" or xaxis == "launchdate":
                        formatted_x = launchday_to_date(myx)
                    else:
                        formatted_x = myx

                    sum_value = float(metrics["sum"])

                    # Store raw data
                    raw_data[formatted_x].append((test_name, sum_value))

                    # Store in test_data format (same as MultiLine)
                    test_data[test_name][0].append(formatted_x)
                    test_data[test_name][1].append(sum_value)

    # Analyze aggregation behavior
    print("=== RAW DATA ANALYSIS ===")
    x_positions = sorted(raw_data.keys())
    total_positions = len(x_positions)
    spike_position = int(total_positions * 0.1)  # Roughly 1/10th position

    print(f"Total x-axis positions: {total_positions}")
    print(
        f"Investigating position ~{spike_position} (1/10th): {x_positions[spike_position] if spike_position < total_positions else 'N/A'}"
    )
    print()

    # Check data density around the spike area
    start_idx = max(0, spike_position - 5)
    end_idx = min(total_positions, spike_position + 5)

    print("=== DATA DENSITY AROUND SPIKE AREA ===")
    for i in range(start_idx, end_idx):
        if i < len(x_positions):
            x_pos = x_positions[i]
            data_points = raw_data[x_pos]
            total_sum = sum(val for _, val in data_points)
            count = len(data_points)

            marker = " <-- SPIKE AREA" if i == spike_position else ""
            print(
                f"Position {i:3d}: {x_pos} | Count: {count:2d} | Total Sum: {total_sum:10.2f} | Avg: {total_sum/count:8.2f}{marker}"
            )

            # Show individual values if there are multiple data points
            if count > 1:
                for test_name, val in data_points:
                    print(f"    {test_name}: {val:10.2f}")

    print()

    # Check if there are more data points being summed at certain positions
    print("=== DATA POINT COUNT DISTRIBUTION ===")
    count_distribution = defaultdict(int)
    sum_distribution = defaultdict(list)

    for x_pos, data_points in raw_data.items():
        count = len(data_points)
        total_sum = sum(val for _, val in data_points)
        count_distribution[count] += 1
        sum_distribution[count].append(total_sum)

    for count in sorted(count_distribution.keys()):
        positions_with_count = count_distribution[count]
        avg_sum = sum(sum_distribution[count]) / len(sum_distribution[count])
        max_sum = max(sum_distribution[count])
        min_sum = min(sum_distribution[count])

        print(
            f"Positions with {count:2d} data points: {positions_with_count:3d} | Avg Sum: {avg_sum:10.2f} | Range: {min_sum:8.2f} - {max_sum:8.2f}"
        )

    print()

    # Find the highest sums
    print("=== TOP 10 HIGHEST SUMS ===")
    all_sums = []
    for x_pos, data_points in raw_data.items():
        total_sum = sum(val for _, val in data_points)
        all_sums.append((total_sum, x_pos, len(data_points)))

    all_sums.sort(reverse=True)
    for i, (total_sum, x_pos, count) in enumerate(all_sums[:10]):
        print(f"{i+1:2d}. {x_pos} | Sum: {total_sum:10.2f} | Data Points: {count}")

    return raw_data, test_data


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

    raw_data, test_data = analyze_data_distribution(
        alltests, xaxis, node_name, metadata_key
    )
