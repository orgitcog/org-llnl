# this creates the test data for the test_plus_24a dataset.
import os
import re
from pathlib import Path

# Constants
SECONDS_IN_MONTH = 2592000  # Approx seconds in 30 days
TIMESTAMP_START = 1609459200  # 2021-01-01
TIMESTAMP_END = 1614556800  # 2021-02-28

# Regex pattern to match standalone 9-10 digit integers (UNIX timestamps)
# This pattern ensures we don't match parts of decimal numbers
# Use word boundaries and negative lookbehind/lookahead for decimal points
timestamp_pattern = re.compile(r"(?<!\d)\b\d{9,10}\b(?!\.\d)")


def shift_timestamps(content: str, shift_multiplier: int) -> str:
    """Shift timestamps in a string by X months if they are within the Jan–Feb 2021 window."""

    def shift_match(match):
        ts = int(match.group())
        if TIMESTAMP_START <= ts < TIMESTAMP_END:
            return str(ts + shift_multiplier * SECONDS_IN_MONTH)
        return str(ts)

    return timestamp_pattern.sub(shift_match, content)


def process_cali_files(directory: str):
    directory = Path(directory)
    cali_files = list(directory.glob("*.cali"))

    print(f"Found {len(cali_files)} cali files.")

    for cali_file in cali_files:
        original_text = cali_file.read_text(errors="ignore")

        # Create 24 shifted versions (1 to 24 months)
        # Skip shift=0 to avoid duplicating the original time window
        for shift in range(1, 80):  # 1 to 24 months instead of 0 to 23
            shifted_text = shift_timestamps(original_text, shift)
            new_filename = f"{cali_file.stem}_shift{shift}.cali"
            new_filepath = cali_file.parent / new_filename
            new_filepath.write_text(shifted_text)
            print(f"Written: {new_filepath} (shifted by {shift} months)")


# Example usage:
process_cali_files(
    "/Users/aschwanden1/datasets/newdemo/test_plus_80"
)  # Use original test data
# This will create shifted files without duplicating the original time window
