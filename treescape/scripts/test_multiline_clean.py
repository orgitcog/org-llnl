#!/Users/aschwanden1/min-venv/bin/python

cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test_plus_24c"
# cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test_plus_48"

# cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test"
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
    # Always be sure to sort your data into some reasonable way.
    alltests = sorted(tsm, key=lambda x: x.metadata[xaxis])

    grapher = tr.MultiLine(alltests)
    for region in initial_regions:
        grapher.plot_sums(xaxis, region, metadata_key)
