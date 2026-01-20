#!/Users/aschwanden1/min-venv/bin/python

cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test"
xaxis = "launchday"
metadata_key = "test"
processes_for_parallel_read = 15
initial_regions = ["main"]

import sys

sys.path.append("/Users/aschwanden1/min-venv/lib/python3.9/site-packages")
sys.path.append("/Users/aschwanden1/treescape")

import treescape as tr

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    caliReader = tr.CaliReader(cali_file_loc, processes_for_parallel_read)

    tsm = tr.TreeScapeModel(caliReader)
    # Always be sure to sort your data into some reasonable way.
    alltests = sorted(tsm, key=lambda x: x.metadata[xaxis])

    sl = tr.StackedLine()

    # Test topmax aggregation specifically
    for testname in sorted({t.metadata[metadata_key] for t in tsm}):
        print(f"Testing topmax for: {testname}")
        # render each test.  click on "Run Info" button to see flamegraph and metadata
        all_tests = [t for t in tsm if t.metadata[metadata_key] == testname]

        sl.setXAggregation("topmax")
        sl.setYAxis("sum")

        print(f"About to generate max SVG for {testname} with {len(all_tests)} tests")
        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/", all_tests, "launchday", ["main"], testname
        )

        print(f"Generated max SVG for {testname}")

        sl.setXAggregation("max")
        sl.setYAxis("sum")

        print(
            f"About to generate topmax SVG for {testname} with {len(all_tests)} tests"
        )
        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/", all_tests, "launchday", ["main"], testname
        )

        print(f"Generated topmax SVG for {testname}")
        break  # Just test one for now
