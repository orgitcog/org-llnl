#!/usr/gapps/spot/treescape-ven/bin/python

cali_file_loc = "/usr/gapps/spot/datasets/newdemo/test"
xaxis = "launchday"
metadata_key = "test"
processes_for_parallel_read = 15
initial_regions = ["main"]

import sys

sys.path.append("/usr/gapps/spot/treescape-ven/lib/python3.9/site-packages")
sys.path.append("/usr/gapps/spot/treescape")


import treescape as tr

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    caliReader = tr.CaliReader(cali_file_loc, processes_for_parallel_read)

    tsm = tr.TreeScapeModel(caliReader)
    # Always be sure to sort your data into some reasonable way.
    alltests = sorted(tsm, key=lambda x: x.metadata[xaxis])

    sl = tr.StackedLine()

    ##
    for testname in {t.metadata[metadata_key] for t in tsm}:
        print(testname)
        # render each test.  click on "Run Info" button to see flamegraph and metadata
        all_tests = [t for t in tsm if t.metadata[metadata_key] == testname]
        sl.exportSVG(
            "/g/g0/pascal/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )
