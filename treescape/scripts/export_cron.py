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

    ##
    for testname in sorted({t.metadata[metadata_key] for t in tsm}):
        print(testname)
        # render each test.  click on "Run Info" button to see flamegraph and metadata
        all_tests = [t for t in tsm if t.metadata[metadata_key] == testname]

        # sl.setYMin(230)
        # sl.setYMax(1500)
        sl.setXAggregation("avg")
        sl.setYAxis("sum")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("sum")
        sl.setYAxis("sum")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("avg")
        sl.setYAxis("max")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("sum")
        sl.setYAxis("max")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("sum")
        sl.setYAxis("avg")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("min")
        sl.setYAxis("min")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("max")
        sl.setYAxis("avg")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("topmax")
        sl.setYAxis("avg")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("topmax")
        sl.setYAxis("max")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("topmax")
        sl.setYAxis("min")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )

        sl.setXAggregation("topmax")
        sl.setYAxis("sum")

        sl.exportSVG(
            "/Users/aschwanden1/svg_imgs/",
            all_tests,
            "launchday",
            ["TimeIncrement", "LagrangeLeapFrog"],
            testname,
        )
