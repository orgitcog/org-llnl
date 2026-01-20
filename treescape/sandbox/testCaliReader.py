## - A summary graph of the performance changes of 'main' for all tests over time, which is intended to reveal performance shifts at a quick glance
## - For each test, a detailed stacked line graph that you can click to dig into the performance of individual code regions.
## - For each day in the stacked line graph for a test, you can expand either the metadata or a flamegraph showing the performance/metadata of that specific day.
##
## The relevant code objects for controlling this are documented below. As of this writing, this is intended as a friendly beta-test. We are interested in feedback
## and bug reports.
##
if __name__ == "__main__":
    cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test"
    #cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test_plus_24c"
    #cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test_plus_6"

    #cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test"
    xaxis = "launchday"
    metadata_key = "test"
    processes_for_parallel_read = 15
    initial_regions = ["main"]

    import sys

    sys.path.append("/Users/aschwanden1/min-venv-local/lib/python3.9/site-packages")
    sys.path.append("/Users/aschwanden1/github/treescape")

    import treescape as tr

    mytime = tr.MyTimer()

    ##
    ##You load your caliper files into a TreeScapeModel, which is essentially a python list of runs. Each list entry is one run. You can
    ##access that run's key/value metadata with the run.metadata dict.
    ##By sorting/filtering/aggregating that python list you can control what data the subsequent visualizations operate on.
    ##This demo will use the metadata kay 'launchday' to sort dataset. It will then use the metadata key 'test' (which represents the test name
    ##associated with that run) to filter the data into different graphs. That will show performance over time per test.
    ##
    inclusive_strs = ["min#inclusive#sum#time.duration",
                                      "max#inclusive#sum#time.duration",
                                      "avg#inclusive#sum#time.duration",
                                      "sum#inclusive#sum#time.duration"]
    mytime.mark('before cali reader')
    caliReader = tr.CaliReader( cali_file_loc, processes_for_parallel_read)
    mytime.mark('after cali')

    tsm = tr.TreeScapeModel( caliReader)
    #Always be sure to sort your data into some reasonable way.
    alltests = sorted(tsm, key=lambda x: x.metadata[xaxis])
