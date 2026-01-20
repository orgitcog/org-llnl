# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from .TreeScapeModel import TreeScapeModel
from .StackedLinePython import StackedLinePython
import os


class StackedLine:

    # setXAggregation params
    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    TOPMAX = "topmax"

    # components
    LINEGRAPHS = "linegraph"
    FLAMEGRAPHS = "flamegraph"

    # metrics
    INCLUSIVE_AVG = "avg#inclusive#sum#time.duration"

    def __init__(self):

        self.nameOfLinesToPlot = None
        self.yMax = None
        self.yMin = None
        self.width = None

        #  Just user doesn't set a height we don't want it to be super tall.
        self.height = 400

        self.xaxis_agg = "sum"
        self.container_id = 0

        # default of launchdate
        self.setXAxis("launchdate")
        self.setYAxis("avg")

        self.setComponents([StackedLine.LINEGRAPHS, StackedLine.FLAMEGRAPHS])

    def setXAxis(self, xaxis_name: str):
        """
        You can set the X axis of the stacked line chart you want to see.  The name given
        must be present as an attribute in the meta dataset.

        :param xaxis_name: examples are jobsize, problem_size, launchdate, etc.
        :return:
        """
        self.xaxis_name = xaxis_name

    def setYAxis(self, yaxis_name: str):
        """
        You can set the Y axis of the stacked line chart you want to see.  the name given
        must be present as an attribute in the meta dataset.

        :param yaxis_name:
        :return:
        """

        valid_options = {
            StackedLine.SUM,
            StackedLine.AVG,
            StackedLine.MAX,
            StackedLine.MIN,
        }

        if yaxis_name not in valid_options:
            print(f"Validation Error: '{yaxis_name}' is not a valid yAxis option.")

        self.yaxis_name = yaxis_name

    def setXAggregation(self, aggregation: str):
        """
        This determines how multiple values are aggregated

        :param aggregation: examples are sum, max, min, average.
        :return:
        """

        valid_options = {
            StackedLine.SUM,
            StackedLine.AVG,
            StackedLine.MAX,
            StackedLine.MIN,
            StackedLine.TOPMAX,
        }

        if aggregation not in valid_options:
            print(
                f"Validation Error: '{aggregation}' is not a valid aggregation option."
            )

        self.xaxis_agg = aggregation

    def errorOut(self, str):
        # Bold red text
        print("\033[1;31m" + str + "\033[0m")

    def setComponents(self, components):

        if not isinstance(components, list):
            self.errorOut(f"Components needs to be a list: {components}")
            return False

        valid_options = {StackedLine.LINEGRAPHS, StackedLine.FLAMEGRAPHS}

        invalid_values = [value for value in components if value not in valid_options]

        if invalid_values:
            self.errorOut(
                f"Validation Error: The following values are not valid component options: {invalid_values}"
            )
            return False

        import json

        jstr = json.dumps(components, indent=4)
        self.components = ";ST.components = " + jstr + ";"

    def setDrillLevel(self, nameOfLinesToPlot):
        """

        You can set the names of the lines you wish to plot

        :param nameOfLinesToPlot: provide an array of annotation (str) you wish to plot
        :return


        """
        if not isinstance(nameOfLinesToPlot, list):
            self.errorOut(f"DrillLevel needs to be a list {nameOfLinesToPlot}")
            return False

        self.nameOfLinesToPlot = nameOfLinesToPlot

    def setYMax(self, yMax):
        self.yMax = yMax

    def setYMin(self, yMin):
        self.yMin = yMin

    def setWidth(self, width):
        self.width = width

    def setHeight(self, height):
        self.height = height

    def get_PJ_bus(self, xaxis):
        """

        For internal use.  The PJ bus allows Python variables to be sent down to the
        javascript layer.

        :return: An object containing the things required for plotting.

        """
        self.xaxis_name = xaxis
        self.entireForest = self.tsm.get_entire_tsm()

        cm = self.entireForest["childrenMap"]
        iterator = iter(cm)
        theKey = "not present"

        perftree = self.entireForest["nodes"][0]["perftree"]

        while theKey not in perftree:
            try:
                # Get the next key from the iterator
                theKey = next(iterator)
            except StopIteration:
                # Handle the case where the iterator is exhausted
                print("No more keys in 'cm'. Exiting loop.")
                break

        # print(perftree)

        if self.nameOfLinesToPlot is None:
            self.setDrillLevel([theKey])

        ret = {
            "entireForest": self.entireForest,
            "nameOfLinesToPlot": self.nameOfLinesToPlot,
            "xaxis_name": self.xaxis_name,
        }

        if self.width is not None:
            ret["set_width"] = self.width

        if self.height is not None:
            ret["set_height"] = self.height

        if self.yMax is not None:
            ret["yMax"] = self.yMax

        if self.yMin is not None:
            ret["yMin"] = self.yMin

        ret["xaxis_agg"] = self.xaxis_agg
        ret["yAxis"] = self.yaxis_name

        return ret

    def render(self, tsm_or_list, **kwargs):
        """
        Render Chart and flamegraph combo.

        :param model: this specifies the data you want to plot.  it's required because we need the profile and th_ens to plot.
        :param make_stub: tells the chart to make a stub about 500 data points on the x axis.
        :param annotations: tells which nodes you want to see line graphed, for example: LagrangeLeapFrog or main
        :param components: say whether you want the 'linegraph' or 'flamegraph' or both.

        if tsm_or_list is just a list then it won't have the childrenMap
        Without the childrenMap, it can't make the flamegraph or come up with the color mapping
        or do the drill down.
        Nevertheless it will degrade gracefully.

        :return:
        """
        if isinstance(tsm_or_list, TreeScapeModel):
            self.tsm = tsm_or_list
        else:
            #  it's a list so make the TreeScapeModel.
            reader = None
            for r in tsm_or_list:
                if r.read_from:
                    reader = r.read_from
                    break
            self.tsm = TreeScapeModel(reader, tsm_or_list)

        for key, value in kwargs.items():
            print(f"{key}: {value}")

        if "drill_level" in kwargs:
            self.setDrillLevel(kwargs["drill_level"])

        if "xaggregation" in kwargs:
            self.setXAggregation(kwargs["xaggregation"])

        if "components" in kwargs:
            self.setComponents(kwargs["components"])

        if "ymin" in kwargs:
            self.setYMin(kwargs["ymin"])

        if "ymax" in kwargs:
            self.setYMax(kwargs["ymax"])

        make_stub = ""
        if "make_stub" in kwargs and kwargs["make_stub"] == 1:
            make_stub = ";ST.MAKE_STUB = true;"

        xaxis = self.xaxis_name
        if "xaxis" in kwargs:
            xaxis = kwargs["xaxis"]

        file_content = ""

        files = [
            "jquery.js",
            "chartmin.js",
            "SelectView.js",
            "FlameGraphModel.js",
            "FlameGraphView.js",
            "stacked.js",
            "TreeScapeModel.js",
            "CurrentPlotModel.js",
            "StackedLine.js",
            "ModifyPJ_BusStubModel.js",
            "jmd5.js",
            "MultiLineChart.js",
            # "vue.js",
            # "incrementExample.js"
        ]

        deploy_directory = os.path.dirname(os.path.abspath(__file__)) + "/../"

        for file_path in files:
            # Open the file and read its contents
            with open(deploy_directory + "js/" + file_path, "r") as file:
                file_content += file.read()

        import json

        from IPython.display import display, Javascript, HTML

        bus_serialization = self.get_PJ_bus(xaxis)
        jds = json.dumps(bus_serialization)
        code = file_content.replace("PJ_BUS_REPLACE_JS", jds)

        import random
        import string

        self.container_id = "".join(
            random.choices(string.ascii_letters + string.digits, k=10)
        )

        container_var = ';var container_id = "' + str(self.container_id) + '";'

        code = code + make_stub + self.components + container_var

        styleCSS = open(deploy_directory + "stacked.css").read()
        # display(HTML('<link rel="stylesheet" href="../stacked.css">' ))
        display(HTML("<style>" + styleCSS + "</style>"))

        # Display HTML container FIRST, before JavaScript
        # This ensures the DOM element exists when JavaScript tries to find it
        html_content = (
            '<div container_id="'
            + str(self.container_id)
            + '" class="stacked-line-component"></div>'
        )
        display(HTML(html_content))

        #  this does not work.
        #  on localhost we have to use custom.js
        # display(HTML('<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>'))

        # Display JavaScript AFTER the HTML container
        display(Javascript(code))

    def getLibs(self):

        with open("../js/md5.js", "r") as file:
            file_content = file.read()

        return file_content

    def getVersion(self):
        return "1.23"

    def exportSVG(self, directory, all_tests, metavar, node_names, testname):

        slp = StackedLinePython(directory, all_tests, testname)
        slp.setYMax(self.yMax)
        slp.setYMin(self.yMin)
        slp.setXAggregation(self.xaxis_agg)
        slp.setYAxis(self.yaxis_name)

        slp.plot_sums(metavar, node_names)
