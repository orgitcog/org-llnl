(function ($) {
    //var currentCell = document.querySelector('.jp-Cell'); // Find the first cell element
    var currentCell = $(document.currentScript).closest('.jp-Cell');
    currentCell = $('.jp-Cell').first();

    // Define the component
    function StackedLine(element2, PJ_Bus) {

        //  used for testing of removing nodes.
        //this.removeRandomPerftreeEntry( PJ_Bus.entireForest );

        this.$conElement = $(element2);
        this.count = 0;

        console.dir(PJ_Bus);
        this.setEntire_(PJ_Bus);
        this.setNameOfLinesToPlot_(PJ_Bus.nameOfLinesToPlot);
        this.setXAxis_(PJ_Bus.xaxis_name);
        this.setXAxisAggregate_(PJ_Bus.xaxis_agg);
        this.setYAxis_(PJ_Bus.yAxis);

        this.PJ_Bus = PJ_Bus;

        //  For our test data this will likely be "main"
        this.topNode = this.getTopNode_( PJ_Bus.entireForest.childrenMap );
        console.log('topNode = ' + this.topNode);

        this.init();
    }


    StackedLine.prototype.getTopNode_ = function(childrenMap) {
        const allNodes = new Set(Object.keys(childrenMap));
        const childNodes = new Set();

        for (const children of Object.values(childrenMap)) {
            for (const child of children) {
                childNodes.add(child);
            }
        }

        // Topmost nodes are those that are never listed as a child
        const topMostNodes = [...allNodes].filter(node => !childNodes.has(node));

        return topMostNodes;
    }

    StackedLine.prototype.lastPlotsPlotted_ = [];
    StackedLine.prototype.nameOfLinesToPlot_ = false;
    StackedLine.prototype.ef_ = {};

    StackedLine.prototype.init = function () {
        this.render();
    };

    StackedLine.prototype.getNameOfLinesToPlot_ = function () {
        return this.nameOfLinesToPlot_;
    };

    StackedLine.prototype.setNameOfLinesToPlot_ = function (names) {

        if (!Array.isArray(names)) {
            alert('setNameOflinesToPlot needs an array.');
        } else {
            this.nameOfLinesToPlot_ = names;
        }
    };

    StackedLine.prototype.removeRandomPerftreeEntry = function (entireForest) {

        if (!entireForest || !Array.isArray(entireForest.nodes)) {
            console.error("Invalid data structure");
            return;
        }

        entireForest.nodes.forEach(node => {
            if (node.perftree && typeof node.perftree === 'object') {
                let keys = Object.keys(node.perftree);
                if (keys.length > 0) {
                    let randomKey = keys[Math.floor(Math.random() * keys.length)];
                    delete node.perftree[randomKey];
                    console.log('delete randomKey: '+randomKey);
                }
            }
        });
    };

    //  Each jupyter notebook output area get's it's own global scope.
    //  And yet all the DOM elements are shared.
    StackedLine.prototype.getNewCanvasChartID = function () {

        for (var x = 0; x < 25; x++) {
            var numCh = $('[chartIdSingular="' + x + '"]').length;
            if (numCh === 0) {
                return x;
            }
        }
    };


    StackedLine.prototype.setXAxis_ = function (xaxis) {
        this.xAxis_ = xaxis;
    };

    StackedLine.prototype.getXAxis_ = function () {
        return this.xAxis_;
    };

    StackedLine.prototype.setXAxisAggregate_ = function (agg) {
        this.xAgg_ = agg;
    };

    StackedLine.prototype.getXAxisAggregate_ = function () {

        if (!this.xAgg_) {
            console.log("Warning: xAxisAggregate is not set.");
        }
        return this.xAgg_;
    };

    StackedLine.prototype.setYAxis_ = function (yaxis) {
        console.log('yAxis = ' + yaxis);
        this.yAxis_ = yaxis;
        this.yAxisTitle = yaxis + ' time/rank';
    };

    StackedLine.prototype.getYAxis_ = function () {

        if (!this.yAxis_) {
            console.log("Warning: yAxis is not set.");
        }
        return this.yAxis_;
    };

    StackedLine.prototype.getXAxisTypeInf_ = function (xaxis_name) {

        var meta_globals = this.PJ_Bus.entireForest.meta_globals;

        //  TODO: figure out what type the xAxis is dynamically.
        var xaxis_type = "default"

        if (meta_globals[xaxis_name]) {
            xaxis_type = meta_globals[xaxis_name]
        }

        return xaxis_type;
    };

    StackedLine.prototype.interpolateArray = function (arr, factor, xaxis) {
        // Validate input
        if (!Array.isArray(arr) || arr.length < 2 || typeof factor !== 'number' || factor <= 0) {
            throw new Error('Invalid input');
        }

        const interpolatedArray = [];
        const numInterpolatedPoints = (arr.length - 1) * factor;

        var agg = this.getXAxisAggregate_();

        for (let i = 0; i < arr.length - 1; i++) {
            for (let j = 0; j < factor; j++) {

                const t = j / factor;

                var x = (1 - t) * arr[i][agg] + t * arr[i + 1][agg];

                if (xaxis) {
                    x = (1 - t) * arr[i] + t * arr[i + 1];
                }

                interpolatedArray.push(x);
            }
        }

        // Add the last point of the original array
        interpolatedArray.push(arr[arr.length - 1][agg]);

        console.dir(interpolatedArray);

        return interpolatedArray;
    }


    StackedLine.prototype.interpolate_x_and_y_ = function (node) {

        this.num_points = node.xaxis.length;
        var interpolation_factor = 300 / this.num_points;

        if (typeof node.xaxis[0] === "string") {
            return node;
        }

        var iay = this.interpolateArray(node.ydata, interpolation_factor);
        var new_arr_y = [];

        for (var j = 0; j < iay.length; j++) {

            new_arr_y[j] = iay[j];
        }

        return {
            name: node.name,
            xaxis: this.interpolateArray(node.xaxis, interpolation_factor, true),
            ydata: iay
        }
    };

    //  xaxis_name could be like "launchday" or "jobsize", some index in metadata.
    StackedLine.prototype.getTopAggs_ = function( xaxis_name, yaxis ) {

        var topAggs = [];

        for (var n = 0; n < this.ef_.nodes.length; n++) {

            var ita = this.ef_.nodes[n];
            var xval = ita.metadata[xaxis_name];
            var avg = 0;

            //  this.topNode will probably be "main" for our test data.
            //  Or whatever you set the original setDrillLevel to.
            if (ita.perftree && ita.perftree[this.topNode]) {
                avg = parseFloat(ita.perftree[this.topNode][ yaxis ]);
            }

            topAggs[xval] = topAggs[xval] || [];
            topAggs[xval].push(avg);
        }

        return topAggs;
    };


    StackedLine.prototype.getPlotByName_ = function (node_name, xaxis_name, aggType ) {

        function sumArray(arr) {
            return arr.reduce((acc, val) => acc + val, 0);
        }

        function minArray(arr) {
            return Math.min(...arr);
        }

        function maxArray(arr) {
            return Math.max(...arr);
        }

        function topmaxArray(arr, selectedIndex) {
            return arr[selectedIndex] || 0;
        }

        function avgArray(arr) {
            return arr.length === 0 ? 0 : sumArray(arr) / arr.length;
        }

        function indexOfMax(arr) {
          if (arr.length === 0) return -1; // handle empty array

          let maxIndex = 0;
          for (let i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) {
              maxIndex = i;
            }
          }
          return maxIndex;
        }

        var xaxis = [];
        var aggs = {};
        var yaxis = this.getYAxis_();

        console.log('YAxis = ' + yaxis);

        var topAggs = this.getTopAggs_( xaxis_name, yaxis );


        for (var n = 0; n < this.ef_.nodes.length; n++) {

            var ita = this.ef_.nodes[n];
            var xval = ita.metadata[xaxis_name];
            var avg = 0;

            if (ita.perftree && ita.perftree[node_name]) {
                avg = parseFloat(ita.perftree[node_name][ yaxis ]);
            }

            aggs[xval] = aggs[xval] || [];
            aggs[xval].push(avg);
        }

        var xarr = [];
        var yarr = [];

        console.dir(topAggs);

        for (var metadataXVal in aggs) {

            //  metadataXVal could be 1663027200 (for example if xaxis is launchday, or launchdate)
            var values = aggs[metadataXVal];
            var u;
            var topVals = topAggs[metadataXVal];
            var topValIndex = indexOfMax(topVals);

            switch (aggType) {
                case "min":
                    u = minArray(values);
                    break;
                case "max":
                    u = maxArray(values);
                    break;
                case "topmax":
                    u = topmaxArray(values, topValIndex);
                    break;
                case "avg":
                    u = avgArray(values);
                    break;
                case "sum":
                default:
                    u = sumArray(values);
                    break;
            }

            xarr.push(metadataXVal);
            yarr.push(u);
        }

        var node = {
            xaxis: xarr,
            ydata: yarr,
            name: node_name
        };

        return {
            xaxis_name: xaxis_name,
            xaxis_type: this.getXAxisTypeInf_(xaxis_name),
            nodes: this.interpolate_x_and_y_(node)
        };
    }


    StackedLine.prototype.getNodeTreeNode_ = function (tree, name) {

        for (var x = 0; x < tree.length; x++) {
            if (name === tree[x].name) {
                return tree[x];
            }
        }
    }


    StackedLine.prototype.stackAddY_ = function (amended_y, nxy) {

        var agg = this.getXAxisAggregate_();

        for (var i in nxy.ydata) {

            amended_y.ydata[i] = amended_y.ydata[i] || 0;
            amended_y.ydata[i] = amended_y.ydata[i] + nxy.ydata[i];
        }

        amended_y.name = nxy.name;
        amended_y.xaxis = nxy.xaxis;
    };


    StackedLine.prototype.getPlotNameClicked_ = function (xClick, yClick) {

        var currPlots = this.getNameOfLinesToPlot_();
        var xaxis_name = this.getXAxis_();
        var aggType = this.getXAxisAggregate_();  // "sum", "min", "max", or "avg"


        //var nodeTree = this.ef_.nodes[currX];
        var candidates = [];
        var amended_y = {
            name: "",
            ydata: []
        };

        for (var plotIdx in currPlots) {

            var plotName = currPlots[plotIdx];
            //var nxy = this.getNodeTreeNode_(nodeTree, plotName);
            var nxy = this.getPlotByName_(plotName, xaxis_name, aggType);

            this.stackAddY_(amended_y, nxy.nodes);

            var candidate = this.isXYmatch_(amended_y, xClick, yClick);

            if (candidate) {
                candidates.push(candidate);
            }
        }

        candidates.sort(this.sortByLowestY);

        if (candidates.length === 0) {
            //  Mean user clicked outside of ALL plot boundaries.
            console.log('clicked outside all plot boundaries.');
            return false;
        }

        return candidates[0].name;
    };


    StackedLine.prototype.sortByLowestY = (a, b) => {
        return a.y_at_this_xClick - b.y_at_this_xClick;
    };


    StackedLine.prototype.isXYmatch_ = function (nxy, xClick, yClick) {

        var xaxis = nxy.xaxis;
        var yaxis = nxy.ydata;

        for (var i = 0; i < xaxis.length; i++) {

            if (xClick >= xaxis[i] && xClick < xaxis[i + 1]) {

                var xdiff = xaxis[i + 1] - xaxis[i];
                var percent_way_thru = (xClick - xaxis[i]) / xdiff;

                var y0 = yaxis[i];
                var y1 = yaxis[i + 1];
                var yrange = y1 - y0;
                var add_y = yrange * percent_way_thru;
                var y_at_this_xClick = y0 + add_y;

                //console.log('yat=' + y_at_this_xClick + '   yClick=' + yClick);

                if (yClick <= y_at_this_xClick) {
                    return {
                        name: nxy.name,
                        y_at_this_xClick: y_at_this_xClick
                    }
                }
            }
        }
    };


    StackedLine.prototype.popupAttributes = function (index) {

        var nodes = ST.self.ef_.nodes;
        var per_index = Math.round((nodes.length / ST.self.numPoints) * index);
        var meta = nodes[per_index].metadata;

        var popa = this.$conElement.find('.popupAttributes');

        if (popa.length === 0) {
            this.$conElement.append();
        }

        var st_show = "";

        for (var x in meta) {

            var convert_str = "";
            if (x === 'launchday') {
                convert_str = "    - " + convertEpochToReadableString(meta[x] * 1000);
            }
            st_show += "<div>" + x + ": " + meta[x] + convert_str + "</div>";
        }

        popa.find('.middle_text').html(st_show);

        this.$conElement.find('.tab').off('click').on('click', this.handleTabClick);
    };


    StackedLine.prototype.handleTabClick = function () {

        var par = $(this).parent();
        par.find('.tab').removeClass('active');
        $(this).addClass('active');

        var is_metadata_tab = $(this).hasClass('metadataTab');
        var grandpa = par.parent();

        if (is_metadata_tab) {
            grandpa.find('.popupAttributes').show();
            grandpa.find('.flameContainer').hide();
        } else {
            grandpa.find('.popupAttributes').hide();
            grandpa.find('.flameContainer').show();
        }
    };


    StackedLine.prototype.handleHover = function (index) {

        this.popupAttributes(index);

        var x = 0;
        //  handle case where user hovers over or default.
        var tree = ST.FlameGraphModel.get(ST.self.ef_, index, "avg");
        ST.FlameGraphView.set(tree.tree);

        var view = ST.FlameGraphView.get( tree.max );
        this.$conElement.find('.flamegraph').html(view);
    };

    StackedLine.prototype.get_options = function (plot, nameOfLinesToPlot) {

        var xAxisName = this.getXAxis_();
        var xAxisType = plot.xaxis_type;
        var ef = this.ef_;
        var self = this;
        ST.self.numPoints = this.num_points;


        var options = {
            responsive: true,
            maintainAspectRatio: false,
            tooltips: {
                mode: 'index',
                intersect: false
            },
            hover: {
                mode: 'index',
                intersect: false
            },
            "scales": {
                "x": {
                    "ticks": {
                        "fontSize": 16,
                        callback: function (value, index, values) {

                            if (xAxisType === 'date') {
                                var dt = plot.nodes.xaxis[value] * 1000;

                                return convertEpochToReadableString(dt);
                            }

                            var tickVal = plot.nodes.xaxis[value];

                            //console.log('tick index: ' + index + '  value: ' + value + '  tv=' + tickVal);

                            if (tickVal > 5) {
                                tickVal = Math.round(tickVal);
                            }
                            if (typeof tickVal === "number" && tickVal <= 5) {
                                tickVal = Math.round(tickVal * 100) / 100;
                            }

                            if (tickVal === parseInt(tickVal)) {
                                return tickVal;
                            } else {
                                return "";
                            }
                        }
                    },
                    "title": {
                        display: true,
                        text: xAxisName
                    }
                },
                "y": {
                    "stacked": true
                }
            },
            onClick: function (event, elements) {

                //  code for passing things back to python.
                //  var kernel = Jupyter.notebook.kernel;
                var varName = 'jsonExample'
                //kernel.execute(varName + '={"name":"Epsilon", "type":"star", "value": ' + Math.random() + '}')

                var key0 = Object.keys(myChart)[0];
                console.log('key0 = ' + key0 + '  event.x = ' + event.x);

                var xCoordinate = myChart[key0].scales['x'].getValueForPixel(event.x);
                var yCoordinate = myChart[key0].scales['y'].getValueForPixel(event.y);
                var xVal = plot.nodes.xaxis[xCoordinate];

                //console.log('Clicked at X:' + xCoordinate + '  ' + yCoordinate + '  Xval = ' + xVal);
                var points = this.getElementsAtEventForMode(event, 'y', {intersect: false}, true);

                var chartClickedName;

                if (isNaN(xVal)) {
                    //  THIS should only be used for Xaxis that are strings because it's not accurate when there's too few X ais points.
                    var dIndex = 0;

                    if (points.length) {
                        dIndex = points[0].datasetIndex;
                    }

                    chartClickedName = nameOfLinesToPlot[dIndex];
                } else {
                    //  X Axis is a Number.
                    //  Currently, our interpolation only support numeric.
                    chartClickedName = ST.self.getPlotNameClicked_(xVal, yCoordinate);
                }

                if (chartClickedName) {
                    ST.self.drillDown(chartClickedName);
                }
                //  else either it's the last layer OR you clicked outside all the charts.
            },
            onHover: function (event, chartElement) {
                // Check if the cursor is hovering over any part of the chart
                if (chartElement && chartElement.length > 0) {
                    // Display information about the hovered data point
                    var datasetIndex = chartElement[0].datasetIndex;
                    var index = chartElement[0].index;

                    //console.log('Hovered over Dataset ' + datasetIndex + ', Index ' + index + '  Per_index = ' + per_index + ', Value ' + value);
                    ST.self.handleHover(index);
                } else {
                    // Cursor is not over any part of the chart
                    var x = event.x;
                    var y = event.y;

                    //console.log('aa Mouse coordinates - X: ' + x + ', Y: ' + y);
                    var key0 = Object.keys(myChart)[0];

                    if (myChart[key0]) {

                        var xValue = myChart[key0].scales['x'].getValueForPixel(event.x);
                        console.log('X Value on the X axis: ' + xValue);

                        //document.getElementsByClassName("ver1")[0].innerHTML = xValue;

                        if (ST.components.indexOf('flamegraph') > -1) {
                            var view = ST.FlameGraphView.get(xValue);
                            ST.self.$conElement.find('.flamegraph').html(view);
                        }
                        //document.getElementsByClassName("flamegraph")[0].innerHTML = view; //ST.FlameGraphView.get(xValue);
                    }
                }
            },
            onMouseMove: function (event, chartEle) {

                console.log('chartEl:');
                console.dir(chartEle);
                // Display X and Y coordinates of the mouse position
                var x = event.x;
                var y = event.y;

                console.log('Mouse coordinates - X: ' + x + ', Y: ' + y);
            },
            "plugins": {
                "title": {
                    display: true,
                    text: this.yAxisTitle,
                    position: 'left'
                },
                "tooltip": {
                    "callbacks": {

                        "label": function (context) {
                            var label = context.dataset.label || '';

                            if (label) {
                                label += ': ';
                            }
                            label += context.parsed.y;

                            return "Meta: " + label;
                        },
                        // Title will be empty for this example
                        "title": function (context) {
                            return 'aff';
                        }
                    }
                }
            }
        };

        if (ST.self.PJ_Bus.yMax !== undefined) {
            options.scales.y.max = ST.self.PJ_Bus.yMax;
        }

        if (ST.self.PJ_Bus.yMin !== undefined) {
            options.scales.y.min = ST.self.PJ_Bus.yMin;
        }

        return options;
    };


    StackedLine.prototype.render_flamegraph = function () {

        var tree = ST.FlameGraphModel.get(this.ef_, 0, "avg");
        console.dir(tree);

        ST.FlameGraphView.set(tree.tree);

        if (ST.components.indexOf('flamegraph') > -1) {
            var view = ST.FlameGraphView.get(tree.max);
            ST.self.$conElement.find('.flamegraph').html(view);
        }
    };


    StackedLine.prototype.render = function () {

        ST.self = this;
        var ft = this.$conElement.find('.flamegraphTab');
        this.$isFlamegraph = ft.hasClass('active');

        var newChartID = this.getNewCanvasChartID();
        var flame = ST.self.get_flame_html();
        var groupByGraphs = ['a'];

        var set_width = this.PJ_Bus.set_width ? ("width: " + this.PJ_Bus.set_width + "px;") : "";
        var set_height = this.PJ_Bus.set_height ? ("height: " + this.PJ_Bus.set_height + "px;") : "";
        var set_height_outer = this.PJ_Bus.set_height ? ("height: " + (this.PJ_Bus.set_height + 40) + "px;") : "";


        if (ST.components.indexOf('linegraph') > -1) {

            var canvases = "";

            for (var x = 0; x < groupByGraphs.length; x++) {
                var subId = groupByGraphs[x];

                canvases += `<canvas chartIdSingular="` +
                    newChartID + `" id="canvasChart` + newChartID + subId + `">s</canvas>`;
            }

            //  <input type='button' class='downloadChart' value='DOWNLOAD'></input>

            var lineChartContainer = this.$conElement.find('.lineChartContainer');
            if (lineChartContainer.length > 0 && false) {
                lineChartContainer.remove('canvas');
                lineChartContainer.find('selectContainer').append(canvases);
            } else {

                this.$conElement.html(
                    `<div class="selectContainer">
                    <input type='button' class='drillUp' value='BACK'></input>                    
                    <input type='button' class='toggle-open-close' value='RUN INFO'></input>` +
                    `</div>` +
                    `<div class="lineChartContainer" style="` + set_width + set_height + `">` +
                    canvases +
                    `</div>` +
                    '<div class="button_and_bellow">' +
                    //'<div class="toggle-bellow-section"><span></span><span></span></div>' +
                    '<div class="bellow-section">' +
                    "<div class='tab_header'>" +
                    "<div class='tab active metadataTab'>Metadata</div>" +
                    "<div class='tab flamegraphTab'>Flamegraph</div>" +
                    "</div>" +
                    "<div class='popupAttributes'>" +
                    "<div class='middle_text'></div>" +
                    "</div>" +
                    "<div class='flamegraphTabContainer'></div>" +
                    flame + '</div>' +
                    '</div>');
            }
        } else {
            this.$conElement.html(flame);
        }

        ST.self.render_flamegraph();

        this.bindEvents();

        for (var x = 0; x < groupByGraphs.length; x++) {

            var subId = groupByGraphs[x];
            ST.self.setup_my_chart(newChartID + subId);
        }

        ST.self.handleHover(0);
    };


    StackedLine.prototype.downloadChart = function () {

        //ctx.fillStyle = 'white'; // Set background to white

        const canvas = document.getElementById('canvasChart0a');
        const ctx = canvas.getContext('2d');

        // Create a white background
        function setCanvasBackground() {
            ctx.fillStyle = 'white'; // Set background to white
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        //setCanvasBackground();
        const image = canvas.toDataURL('image/png');

        const a = document.createElement('a');
        a.href = image;
        a.download = 'chart.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }


    StackedLine.prototype.toggleContent = function () {
        var content = this.$conElement.find('.toggle-bellow-section');

        if (content.hasClass('open')) {
            console.log('open toggle');
            content.removeClass('open');
        } else {
            content.addClass('open');
        }

        ST.self.handleHover(0);
    }


    StackedLine.prototype.setup_my_chart = function (chId) {

        var nameOfLinesToPlot = this.getNameOfLinesToPlot_();
        var xaxis_name = this.getXAxis_();

        var aggType = this.getXAxisAggregate_();  // "sum", "min", "max", or "avg"
        console.log('setup my chart: ' + xaxis_name);

        //  in theorey they should all have the same x axis
        //  so it shouldn't matter which plot we get the x axis from.
        var plotName = nameOfLinesToPlot[0];
        var plot = this.getPlotByName_(plotName, xaxis_name, aggType);

        ST.self.typecast_xaxis(plot);

        console.log('plot:' + plot.nodes.xaxis.length);

        var options = ST.self.get_options(plot, nameOfLinesToPlot);
        var data = {
            "labels": plot.nodes.xaxis, //[1, 2, 3, 4, 5, 6, 7, 8],
            "datasets": ST.self.get_data_sets(nameOfLinesToPlot, xaxis_name, aggType)
        };


        if (myChart[chId]) {
            myChart[chId].destroy();
        }

        var ctx = document.getElementById("canvasChart" + chId).getContext('2d');
        //ctx.fillStyle = 'white'; // Set background to white

        myChart[chId] = new Chart(ctx, {
            "type": "line",
            "data": data,
            "options": options
        });

        console.log('Rendering myChart with ' + chId);
    };


    StackedLine.prototype.make_png_image = function (chartId) {

        const a = document.createElement('a');
        a.href = myChart[chartId].toBase64Image();
        a.download = 'chart_' + chartId + '.png';
        a.click();
    };

    StackedLine.prototype.get_flame_html = function () {
        return `
            <div class="flameContainer">
            <div class="flamegraph"></div>
            </div>`;
    };


    StackedLine.prototype.get_data_sets = function (nameOfLinesToPlot, xaxis, xagg) {

        var datasets = [];
        window.plots = [];

        for (var x = 0; x < nameOfLinesToPlot.length; x++) {

            var node_name = nameOfLinesToPlot[x];
            var plot = this.getPlotByName_(node_name, xaxis, xagg);

            var fillColor = ST.FlameGraphModel.get_color_by_name(node_name);
            window.plots.push(plot);

            datasets.push({
                "label": node_name,
                "fillColor": fillColor, // fillColors[x],
                "strokeColor": "rgba(151,187,205,1)",
                //"pointColor": "rgba(151,0,0,1)",
                "pointRadius": 0,
                //"pointStrokeColor": "#fff",
                //"pointHighlightFill": "#fff",
                //"pointHighlightStroke": "rgba(151,187,205,1)",
                "data": plot.nodes.ydata,
                "fill": "true",
                "backgroundColor": fillColor, // backgroundColors[x]
            });
        }

        for (var x = 0; x < datasets.length; x++) {
            datasets[x].fill = true;
        }

        return datasets;
    }


    StackedLine.prototype.typecast_xaxis = function (plot) {

        function isNumberString(value) {
            return !isNaN(value) && !isNaN(parseFloat(value));
        }

        if (typeof plot.nodes.xaxis[0] === 'string' && isNumberString(plot.nodes.xaxis[0])) {

            for (var x = 0; x < plot.nodes.xaxis.length; x++) {
                plot.nodes.xaxis[x] = +plot.nodes.xaxis[x];
            }
        }
    };


    StackedLine.prototype.setEntire_ = function (model) {

        this.ef_ = model.entireForest;
    };


    StackedLine.prototype.getPlotChildren_ = function (name) {
        return this.ef_.childrenMap[name];
    };


    StackedLine.prototype.drillDown = function (chartClickedName) {

        console.log("clicked on = " + chartClickedName);

        var lastEntry = this.lastPlotsPlotted_[this.lastPlotsPlotted_.length - 1];

        if (this.nameOfLinesToPlot_ && this.nameOfLinesToPlot_.length > 0 &&
            this.nameOfLinesToPlot_ !== lastEntry) {

            //  only cache if we're not hitting the last line.
            this.lastPlotsPlotted_.push(this.nameOfLinesToPlot_);
        }

        var nextPlot = this.getPlotChildren_(chartClickedName);

        if (nextPlot && nextPlot.length) {

            var reshow = this.button_and_bellow.hasClass('opened');
            console.log('reshow: ' + reshow);

            this.nameOfLinesToPlot_ = nextPlot;
            this.setNameOfLinesToPlot_(this.nameOfLinesToPlot_);
            this.render();
            this.toggleFlamegraph(reshow);

            ST.self.handleHover(0);

        } else {
            console.log('You reached the last level.');
        }
    };


    StackedLine.prototype.drillUp_ = function () {

        if (this.lastPlotsPlotted_.length > 0) {

            var lastArr = this.lastPlotsPlotted_.pop();
            var reshow = this.button_and_bellow.hasClass('opened');
            console.log('reshow: ' + reshow);

            this.setNameOfLinesToPlot_(lastArr);
            this.render();
            this.toggleFlamegraph(reshow);
        }
    };


    StackedLine.prototype.toggleFlamegraph = function (justReshow) {

        var height = '500px';
        var open;

        if (justReshow === true || justReshow === false) {
            open = justReshow;
        } else {
            //  toggle
            open = !this.button_and_bellow.hasClass('opened');
        }

        if (open) {
            console.log('was closed a');
            console.log('opening');
            this.below_section.show();
            this.button_and_bellow.addClass('opened');
            this.button_and_bellow.css('height', height);
        } else {

            console.log('was open a');
            height = '0px';
            this.below_section.hide();
            this.button_and_bellow.removeClass('opened');
            this.button_and_bellow.css('height', height);
        }

        var conEl = this.$conElement;
        var isf = this.$isFlamegraph;

        setTimeout(function () {
            var taClass = isf ? ".flamegraphTab" : ".metadataTab";
            console.log(taClass);
            var ta = conEl.find(taClass);
            console.dir(ta);
            ta.click();
        }, 0);
    };


    StackedLine.prototype.bindEvents = function () {

        var self = this;
        this.below_section = this.$conElement.find('.bellow-section');
        this.button_and_bellow = this.$conElement.find('.button_and_bellow');

        //  .toggle-bellow-section,
        this.$conElement.find('.toggle-open-close').off().on('click', $.proxy(self.toggleFlamegraph, self));

        this.$conElement.find('.drillUp').off().on('click', function () {
            self.drillUp_();
        });

        this.$conElement.find('.downloadChart').off().on('click', self.downloadChart);
    };


    StackedLine.prototype.updateDisplay = function () {
        this.$conElement.find('.count-display').text(this.count);
    };

    window.INIT = function (retryCount) {
            retryCount = retryCount || 0;
            var maxRetries = 20; // Max 1 second of retries (20 * 50ms)

            var element = currentCell;
            var el = $('.stacked-line-component[container_id="' + container_id + '"]');

            if (el.length === 0) {
                if (retryCount < maxRetries) {
                    // Container not in DOM yet, wait and retry
                    console.log('Container not found for container_id=' + container_id + ', retrying... (' + (retryCount + 1) + '/' + maxRetries + ')');
                    setTimeout(function() { window.INIT(retryCount + 1); }, 50);
                } else {
                    console.error('Failed to find container after ' + maxRetries + ' retries for container_id=' + container_id);
                }
                return;
            }

            console.log('Initializing StackedLine for container_id=' + container_id);
            el.each(function () {
                new StackedLine(this, window.PJ_Bus);
            });

        }

    // Initialize the component
    $(document).ready(function () {


        setTimeout(window.INIT, 0);
    });
})(jQuery);
