ST.TreeScapeModel = function () {

    var ef_, model_;

    var stubEntire_ = function( model ) {

        var ld_arr = model.entireForest.nodes.launchdate;

        //ld_arr.length
        ///  here we iterate over 'main', etc.
        for( var ldi=0; ldi < ld_arr.length; ldi++ ) {

            var ld_obj = ld_arr[ldi];
            var xaxis = ld_obj.xaxis;
            var first = ld_obj.xaxis[ xaxis.length - 1 ];
            var count = 0;

            for( var add_idx = xaxis.length; add_idx < 10000; add_idx++ ) {

                ld_obj.xaxis[ add_idx ] = first + 86000 * count;
                ld_obj.ydata[ add_idx ] = {
                    "sum": 800 + rand_(500),
                    "min": rand_(10),
                    "max": 80 + rand_(30),
                    "avg": rand_(20)
                };

                count++;
            }
        }

        console.log('Nodes: ');
        console.dir(model.entireForest.nodes);
    };


    var rand_ = function( r ) {
        return parseInt(Math.random()*r);
    };


    var setEntire_ = function (model) {

        //stubEntire_( model );

        ef_ = model.entireForest;
        model_ = model;

        console.log('Model:');
        console.dir(model_);
    };


    function interpolateArray(arr, factor, xaxis) {
        // Validate input
        if (!Array.isArray(arr) || arr.length < 2 || typeof factor !== 'number' || factor <= 0) {
            throw new Error('Invalid input');
        }

        const interpolatedArray = [];
        const numInterpolatedPoints = (arr.length - 1) * factor;

        var agg = ST.CurrentPlotModel.getXAxisAggregate();

        for (let i = 0; i < arr.length - 1; i++) {
            for (let j = 0; j < factor; j++) {

                const t = j / factor;

                var x = (1 - t) * arr[i][agg] + t * arr[i + 1][agg];

                if( xaxis ) {
                    x = (1 - t) * arr[i] + t * arr[i + 1];
                }

                interpolatedArray.push(x);
            }
        }

        // Add the last point of the original array
        interpolatedArray.push(arr[arr.length - 1][agg]);

        return interpolatedArray;
    }

    /*
        Each node has an xaxis array
        yaxis array
     */
    var interpolate_x_and_y_ = function (node) {

        var num_points = node.xaxis.length;
        var interpolation_factor = 300 / num_points;

        if (typeof node.xaxis[0] === "string") {
            return node;
        }

        var iay = interpolateArray(node.ydata, interpolation_factor);
        var new_arr_y = [];

        for( var j=0; j < iay.length; j++ ) {

            new_arr_y[j] = iay[j];
        }

        return {
            name: node.name,
            xaxis: interpolateArray(node.xaxis, interpolation_factor, true),
            ydata: iay
        }
    };


    var getPlotByName_ = function (node_name) {

        var xaxis_name = ST.CurrentPlotModel.getXAxis();
        var nodes = ef_.nodes[xaxis_name];

        if (!nodes) {
            console.log("Could not find xaxis: " + xaxis_name);
            return false;
        }

        for (var x in nodes) {

            if (nodes[x].name === node_name) {

                return {
                    xaxis_name: xaxis_name,
                    xaxis_type: getXAxisTypeInf_(xaxis_name),
                    nodes: interpolate_x_and_y_(nodes[x])
                };
            }
        }

        var msg = 'Could Not find plot named: ' + node_name + ' under xaxis_name: ' + xaxis_name;
        console.log(msg);

        return false;
    };


    var getPlotChildren_ = function (name) {
        return ef_.childrenMap[name];
    };

    var getPlotParent_ = function( name ) {
        return ef_.parentMap[name];
    };

    var getXAxisTypeInf_ = function (xaxis_name) {

        //  TODO: figure out what type the xAxis is dynamically.
        var xaxis_type = "default"
        if (xaxis_name == "launchdate") {
            xaxis_type = "date";
        }

        return xaxis_type;
    };

    //  useful for choosing an X Axis
    var getAttributes_ = function () {

        var atts = ["select attribute"];
        for (var attribute in ef_.nodes) {
            atts.push(attribute);
        }

        return atts;
    };

    var lastPlotsPlotted_ = [];
    var nameOfLinesToPlot_;

    var drillUp_ = function() {

        if( lastPlotsPlotted_.length > 0 ) {

            var lastArr = lastPlotsPlotted_.pop();


            ST.CurrentPlotModel.setNameOfLinesToPlot(lastArr);
            renderer_instance();
        }
    };


    var drillDown_ = function (chartClickedName) {

        console.log("clicked on = " + chartClickedName);

        var lastEntry = lastPlotsPlotted_[lastPlotsPlotted_.length - 1];

        if( nameOfLinesToPlot_ && nameOfLinesToPlot_.length > 0 && nameOfLinesToPlot_ !== lastEntry) {
            //  only cache if we're not hitting the last line.
            lastPlotsPlotted_.push( nameOfLinesToPlot_ );
            console.dir(lastPlotsPlotted_);
        }

        var nextPlot = getPlotChildren_(chartClickedName);

        if (nextPlot && nextPlot.length) {

            nameOfLinesToPlot_ = nextPlot;
            ST.CurrentPlotModel.setNameOfLinesToPlot(nameOfLinesToPlot_);
            renderer_instance();
        } else {
            console.log('You reached the last level.');
        }
    };


    var getNodeTreeNode_ = function( tree, name ) {

        for( var x=0; x < tree.length; x++ ) {
            if( name === tree[x].name ) {
                return tree[x];
            }
        }
    }


    var stackAddY_ = function( amended_y, nxy ) {

        var agg = ST.CurrentPlotModel.getXAxisAggregate();

        for( var i in nxy.ydata) {

            amended_y.ydata[i] = amended_y.ydata[i] || 0;
            amended_y.ydata[i] = amended_y.ydata[i] + nxy.ydata[i][agg];
        }

        amended_y.name = nxy.name;
        amended_y.xaxis = nxy.xaxis;
    };


    var getPlotNameClicked_ = function (xClick, yClick) {

        var currPlots = ST.CurrentPlotModel.getNameOfLinesToPlot();
        var currX = ST.CurrentPlotModel.getXAxis();

        var nodeTree = ef_.nodes[currX];
        var candidates = [];
        var amended_y = {
            name: "",
            ydata: []
        };

        for (var plotIdx in currPlots) {

            var plotName = currPlots[plotIdx];
            var nxy = getNodeTreeNode_( nodeTree, plotName );

            stackAddY_( amended_y, nxy );

            var candidate = isXYmatch_(amended_y, xClick, yClick);

            if (candidate) {
                candidates.push( candidate );
            }
        }

        candidates.sort(sortByLowestY);

        if( candidates.length === 0 ) {
            //  Mean user clicked outside of ALL plot boundaries.
            console.log('clicked outside all plot boundaries.');
            return false;
        }

        return candidates[0].name;
    };


    const sortByLowestY = (a, b) => {
      return a.y_at_this_xClick - b.y_at_this_xClick;
    };


    var isXYmatch_ = function (nxy, xClick, yClick) {

        var xaxis = nxy.xaxis;
        var yaxis = nxy.ydata;

        for( var i=0; i < xaxis.length; i++ ) {

            if( xClick >= xaxis[i] && xClick < xaxis[i+1] ) {

                var xdiff = xaxis[i+1] - xaxis[i];
                var percent_way_thru = (xClick-xaxis[i])/xdiff;

                var y0 = yaxis[i];
                var y1 = yaxis[i+1];
                var yrange = y1 - y0;
                var add_y = yrange * percent_way_thru;
                var y_at_this_xClick = y0 + add_y;

                //console.log('yat=' + y_at_this_xClick + '   yClick=' + yClick);

                if( yClick <= y_at_this_xClick ) {
                    return {
                        name: nxy.name,
                        y_at_this_xClick: y_at_this_xClick
                    }
                }
            }
        }
    };


    var getEntire_ = function() {
        return ef_;
    };


    return {
        drillUp: drillUp_,
        getPlotNameClicked: getPlotNameClicked_,
        drillDown: drillDown_,
        getAttributes: getAttributes_,
        getPlotByName: getPlotByName_,
        setEntire: setEntire_,
        getEntire: getEntire_
    }
}();