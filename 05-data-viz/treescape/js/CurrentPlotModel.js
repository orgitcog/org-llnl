var ST = ST || {};
ST.CurrentPlotModel = function () {

    var nameOfLinesToPlot_,
        xAxis_,
        xAgg_ = "sum";

    var getNameOfLinesToPlot_ = function () {
        return nameOfLinesToPlot_;
    };

    var setNameOfLinesToPlot_ = function (names) {

        if (!Array.isArray(names)) {
            alert('setNameOflinesToPlot needs an array.');
        } else {
            nameOfLinesToPlot_ = names;
        }
    };

    var setXAxis_ = function (xaxis) {
        xAxis_ = xaxis;
    };

    var getXAxis_ = function () {
        return xAxis_;
    };

    var setXAxisAggregate_ = function( agg ) {
        xAgg_ = agg;
    };

    var getXAxisAggregate_ = function() {
        return xAgg_;
    };

    var getNewCanvasChartID_ = function() {

        for( var x=0; x < 1000; x++ ) {

            var found = $('#canvasChart' + x).length > 0;

            if( !found ) {
                return x;
            }
        }
    }

    return {
        getNewCanvasChartID: getNewCanvasChartID_,
        getXAxisAggregate: getXAxisAggregate_,
        setXAxisAggregate: setXAxisAggregate_,
        getNameOfLinesToPlot: getNameOfLinesToPlot_,
        setNameOfLinesToPlot: setNameOfLinesToPlot_,
        setXAxis: setXAxis_,
        getXAxis: getXAxis_
    }
}();