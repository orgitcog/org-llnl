ST.ModifyPJ_BusStubModel = function() {

    var add_ = function( PJ_bus ) {

        var LD_arr = PJ_bus.entireForest.nodes.launchdate;
        console.log("LD_arr: " );
        console.dir(LD_arr);

        for( var i=0; i < LD_arr.length; i++ ) {

            var time = 1569971895;

            for( var j=19; j < 500; j++ ) {

                LD_arr[i].xaxis[j] = time + 599000 * ( j - 19 );

                LD_arr[i].ydata[j] = {
                    "avg" : rand_(50),
                    "max" : rand_(9000),
                    "min" : rand_(10),
                    "sum" : rand_(200)
                };
            }
        }
    };

    var rand_ = function( r ) {
        return parseInt(Math.random()*r);
    };

    return {
        add: add_
    }
}();