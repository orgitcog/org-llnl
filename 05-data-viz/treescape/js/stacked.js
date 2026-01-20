var ST = ST || {};

function convertEpochToReadableString(epochTimestamp) {
    epochTimestamp = +epochTimestamp;
    // Create a new Date object with the epoch timestamp
    var date = new Date(epochTimestamp);

    var months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', "Nov", "Dec"];
    // Use the Date object methods to get the different components of the date
    var year = date.getFullYear() - 2000;
    var md = date.getMonth(); // Months are zero-based
    var month = months[md];
    var day = ('0' + date.getDate()).slice(-2);
    var hours = ('0' + date.getHours()).slice(-2);
    var minutes = ('0' + date.getMinutes()).slice(-2);
    var seconds = ('0' + date.getSeconds()).slice(-2);

    // Create a readable string in the desired format
    var readableString = year + '-' + month + '-' + day + ' ' + hours + ':' + minutes;// + ':' + seconds;

    return readableString;
}

function updateContainerDisplay( renderFormat ) {

    if (renderFormat === "flamegraph") {
        $('.lineChartContainer').hide();
        $('.flameContainer').show();
    }

    if (renderFormat === "linegraph") {
        $('.lineChartContainer').show();
        $('.flameContainer').hide();
    }
}

//  Entry point from Python.
var render = function (chartjs) {

    //console.log('rerender 2');
    var PJ_Bus = PJ_BUS_REPLACE_JS;
    var renderFormat = "both";

    if( ST.MAKE_STUB ) {
        console.log('making stub');
        ST.ModifyPJ_BusStubModel.add(PJ_Bus);
    } else {
        console.log('not making stub.');
    }

    console.dir(PJ_Bus);
    window.PJ_Bus = PJ_Bus;

    renderer_instance( true );
    updateContainerDisplay( renderFormat );
};

var myChart = [];

var renderer_instance = function( brand_new_cell_render ) {

    //  if we're rerendering, for example after a select change then a
    //  lineChartContainer will already be present.
    //  if it's a brand new one, then it goes to element.append to make one.
    //  Each cell get's it's own 'element' object but they share global object space.
    var lineCharts = $(element).find('.lineChartContainer').length;
    var make_a_new_chart = lineCharts < 1;

    if( make_a_new_chart ) {

	    const output_area = document.querySelector('.jp-OutputArea-output');
    	const container = document.createElement('div');
    	container.className = 'stacked-line-component';
    	//output_area.appendChild(container);

	    
	    //      element.append('' +
            //'<script src="https://cdn.jsdelivr.net/npm/vue@3.2.20"></script>' +
            //'<script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>' +
            //'<div id="mainApp" class="mainApps">' +
            //'<button @click="increment">Increment</button>\n' +
            //'    <p>Count2: {{ count }}</p>' +
            //'</div>' +
//            '<div class="stacked-line-component"></div>'
            //'<script src="/files/js/StackedLine.js"></script>'
 //       );
    }

    ST.SelectView.bind();
};

//require(['https://cdn.jsdelivr.net/npm/chart.js'], render);
setTimeout( render, 0);
