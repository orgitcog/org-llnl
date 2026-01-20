// Scripting for the Assured Timing Detector Status Display
// Copyright (C) 2019, Lawrence Livermore National Security, LLC.
//    All rights reserved. LLNL-CODE-837067
//
// The Department of Homeland Security sponsored the production of this
//  material under DOE Contract Number DE-AC52-07NA27344 for the management and
//  operation of Lawrence Livermore National Laboratory.  Contract No.
//  DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
//  Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
//  See license for disclaimers, notice of U.S. Government Rights and license
//  terms and conditions.

  'use strict';

  // Streaming data source for detection metric plot:
  var source;
  // Plots and plot elements:
  var alert_annot_plot;
  var alert_annot_source;
  var det_metric_plot;
  var det_metric_line;
  var context_fig;
  var context_line;
  var context_alert_annot_plot;
  // Cached view collection from construction of plot layout:
  //   (used to manage inspection tools)
  var root_views;
  // WebSocket handle, URL, and message queue:
  var ws;
  var ws_url = "ws://localhost:8118";
  var ws_msg_queue = [];
  // Timer identifiers:
  var update_timer_id;
  var update_unack_timer_id;
  var alert_info_timer_id;

  // Last index added to the detection metric plot data stream:
  var last_idx;
  // Detection region boundaries and annotations:
  var det_region_start_times = [];
  var det_region_end_times = [];
  var det_region_annot = [];
  // Time at which the last successful calibration completed, if any:
  var last_cal_time;
  // Count of unacknowledged alerts and time of oldest such alert:
  var num_unack_alerts = 0;
  var unack_start_time;

  // Default and minimum permitted extents of the x-axis for the metric plot:
  const default_x_extent = 1000 * 60 * 30; // 30 minutes
  const min_x_extent = 1000 * 60; // 1 minute

  // Timestamp at which the initial extent of the detection metric plot begins
  //   (simply a placeholder allowing plot setup prior to receipt of data)
  const start_timestamp = new Date("1970-12-03T12:21:03");

// Stop the heartbeat timer, style the heartbeat indicator as disconnected
//  and show the calibration status as unknown
function stop_heartbeat()
{
    clearInterval(update_timer_id);
    const heartbeat_elt = document.getElementById("proc_heartbeat");
    heartbeat_elt.style.backgroundColor = "darkred";
    set_annunciator_status(''); // show annunciator as inactive
    const connect_btn = document.getElementById("connect_toggle_btn");
    connect_btn.classList.remove("connected");
    connect_btn.innerText = "Connect";
    const cal = document.getElementById("cal");
    cal.classList.add("unknown");
    cal.classList.remove("valid");
    cal.classList.remove("invalid");
    cal.classList.remove("in-progress");
    cal.title = "";
    cal_elapsed.innerText = "";
}

// Toggle the WebSocket connection status
function connect_ws_toggle()
{
    if ((ws === undefined) || (ws.readyState === WebSocket.CLOSED))
    {
        connect_ws();
    }
    else
    {
        disconnect_ws();
    }
}

// Connect to a WebSocket server to received data updates to plot
function connect_ws()
{
    console.log("Connecting to WebSocket at " + ws_url);
    ws = new WebSocket(ws_url);
    ws.addEventListener("close", handle_ws_close);
    ws.addEventListener("message", handle_ws_message);
    ws.addEventListener("error", handle_ws_error);
    update_timer_id = setInterval(process_ws_queue, 200);
    set_annunciator_status("green");
    const connect_btn = document.getElementById("connect_toggle_btn");
    connect_btn.classList.add("connected");
    connect_btn.innerText = "Disconnect";
}

// Style the annunicator to reflect a new alert status (red/yellow/green)
function set_annunciator_status(new_status)
{
    const annun_red = document.getElementById("annunciator-red");
    const annun_yellow = document.getElementById("annunciator-yellow");
    const annun_green = document.getElementById("annunciator-green");
    const new_status_lc = new_status.toLowerCase();

    annun_red.classList.toggle("inactive", new_status_lc !== "red");
    annun_yellow.classList.toggle("inactive", new_status_lc !== "yellow");
    annun_green.classList.toggle("inactive", new_status_lc !== "green");
}

// Disconnect from the WebSocket server and stop the heartbeat timer
function disconnect_ws()
{
    ws.close();
    stop_heartbeat();
}

// Handle a normal closure of the WebSocket connection
function handle_ws_close(evt)
{
    console.log("WebSocket closed: ", evt);
}

// Handle an error raised by the WebSocket connection and stop heartbeat timer
function handle_ws_error(evt)
{
    console.error("Got WebSocket error: ", evt);
    stop_heartbeat();
    alert("Got WebSocket error: " + evt.type);
}

// Package a new plot datum as a Bokeh stream item
function make_stream_item(x_idx, timestamp, y_val, y_thresh_val)
{
    const new_date = new Date(timestamp);
    const timestamp_ms_vs_epoch = new_date.valueOf();
    return {
        index: x_idx,
        timestamp: timestamp,
        x: timestamp_ms_vs_epoch,
        y: y_val,
        y_thresh: y_thresh_val };
}

// Add a new alert to the alert history table (including attributes for use by
//  the callout box and required event handlers to display it)
function add_alert_history_table_item(msg_obj)
{
    const alert_level = msg_obj.level.toLowerCase();
    const history_tbl = document.querySelector(
        "table#alert_history_tbl > tbody");
    let new_history_tbl_row = history_tbl.insertRow(0);
    const alert_idx = history_tbl.rows.length;
    let new_idx_cell = new_history_tbl_row.insertCell(0);
    new_idx_cell.innerText = alert_idx;
    new_idx_cell.classList.add("integer");
    new_history_tbl_row.insertCell(1).innerText = msg_obj.timestamp;
    let new_level_cell = new_history_tbl_row.insertCell(2);
    new_level_cell.innerText = alert_level;
    new_level_cell.classList.add("category");
    new_level_cell.classList.add(alert_level + "-bullet");
    new_history_tbl_row.insertCell(3).innerText = msg_obj.reason_desc;

    // Set (non-displayed) attributes for use in callout box
    new_history_tbl_row.setAttribute("data-reason-code",
        msg_obj.reason_code);
    new_history_tbl_row.setAttribute("data-reason-extra",
        msg_obj.reason_extra);
    new_history_tbl_row.setAttribute("data-rx-timestamp",
        msg_obj.rx_timestamp.toISOString());
    new_history_tbl_row.setAttribute("data-acked", false);

    new_history_tbl_row.addEventListener("click",
        handle_history_row_click);
    new_history_tbl_row.addEventListener("mouseenter",
        handle_history_row_hover);
    new_history_tbl_row.addEventListener("mouseleave",
        handle_history_row_leave);
}

// Add an annotation for the specified alert, including a triangle along the
//  x-axis for all alerts and an arrow from start to end of the alert interval
//  (if the previous alert level was above green)
function add_alert_annot(alert_level, alert_timestamp)
{
    const alert_above_green = alert_level !== "green";
    const new_x = new Date(alert_timestamp).valueOf();
    let new_color;
    switch (alert_level)
    {
        case "red":
            new_color = "#8b0000"; // AKA darkred
            break;
        case "yellow":
            new_color = "#f3f307"; // AKA var(--dim-yellow)
            break;
        case "green":
            new_color = "#569e56";
            break;
        default:
            new_color = "gray";
            break;
    }
    alert_annot_source.stream({ x: new_x, y: 0, level: alert_level,
        color: new_color });
    if (alert_annot_source.data.x.length > 1)
    {
        const alert_start_idx = alert_annot_source.data.x.length - 2;
        const alert_start_x =
            alert_annot_source.data.x[alert_start_idx];
        const last_alert_level = alert_annot_source.data.level[alert_start_idx];
        if (last_alert_level === "green")
        {
            // Annotate red->yellow and [anything]->green but not
            //  green->[anything] alerts
            return;
        }
        const arrow_color = alert_annot_source.data.color[alert_start_idx];
        // TODO: There is likely an available v_compute transformation
        //  for the figure y_scale, but it not readily accessible
        const y_px_to_data_space = function (y_px) {
            const y_data_range_len = det_metric_plot.y_range.end -
                det_metric_plot.y_range.start;
            return y_px * y_data_range_len / det_metric_plot.height; };
        const arrow_y = y_px_to_data_space(12);
        const arrow_annot = new Bokeh.Arrow({
            end: new Bokeh.VeeHead({ size: 7, line_color: arrow_color }),
            line_color: arrow_color,
            line_width: 2,
            x_start: alert_start_x,
            y_start: arrow_y,
            x_end: new_x,
            y_end: arrow_y
            });
        console.log("Added alert arrow annotation: ", alert_start_x,
            "->", new_x);
        add_layout_and_refresh(det_metric_plot, arrow_annot);
    }
}

// Workaround Bokeh bug #8862 by emitting a change notification after a call
//  to add_layout
function add_layout_and_refresh(plot_obj, item_to_add)
{
    plot_obj.add_layout(item_to_add);
    plot_obj.properties.renderers.change.emit(); // FIXME Workaround
}

// Show a dialog box requesting operator acknowledgement of all outstanding
//  alert notifications, launching (if necessary) a timer to track the length
//  of time the alerts remained unacknowledged
function require_ack(alert_level)
{
    const alert_above_green = alert_level !== "green";
    const cb_req_ack = document.querySelector(
        "input[type='radio'][name='req_ack']:checked");
    let show_ack_box = false;
    switch (cb_req_ack.value.toLowerCase())
    {
        case "none":
            show_ack_box = false;
            break;
        case "red-only":
            show_ack_box = (alert_level === "red");
            break;
        case "all":
            show_ack_box = alert_above_green;
            break;
        default:
            console.error("Unknown required ACK type:", cb_req_ack.value);
            break;
    }
    const alert_ack_btn = document.getElementById("alert_ack_btn");
    if (alert_above_green)
    {
        ++num_unack_alerts;
        alert_ack_btn.innerText = "Acknowledge " + num_unack_alerts +
            ((num_unack_alerts == 1) ? " Alert" : " Alerts");
    }
    if (show_ack_box)
    {
        const alert_box = document.getElementById("alert_ack_box");
        const was_hidden = alert_box.classList.contains("hidden");
        if (was_hidden)
        {
            unack_start_time = Date.now();
            update_unack_timer_id = setInterval(update_unack_time, 1000);
            update_unack_time(); // immediately so valid when box is shown
        }
        alert_box.classList.remove("hidden");
        alert_ack_btn.focus();
    }
}

// Sound an audible alert if the level is above green and the user has opted
//  to enable such alerts
function maybe_sound_alert(alert_level)
{
    const alert_above_green = alert_level !== "green";
    const cb_audible = document.getElementById("cb_audible");
    if (cb_audible.checked && alert_above_green)
    {
        sound_alert();
    }
}

// Process a received alert message (updating annunciator, sounding alert,
//  adding a history table row, requesting ack and annotating as appropriate)
function process_alert_msg(msg_obj)
{
    const alert_level = msg_obj.level.toLowerCase();
    msg_obj.rx_timestamp = new Date();

    // Update the annunciator ("stop light" indicator)
    set_annunciator_status(alert_level);

    // Sound alert, if indicated (level > green and user-requested)
    maybe_sound_alert(alert_level);

    // Add to history table for alerts
    add_alert_history_table_item(msg_obj);

    // Add to annotation
    add_alert_annot(alert_level, msg_obj.timestamp);

    // Require acknowledgement, if requested
    require_ack(alert_level);
}

// Process a received status message (here, handling only the calibration status
//  updates)
function process_status_msg(msg_obj)
{
    if (msg_obj.component.toLowerCase() !== "cal")
        return;
    const cal = document.getElementById("cal");
    let is_start = false;
    let is_success = false;
    let is_constrained_success = false;
    let is_invalid = false;
    switch (msg_obj.event.toLowerCase())
    {
        case "start":
            is_start = true;
            cal.title = "Calibration started at: " + msg_obj.timestamp;
            break;
        case "success":
            is_success = true;
            last_cal_time = {
                local: Date.now(), remote: new Date(msg_obj.timestamp) };
            cal.title = "Last calibration completed: " + msg_obj.timestamp;
            break;
        case "success (constrained)":
            is_constrained_success = true;
            last_cal_time = {
                local: Date.now(), remote: new Date(msg_obj.timestamp) };
            cal.title = "Last calibration completed: " + msg_obj.timestamp +
                "\n(* solution constrained)";
            break;
        case "failure":
            cal.title = "Last calibration failed at: " + msg_obj.timestamp;
            is_invalid = true;
            break;
        default:
            console.error("Unknown cal status:", msg_obj.event);
            is_invalid = true;
            break;
    }
    cal.classList.remove("unknown");
    cal.classList.toggle("invalid", is_invalid);
    cal.classList.toggle("valid", is_success);
    cal.classList.toggle("valid-constrained", is_constrained_success);
    cal.classList.toggle("in-progress", is_start);
}

// Format a duration in milliseconds into a colon-separated string with hour,
//  minute and (if requested) second components
function breakdown_duration(duration_ms, options = { str_with_s: true,
  omit_zeroes: true })
{
    const ms_per_s = 1000;
    const ms_per_min = 60 * ms_per_s;
    const ms_per_h = 60 * ms_per_min;
    const fmt_2dig = function (x) {
        return Math.floor(x).toString().padStart(2, "0"); };
    const duration_h = fmt_2dig(duration_ms / ms_per_h);
    const duration_min = fmt_2dig((duration_ms % ms_per_h) / ms_per_min);
    const duration_s = fmt_2dig((duration_ms % ms_per_min) / ms_per_s);
    let ret = {
        hours: duration_h, minutes: duration_min, seconds: duration_s };
    ret.str = (options.omit_zeroes && (duration_h == 0)) ? "" :
        (duration_h + "h:");
    ret.str += duration_min + "m";
    if (options.str_with_s && (duration_h.length == 2))
    {
        // Drop :##s if >99 hours (for conciseness)
        ret.str += ":" + duration_s + "s";
    }
    return ret;
}

// Update the display of the duration since the last successful calibration
function update_cal_elapsed(remote_timestamp)
{
    const cal_elapsed = document.getElementById("cal_elapsed");
    let new_elapsed_str = "";
    if (last_cal_time !== undefined)
    {
        const curr_local = Date.now();
        const local_elapsed_ms = curr_local - last_cal_time.local;
        const remote_elapsed_ms = remote_timestamp - last_cal_time.remote;
        const elapsed_ms = remote_elapsed_ms;
        // TODO: Warn if local and remote computations differ greatly
        const ms_per_h = 3600 * 1000;
        const ms_per_min = 60 * 1000;
        const fmt_2dig = function (x) {
            return Math.floor(x).toString().padStart(2, "0"); };
        const elapsed_h = fmt_2dig(elapsed_ms / ms_per_h);
        const elapsed_min = fmt_2dig((elapsed_ms % ms_per_h) / ms_per_min);
        const elapsed_dur = breakdown_duration(elapsed_ms,
            { str_with_s: false, omit_zeroes: false });
        new_elapsed_str = elapsed_dur.str + " since last cal";
    }
    cal_elapsed.innerText = new_elapsed_str;
}

// Enqueue (and potentially, rapidly process) an incoming WebSocket message
// N.B. - Keep this function fast, or messages can lose sequencing
function handle_ws_message(evt)
{
    const msg_obj = JSON.parse(evt.data);
    switch (msg_obj.msg_type.toLowerCase())
    {
        case "alert":
          process_alert_msg(msg_obj);
          break;
        case "status":
          process_status_msg(msg_obj);
          break;
    }
    ws_msg_queue.push(msg_obj);
}

// Produce a Bokeh stream item for the detection metric plot based on a
//  WebSocket message reporting such information
function ws_msg_to_stream_item(ws_msg)
{
    if (ws_msg.msg_type !== "metric")
        return null;
    if (last_idx === undefined)
    {
        if (source.data.index.length > 0)
        {
            last_idx = Math.max(...source.data.index.filter(function (v) {
                return !isNaN(v); })) + 1;
        }
        else
        {
            last_idx = 0;
        }
    }
    const next_idx = last_idx + 1;
    last_idx = next_idx;
    const stream_item = make_stream_item(next_idx, ws_msg.timestamp,
        ws_msg.det_metric, ws_msg.threshold);
    return stream_item;
}

// Add an annotation to the beginning & end of detection regions
//  (where the threshold is exceeded)
function annotate_det_regions()
{
    if (source.data.y.length < 2)
    {
        return;
    }
    let start_idx = 1;
    if (det_region_start_times.length > 0)
    {
        const search_time_start = Math.max(...det_region_start_times,
            ...det_region_end_times);
        start_idx = source.data.x.findIndex(
            function (t) { return t > search_time_start; });
    }
    for (let i = start_idx; i < source.data.y.length; ++i)
    {
        const was_above_thresh =
            source.data.y[i - 1] > source.data.y_thresh[i - 1];
        const is_above_thresh =
            source.data.y[i] > source.data.y_thresh[i];
        if (is_above_thresh && !was_above_thresh)
        {
            // Add annotation
            const new_region_start = source.data.x[i];
            det_region_start_times.push(source.data.x[i]);
            let new_annot = new Bokeh.BoxAnnotation({
                left: new_region_start,
                fill_color: "red",
                fill_alpha: 0.2 });
            let new_context_annot = new Bokeh.BoxAnnotation({
                left: new_region_start,
                fill_color: "red",
                fill_alpha: 0.6 });
            det_region_annot.push({
                main: new_annot, context: new_context_annot});
            add_layout_and_refresh(det_metric_plot, new_annot);
            add_layout_and_refresh(context_fig, new_context_annot);
            console.log("Annotation detection region started: [" +
                new_region_start + ", ]");
        }
        else if (was_above_thresh && !is_above_thresh)
        {
            const new_region_end = source.data.x[i];
            let corresponding_left =
                det_region_start_times[det_region_start_times.length - 1];
            if (det_region_annot.length == 0)
            {
                continue;
            }
            let last_annot_pair = det_region_annot[det_region_annot.length - 1];
            let last_annot = last_annot_pair.main;
            let last_context_annot = last_annot_pair.context;
            det_region_end_times.push(new_region_end);
            last_annot.right = new_region_end;
            last_context_annot.right = new_region_end;
            console.log("Annotation detection region [" +
                corresponding_left + ", " + new_region_end + "]");
        }
    }
}

// Process a batch of items to stream to the detection metric plot by converting
//  to structure-of-arrays, streaming to the plot, and annotating detection
//  regions
function process_stream_item_batch(stream_items)
{
    if (stream_items.length == 0)
    {
        return;
    }
    let stream_item_array = {};
    for (let x in stream_items[0])
    {
        stream_item_array[x] = stream_items.map(function (y) {
            return y[x]; });
    }
    stream_item_to_plot(stream_item_array);
    annotate_det_regions();
}

// Apply styling to "pulse" the heartbeat indicator
function proc_heartbeat_pulse()
{
    const elt = document.getElementById("proc_heartbeat");
    const color_off = "darkgreen";
    const color_on = "green";

    if (elt.style.backgroundColor !== color_on)
    {
        elt.style.backgroundColor = color_on;
    }
    else
    {
        elt.style.backgroundColor = color_off;
    }
}

// Process any enqueued WebSocket messages
function process_ws_queue()
{
    const max_items = 2000;
    if (ws_msg_queue.length > 0)
    {
        console.log("proc_wsq start");
        const start_time = performance.now();
        const ws_msgs_to_proc = ws_msg_queue.splice(0, max_items);
        const raw_stream_items = ws_msgs_to_proc.map(ws_msg_to_stream_item);
        const stream_items = raw_stream_items.filter(
            function (v) { return v !== null; });
        process_stream_item_batch(stream_items);
        const ms_per_msg = (performance.now() - start_time) /
            ws_msgs_to_proc.length;
        console.log("proc_wsq done; processed " + ws_msgs_to_proc.length
            + " at " + ms_per_msg + "ms/msg");

        if (stream_items.length > 0)
        {
            const last_proc_timestamp_ms_vs_epoch =
                stream_items[stream_items.length - 1].x;
            update_cal_elapsed(new Date(last_proc_timestamp_ms_vs_epoch));
        }

        proc_heartbeat_pulse();
    }
    else if (ws.readyState == WebSocket.CLOSED)
    {
        // Nothing left to process, and socket is closed, so no new data
        //  expected; stop heartbeat and processing interval timer
        stop_heartbeat();
        return;
    }
}

// Update the duration for which alerts have been unacknowledged within the
//  alert ack dialog box
function update_unack_time()
{
    if (unack_start_time === undefined)
        return;
    const unack_elapsed = document.querySelector("#unack_elapsed");
    const unack_elapsed_ms = Date.now() - unack_start_time;
    const unack_elapsed_dur = breakdown_duration(unack_elapsed_ms);
    unack_elapsed.innerText =
        ((num_unack_alerts == 1) ? "Alert " : "Alerts ") +
        "unacknowledged for " + unack_elapsed_dur.str;
}

// Called by the document on_load; adds the plot elements to the display and
//  connects event listeners
function load_plot()
{
    const timestamps_ms_vs_epoch = [new Date(start_timestamp).valueOf()];
    source = new Bokeh.ColumnDataSource({ data: {
        index: [],
        timestamp: [],
        x: [],
        y: [],
        y_thresh: [] } });
    alert_annot_source = new Bokeh.ColumnDataSource({ data: {
        x: [],
        y: [],
        level: [],
        color: [] } });
    const default_x_range = new Bokeh.Range1d({
        start: timestamps_ms_vs_epoch[0],
        end: timestamps_ms_vs_epoch[0] + default_x_extent,
        min_interval: min_x_extent
        });

    det_metric_plot = Bokeh.Plotting.figure({
        title: "Timing Anomaly Detection Results",
        tools: "xpan, xwheel_zoom, box_zoom, xzoom_out, xzoom_in, \
            reset, save",
        // output_backend: "webgl", // maybe use after issue #8346 fixed
        plot_width: 700,
        plot_height: 400,
        x_axis_label: "Time (UTC)",
        x_axis_type: "datetime",
        x_range: default_x_range,
        y_axis_label: "Detection Metric",
        y_range: new Bokeh.Range1d({ start: 0, end: 10000 }),
        toolbar_location: "above"
        });
    det_metric_plot.toolbar["logo"] = null; // inline CSS actually does this
    const boxzoom = det_metric_plot.toolbar.tools.find(function (t) {
        return t.type === "BoxZoomTool" });
    boxzoom.dimensions = "width"; // x-zoom only
    det_metric_plot.xaxis.axis_label_text_font_size = "11pt";
    det_metric_plot.yaxis.axis_label_text_font_size = "11pt";
    det_metric_plot.xaxis.major_label_text_font_size = "11pt";
    det_metric_plot.yaxis.major_label_text_font_size = "10pt";
    det_metric_line = det_metric_plot.line({ field: "x" }, { field: "y" }, {
        source: source,
        legend: "Detection Metric"
        } );
    let det_metric_line_thresh = det_metric_plot.line(
        { field: "x" }, { field: "y_thresh" }, {
        source: source,
        color: "firebrick",
        legend: "Threshold",
        line_dash: [1, 3],
        line_width: 2 } );

    // Annotate alerts along the x-axis
    alert_annot_plot = det_metric_plot.diamond({ field: "x" }, { field: "y" }, {
        source: alert_annot_source,
        fill_color: { field: "color" },
        line_color: "#808080",
        size: 20
        } );

    const hover_tool = new Bokeh.HoverTool({
        tooltips: [
            ["Timestamp", "@timestamp"],
            ["Det. Metric", "@y{0,0.0}"],
            ["Threshold", "@y_thresh{0,0.0}"],
            ["Index", "$index"]],
        renderers: [det_metric_line],
        //active: false,
        mode: "vline",
        toggleable: true
        });
    det_metric_plot.add_tools(hover_tool);

    det_metric_plot.legend.location = "top_left";

    context_fig = Bokeh.Plotting.figure({ 
        title: "Detection Result History",
        tools: "",
        toolbar_location: null,
        // output_backend: "webgl", // maybe use after issue #8346 fixed
        plot_width: 700,
        plot_height: 150,
        // x_axis_label: "Time (UTC)",
        x_axis_type: "datetime",
        y_minor_ticks: 2,
        y_range: det_metric_plot.y_range
        });
    context_fig.xaxis.major_label_text_font_size = "11pt";
    context_line = context_fig.line( { field: "x" }, { field: "y" }, {
        source: source
        } );
    // Annotate alerts along the x-axis
    context_alert_annot_plot = context_fig.diamond(
        { field: "x" }, { field: "y" }, {
        source: alert_annot_source,
        fill_color: { field: "color" },
        line_color: "#808080",
        size: 15
        } );
    context_fig.ygrid.grid_line_color = null;

    let plot_host = document.getElementById("plot_host");
    const gridplot = Bokeh.Plotting.gridplot([[det_metric_plot], [context_fig]],
        {
        sizing_mode: "fixed",
        toolbar_location: "above",
        plot_width: 700
        });
    const root_views_promise = Bokeh.Plotting.show(gridplot, plot_host);

    // TODO Remove this workaround when HoverTool icon status reflects initial
    //  setting of active to false
    hover_tool.active = false;
    root_views_promise.then(function (v) { root_views = v; }).then(
        inactivate_insp_tools);

    add_event_listeners();
}

function add_event_listeners()
{
    document.getElementById("cb_audible").addEventListener("change",
        handle_audible_change);

    const options_collapse_btn = document.querySelector(
        "div#misc_controls > .collapse-btn");
    options_collapse_btn.addEventListener("click", function (evt) {
        this.parentElement.classList.toggle("collapsed"); });
    const connect_toggle_btn = document.getElementById(
        "connect_toggle_btn");
    connect_toggle_btn.addEventListener("click", connect_ws_toggle);
    const alert_ack_btn = document.getElementById("alert_ack_btn");
    alert_ack_btn.addEventListener("click", handle_alert_ack);
    const error_box_close_btn = document.querySelector(
        "div#error_box #error_box_close_btn");
    error_box_close_btn.addEventListener("click", handle_error_box_close);
}

function add_context_range_tool()
{
    const range_tool = new Bokeh.RangeTool({
        x_range: det_metric_plot.x_range,
        y_interaction: false,
        overlay: new Bokeh.BoxAnnotation({
            fill_color: "navy",
            fill_alpha: 0.2,
            left: det_metric_plot.x_range.start,
            right: det_metric_plot.x_range.end
            })});
    context_fig.add_tools(range_tool);
}

// Inactivate all inspection tools
//  (this is necessary since Bokeh currently doesn't seem to have the active
//   status of the hover tool synchronized properly with its button display)
function inactivate_insp_tools()
{
    // First, before inactivating the inspection tools, fire a simulated
    //  mouse leave event, which will remove existing renderings (e.g., tooltip)
    const ui_bus = root_views.child_views[1].child_views[0].ui_event_bus;
    ui_bus.hit_area.dispatchEvent(new MouseEvent('mouseleave'));

    const insp_tools = root_views.child_models[0].toolbar.inspectors;
    const orig_active_status = insp_tools.map(function (t) {
        return t.active; }).slice();
    insp_tools.forEach(function (t) { t.active = false; });
    return orig_active_status;
}

// Similar to js_on_event, exposed by Bokeh but not BokehJS API
function connect_js_event(bokeh_obj, evt, callback)
{
    if (bokeh_obj.js_event_callbacks[evt] === undefined)
        bokeh_obj.js_event_callbacks[evt] = [];
    bokeh_obj.js_event_callbacks[evt].push(callback);
    console.log("Added callback for", evt, "; # of callbacks is now",
        bokeh_obj.js_event_callbacks[evt].length);
    bokeh_obj.properties.js_event_callbacks.change.emit();
}

// Similar to js_on_change, exposed by Bokeh but not BokehJS API
function connect_js_property_callback(bokeh_obj, prop, callback)
{
    if (bokeh_obj.js_property_callbacks[prop] === undefined)
        bokeh_obj.js_property_callbacks[prop] = [];
    bokeh_obj.js_property_callbacks[prop].push(callback);
    console.log("Added callback for", prop, "; # of callbacks is now",
        bokeh_obj.js_property_callbacks[prop].length);
    bokeh_obj.properties.js_property_callbacks.change.emit();
}

// Scroll the detection metric plot to a new position, dictated by either a
//  new_start or new_end (either of which can be a field of arg. new_pos)
function scroll_plot_to(new_pos)
{
    let delta_x;
    if ("new_start" in new_pos)
    {
        delta_x = new_pos.new_start - det_metric_plot.x_range.start;
    }
    else if ("new_end" in new_pos)
    {
        delta_x = new_pos.new_end - det_metric_plot.x_range.end;
    }
    det_metric_plot.x_range.start += delta_x;
    det_metric_plot.x_range.end += delta_x;
    det_metric_plot.x_range.change.emit();
}

// Attempt to fix duplicate timestamps in the data stream by slightly shifting
//  entries if needed
function attempt_repeat_fixup(item)
{
    let ret_item_x = [];
    const max_ms_mod = 5000;
    const max_source_x = Math.max(...source.data.x);
    for (let i = 0; i < item.x.length; ++i)
    {
        for (let delta_ms = 0; delta_ms <= max_ms_mod; ++delta_ms)
        {
            const new_x_elt = item.x[i] + delta_ms;
            if ((new_x_elt > max_source_x) &&
                !ret_item_x.includes(new_x_elt) &&
                !source.data.x.includes(new_x_elt))
            {
                ret_item_x.push(new_x_elt);
                break;
            }
        }
    }
    if (ret_item_x.length !== item.x.length)
        return; // failure to fixup all entries
    let ret_item = item;
    ret_item.x = ret_item_x;
    return ret_item;
}

// Stream a new batch of detection metric data to the plot, scrolling the plot
//  as needed
function stream_item_to_plot(item)
{
    // See Bokeh issues #6470 and #7072 re: slowdown in streaming as data
    //  grows in size
    const was_empty = (source.data.x.length == 0);
    const was_at_end = was_empty ||
        (det_metric_plot.x_range.end >=
            source.data.x[source.data.x.length - 1]);
    const no_repeats = item.x.every(function (v) {
        return !source.data.x.includes(v); });
    if (!no_repeats)
    {
        let item_fixup = attempt_repeat_fixup(item);
        if (item_fixup)
        {
            item = item_fixup;
            console.warn("Needed to fix duplicate timestamps");
        }
        else
        {
            show_error_box("ERROR: Received data with a duplicate timestamp; \
                errors may occur in the display");
        }
    }
    source.stream(item);
    if (was_empty)
    {
        // Add the range tool now that there's a valid x-range
        add_context_range_tool();

        det_metric_plot.x_range.reset_start = source.data.x[0];
        det_metric_plot.x_range.reset_end = source.data.x[0] + default_x_extent;
    }
    if (was_at_end)
    {
        const new_last_x = item.x[item.x.length - 1];
        scroll_plot_to({ new_end: new_last_x });
    }
}

// Handle a click on an alert history table row by scrolling the detection
//  metric plot to (slightly before) the start of the selected event
function handle_history_row_click(evt)
{
    const timestamp = this.cells[1].innerText;
    const new_x = new Date(timestamp).valueOf();
    const pre_margin = 60000;
    scroll_plot_to({ new_start: new_x - pre_margin });
}

// Handle a hover event over a row of the alert history table by showing
//  a callout with additional event information
function handle_history_row_hover(evt)
{
    if (evt.relatedTarget && (evt.relatedTarget.id === "callout_triangle"))
    {
        return; // ignore this event; did not actually (re-)enter row
    }
    const alert_info_box = document.getElementById("alert_info_box");
    // Need to temporarily make the info box invisible prior to removing
    //  display:none so that dimensions can be computed; finally, visibility
    //  is restored once the box is repositioned
    alert_info_box.style.visibility = "hidden";
    alert_info_box.classList.remove("hidden");

    // Change text before querying dimensions (which may change accordingly)
    const alert_info_reason = document.querySelector("#alert_info_reason");
    const alert_info_reason_extra = document.querySelector(
        "#alert_info_reason_extra");
    const alert_info_type = document.querySelector("#alert_info_type");
    const alert_info_timestamp = document.querySelector(
        "#alert_info_timestamp");
    const alert_info_rx_timestamp = document.querySelector(
        "#alert_info_rx_timestamp");
    const alert_info_acked = document.querySelector("#alert_info_acked");
    const hover_tr = evt.currentTarget;
    const rx_timestamp = hover_tr.getAttribute("data-rx-timestamp");
    const since_rx_timestamp_ms = Date.now() - new Date(rx_timestamp);
    const since_rx_timestamp = breakdown_duration(since_rx_timestamp_ms);
    let acked = hover_tr.getAttribute("data-acked");
    if (acked == "false")
    {
        acked = "<not acknowledged>";
    }
    else
    {
        const since_ack_ms = Date.now() - new Date(acked);
        const since_ack = breakdown_duration(since_ack_ms);
        acked += " (" + since_ack.str + " ago)";
    }
    const alert_code = hover_tr.getAttribute("data-reason-code");
    alert_info_type.innerText = hover_tr.cells[2].innerText + " Alert " +
        "(Alert #" + hover_tr.cells[0].innerText + ")";
    alert_info_type.classList = hover_tr.cells[2].classList;
    alert_info_timestamp.innerText = hover_tr.cells[1].innerText;
    alert_info_rx_timestamp.innerText = rx_timestamp + " (" +
        since_rx_timestamp.str + " ago)";
    alert_info_acked.innerText = acked;
    alert_info_reason.innerText = hover_tr.cells[3].innerText + " (code #" +
        alert_code + ")";
    let reason_extra = hover_tr.getAttribute("data-reason-extra");
    if (!reason_extra || (reason_extra.length == 0))
    {
        reason_extra = "<not specified>";
    }
    alert_info_reason_extra.innerText = reason_extra;

    // Compute the new callout top position
    const alert_info_height = alert_info_box.clientHeight;
    const alert_info_parent_cstyle = window.getComputedStyle(
        alert_info_box.parentElement);
    const parent_margin_top = parseFloat(alert_info_parent_cstyle["marginTop"]);
    const callout_height = 5;
    const new_box_top = evt.pageY - evt.offsetY - alert_info_height -
        parent_margin_top - callout_height;

    // Make the info box callout visible after a brief delay, so it only
    //  appears after a pointer hover (there is no harm if the pointer
    //  subsequently leaves, since the hidden class is restored in that
    //  case; visibility just returns without effect to the default)
    const move_and_make_visible = function () {
        alert_info_box.style.top = new_box_top + "px";
        alert_info_box.style.visibility = "visible"; };
    alert_info_timer_id = window.setTimeout(move_and_make_visible, 500);
}

// Hide the callout box when the cursor leaves an alert history row
function handle_history_row_leave(evt)
{
    if (evt.relatedTarget && (evt.relatedTarget.id === "callout_triangle"))
    {
        return; // ignore this event; did not actually leave row
    }
    const alert_info_box = document.getElementById("alert_info_box");
    alert_info_box.classList.add("hidden");
    window.clearTimeout(alert_info_timer_id);
}

// Handle operator acknowledgement of outstanding alerts (by marking the alert
//  with the time at which they were acknowledged and hiding the alert
//  acknowledgement dialog box)
function handle_alert_ack(evt)
{
    const alert_box = document.getElementById("alert_ack_box");
    alert_box.classList.add("hidden");
    clearInterval(update_unack_timer_id);
    num_unack_alerts = 0;
    unack_start_time = undefined;

    const history_tbl = document.querySelector(
        "table#alert_history_tbl > tbody");
    const ack_time_str = new Date().toISOString();
    for (let row of history_tbl.rows)
    {
        if (row.getAttribute("data-acked") == "false")
        {
            row.setAttribute("data-acked", ack_time_str);
        }
    }
}

// Show an error message box (e.g., if duplicate timestamps are detected)
function show_error_box(err_msg)
{
    const error_box = document.querySelector("div#error_box");
    const error_text = document.querySelector("div#error_box #error_text");
    error_text.innerText = err_msg;
    console.error(err_msg);
    error_box.classList.remove("hidden");
}

function handle_error_box_close(evt)
{
    const error_box = document.querySelector("div#error_box");
    error_box.classList.add("hidden");
}

// Sound an audible sawtooth-wave alert tone
function sound_alert()
{
    const audio_ctx = new AudioContext();
    const osc_node = new OscillatorNode(audio_ctx, { type: "sawtooth",
        frequency: 800});

    const gain_node = audio_ctx.createGain();
    const vol_elt = document.getElementById("vol");
    console.log("Vol: ", vol_elt.value);
    gain_node.gain.value = vol_elt.valueAsNumber;
    osc_node.connect(gain_node);
    gain_node.connect(audio_ctx.destination);
    const alert_len_s = 0.5; // length of tone in seconds
    osc_node.start();
    osc_node.stop(audio_ctx.currentTime + alert_len_s);
}

function handle_audible_change(evt)
{
    const vol = document.getElementById("vol");
    vol.disabled = !this.checked;
}

function handle_footer_toggle()
{
    const footer_more = document.getElementById("footer_more");
    footer_more.classList.toggle("hidden");
    const footer_toggle = document.getElementById("footer_toggle");
    if (footer_more.classList.contains("hidden"))
    {
        footer_toggle.innerHTML = "More &raquo;";
    }
    else
    {
        footer_toggle.innerHTML = "&laquo; Less";
    }
}

