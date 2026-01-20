# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from datetime import datetime, timedelta

from pyspark.sql.functions import desc

from bokeh.layouts import column
from bokeh.plotting import figure, show
from bokeh.models import CustomJS, ColumnDataSource, HoverTool, DatetimeTickFormatter

def to_timedelta(runtime):
    """
    Utility function to convert runtimes into a number of seconds.
    """
    if len(runtime) == 8:
        t = datetime.strptime(runtime,"%H:%M:%S")
        return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds()
    
    t = datetime.strptime(runtime,"%d-%H:%M:%S")
    return timedelta(days=t.day, hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds()

def group_hist(df, hist_var):
    """
    Utility function to generate bins and counts for grouped histogram.
    :param df: Input dataframe.
    :param hist_var: Histogram variable column name.
    :return: Array of bins.
    :return: Array of the counts for each bin.
    :return: Original df which each object labelled as part of which bin.
    """
    if 'time' in hist_var or 'Time' in hist_var:
        hist_array = df[hist_var].map(to_timedelta)

        secs_in_hr = 3600
        num_bins = int((max(hist_array) - min(hist_array)) // secs_in_hr + 1)
        counts, bins = np.histogram(hist_array, bins=num_bins, 
                                    range=(min(hist_array) // secs_in_hr, num_bins * secs_in_hr))
        
        bins = [str(int(b)) for b in bins / secs_in_hr][:-1]
        counts = list(counts)
        df_bins = [str(int(b / secs_in_hr)) for b in hist_array] 
        return bins, counts, df_bins

def indiv_hist(df, hist_var):
    """
    Utility function to generate bins and counts for individualized histogram.
    :param df: Input dataframe.
    :param hist_var: Histogram variable column name.
    :return: Array of bins.
    :return: Array of the counts for each bin.
    :return: Original df which each object labelled as part of which bin.
    """
    hist_counts = df.groupBy(hist_var).count().sort(desc('count')).toPandas()
    bins = [str(s) for s in hist_counts[hist_var]]
    counts = list(hist_counts['count'])
    df_bins = [str(s) for s in df.toPandas()[hist_var]]
    return bins, counts, df_bins
        
def gantt_dtf():
    """
    Utility function to set datetime formats on Bokeh plot.
    """
    return DatetimeTickFormatter(
        seconds = ['%F %H:%M:%S'],
        minsec = ['%F %H:%M:%S'],
        minutes = ['%F %H:%M'],
        hourmin = ['%F %H:%M'],
        hours = ['%F %H:%M'],
        days = ['%F'],
        months = ['%Y-%m'],
        years = ['%Y']
    )

def gantt_hover_tool(start_column, end_column, hist_var):
    """
    Utility function to set datetime formats on hover tool.
    """
    return HoverTool(
        tooltips=[
            (start_column, '@' + start_column + '{%F %H:%M:%S.%3N}'),
            (end_column, '@' + end_column + '{%F %H:%M:%S.%3N}'),
            (hist_var, '@' + hist_var)
        ],

        formatters={
            start_column: 'datetime',
            end_column: 'datetime',
        },
    )

def quote(string):
    """
    Utility function to put quotes around var names in string.
    """
    return'\'{}\''.format(string)

def hist_gantt_callback():
    """
    Utility function to return JS callback code for histogram gantt function.
    """
    return """
        var d0 = c0.data;
        var d1 = cb_obj.data;
        var d2 = c2.data;

        var inds = cb_obj.selected.indices;
        var metrics = [];
        for (var i = 0; i < inds.length; i++) {{
            metrics.push(d1['bins'][inds[i]]);
        }}

        d2[{start_column}] = [];
        d2[{end_column}] = [];
        d2[{hist_var}] = [];
        d2['bottom'] = [];
        d2['top'] = [];
        
        if ({gantt_pooled}) {{
            var active = [];
            for (var i = 0; i < d0['index'].length; i++) {{
                if (metrics.includes(d0[{bin_column}][i])) {{
                    var start = d0[{start_column}][i];
                    var end = d0[{end_column}][i];

                    d2[{start_column}].push(d0[{start_column}][i]);
                    d2[{end_column}].push(d0[{end_column}][i]);
                    d2[{hist_var}].push(d0[{hist_var}][i]);

                    var pool = -1;
                    var min_end = Number.MAX_SAFE_INTEGER;
                    for (var j = 0; j < active.length; j++) {{
                        jth_end = active[j];
                        if (start > jth_end && jth_end < min_end) {{
                            pool = j;
                            min_end = jth_end;
                        }}
                    }}
                    
                    if (pool == -1) {{
                        active.push(end);
                        pool = active.length - 1;
                    }} else {{
                        active[pool] = end;
                    }}
                    
                    d2['bottom'].push(pool - 0.25);
                    d2['top'].push(pool + 0.25);
                }}
            }}
            
            f2.title.setv({{"text": "Gantt Chart of Allocations of Size " + metrics.toString()}});
            f2.x_range.setv({{"start": Math.min(...d2[{start_column}]), "end": Math.max(...d2[{end_column}])}});
            f2.y_range.setv({{"start": -1, "end": active.length + 1}});
            
        }} else {{
            var count = 0;
            for (var i = 0; i < d0['index'].length; i++) {{
                if (metrics.includes(d0[{bin_column}][i])) {{
                    d2[{start_column}].push(d0[{start_column}][i]);
                    d2[{end_column}].push(d0[{end_column}][i]);
                    d2[{hist_var}].push(d0[{hist_var}][i]);
                    d2['bottom'].push(count + 0.75);
                    d2['top'].push(count + 1.25);
                    count++;
                }}
            }}
            
            f2.title.setv({{"text": "Gantt Chart of Allocations of Size " + metrics.toString()}});
            f2.x_range.setv({{"start": Math.min(...d2[{start_column}]), "end": Math.max(...d2[{end_column}])}});
            f2.y_range.setv({{"start": 0, "end": count + 1}});
        }}
        
        c2.change.emit();
    """