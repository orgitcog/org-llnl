# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from pyspark.sql.functions import col, desc

from bokeh.layouts import column
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import CustomJS, ColumnDataSource, HoverTool
output_notebook()

from sonar_driver.spark import bokeh_utils

def plot_analytics(x_axis, y_axis, plot_title=None, x_title=None, y_title=None):
    """
    General plotting function for a plotly graph.
    :x_axis: List of x values.
    :y_axis: List of y values.
    :plot_title: Title of plot.
    :x_title: Title of x-axis.
    :y_title: Title of y-axis.   
    :return: None.
    """
    plotly.offline.init_notebook_mode(connected=True)
    
    plotly.offline.iplot({
        "data": [go.Scatter(x=x_axis, 
                            y=y_axis)],
        "layout": go.Layout(title=plot_title,
                            xaxis=dict(title=x_title),
                            yaxis=dict(title=y_title))
    })

def plot_derivatives(sparkdf, column, window_size, slide_length):
    """
    Plots job start/completion rate versus time for a specified window size and slide length. 
    :param sparkdf: Input Spark dataFrame.
    :param column: Variable for which to calculate discrete derivative..
    :window_size: Window size of discrete derivatives.
    :slide_length: Slide length of window measurements.
    :return: None.
    """
    df = sparkdf.toPandas()
    
    x_axis = df['time']
    y_axis = df['count']
    
    verb = "Completed" if column == "EndTime" else "Started"
    plot_title = "Number of Jobs " + verb + " Per Sec, Window=" + str(window_size) + ", Slide=" + str(slide_length)
    x_title = "Time"
    y_title = "Jobs " + verb + "/Sec in Previous " + str(window_size) + " Sec"
    
    plot_analytics(x_axis, y_axis, plot_title=plot_title, x_title=x_title, y_title=y_title)

def plot_integrals(sparkdf, slide_length):
    """
    Plots number of jobs running concurrently versus time with a specified slide length to determine timesteps.
    :sparkdf: Input Spark DataFrame.
    :slide_length: Slide length of timesteps.
    :return: None.
    """
    df = sparkdf.toPandas()
    
    x_axis = df['time']
    y_axis = df['count']
    
    plot_title = "Number of Jobs Running vs. Time, Slide Length=" + str(slide_length)
    x_title = "Time"
    y_title = "Number of Jobs Running"
    
    plot_analytics(x_axis, y_axis, plot_title=plot_title, x_title=x_title, y_title=y_title)
    
def plot_hist_gantt(sparkdf, start_column, end_column, hist_var, df=pd.DataFrame(),
                    hist_grouped=False, gantt_pooled=False, unit=None):
    """
    Plots histogram and linked Gantt chart of objects in Spark dataframe.
    :sparkdf: Input Spark dataframe.
    :param start_column: Start time column name.
    :param end_column: End time column name.
    :param hist_var: Histogram variable column name.
    :df: Pandas dataframe which can be optionally inputed to reduce redundant operations.
    :param hist_grouped: If True, each bin in the histogram will be a range of values; if False, each bin will be an individual value.
    :param gantt_pooled: if True, resulting Gantt chart will pool objects; if False, each object will reside on its own horizontal line.
    :param unit: Unit of timestamps in sparkdf.
    :return: None.
    """
    
    if df.empty:
        df = sparkdf.sort(start_column).toPandas()
    
    if unit:
        df[start_column] = pd.to_datetime(df[start_column], unit=unit)
        df[end_column] = pd.to_datetime(df[end_column], unit=unit)
    
    # Histogram
    if hist_grouped:
        bins, counts, df_bins = bokeh_utils.group_hist(df, hist_var)
    else:
        bins, counts, df_bins = bokeh_utils.indiv_hist(sparkdf, hist_var)
        
    bin_column = 'bin'
    df[bin_column] = df_bins
    c0 = ColumnDataSource(data=df)
    
    c1 = ColumnDataSource(data=dict(bins=bins, counts=counts))
    
    f1 = figure(title="Distribution of " + hist_var.capitalize() + "s", 
                tools='box_select', height=500, x_range=bins)
    f1.vbar(source=c1, x='bins', top='counts', width=0.9)
    
    bar_width = 37 if hist_grouped else 75
    calc_width = lambda h, w: 500 if len(h) < 5 else len(h) * w
    f1.width = calc_width(counts, bar_width)
    
    f1.xaxis.axis_label = hist_var.capitalize()
    f1.xgrid.grid_line_color = None
    f1.yaxis.axis_label = 'Count'
    f1.y_range.start = 0
    
    # Gantt Chart
    c2 = ColumnDataSource(data={start_column: [], end_column: [], hist_var: [], 
                                'bottom': [], 'top': []})
    
    f2 = figure(title='Gantt Chart', tools='box_zoom,reset', width=1000, height=500, 
                x_axis_type='datetime')
    f2.quad(source=c2, left=start_column, right=end_column, bottom='bottom', top='top')
    
    f2.xaxis.axis_label = 'Time'
    f2.xaxis.formatter = bokeh_utils.gantt_dtf() 
    
    f2.add_tools(bokeh_utils.gantt_hover_tool(start_column, end_column, hist_var))
    
    quote = bokeh_utils.quote
    jscode = bokeh_utils.hist_gantt_callback().format(start_column = quote(start_column), end_column = quote(end_column), 
                                                      bin_column = quote(bin_column), hist_var = quote(hist_var),
                                                      gantt_pooled = str(gantt_pooled).lower())
    c1.callback = CustomJS(code=jscode, args=dict(c0=c0, c2=c2, f2=f2))
    
    layout = column(f1, f2)
    show(layout)