# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from pyspark.sql.functions import lead, lag
from pyspark.sql.window import Window
from pyspark.sql.functions import udf, col, explode, lit, split
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, IntegerType, StringType, TimestampType

def split_dataframes(sparkdf, column):
    """
    Split a dataframe into multiple dataframes by distinct values along a column.
    :param sparkdf: Input Spark dataframe.
    :param column: Column name to split over.
    :return: A list of Spark dataframes, where each has a cardinality of 1 in the column.
    """

    # Get distinct values
    distinct_values = [
        row[column] for row in
        sparkdf.select(column).distinct().collect()
    ]

    # Filter by each distinct value
    return [
        sparkdf.filter(col(column) == value)
        for value in distinct_values
    ]


def finite_difference(sparkdf, xaxis, yaxes, window_size, partitionAxis=None, monotonically_increasing=False):
    """
    Calculate the finite difference dY/dX for 1 or more Y=f(X) axes with respect to a single X axis
    :param sparkdf: Input Spark dataframe.
    :param xaxis: Column name for X axis.
    :param yaxes: List of column names for Y axes.
    :param window_size: Width of window over which to calculate finite difference (in number of points).
    :param partitionAxis: Categorical axis to partition
    :param monotonically_increasing: Whether Y axes should be monotonically increasing (e.g. counters). If set,
           negative finite differences in Y will be set to zero.
    :return: A Spark dataframe with one new column per Y axis calculated as dY/dX.
    """

    original_columns = sparkdf.schema.fieldNames()

    # Get the first value, so we can use it to define the initial "previous value"
    first_row = sparkdf.first()
    first_x = first_row[xaxis]

    # Slide over this window
    window = Window.orderBy(xaxis)
    if partitionAxis is not None:
        window = window.partitionBy(partitionAxis)

    # Create function to calculate difference between two columns
    delta_fn = udf(lambda col1, col2: col1 - col2, DoubleType())
    div_fn = udf(lambda col1, col2: col1 / col2 if col2 > 0 else float('nan'), DoubleType())

    # Get delta X
    xaxis_lag = xaxis + '_lag_' + str(window_size)
    xaxis_delta = xaxis + '_delta_' + str(window_size)
    df = (
        sparkdf
            .withColumn(xaxis, sparkdf[xaxis].cast(DoubleType()))
            .withColumn(xaxis_lag, lag(xaxis, window_size, first_x).over(window))
            .withColumn(xaxis_delta, delta_fn(col(xaxis), col(xaxis_lag)))
            .drop(xaxis_lag)
    )

    # Get delta y and dY/dX for each y
    for yaxis in yaxes:
        yaxis_lag = yaxis + '_lag_' + str(window_size)
        yaxis_delta = yaxis + '_delta_' + str(window_size)
        rate = 'rate_' + yaxis + '_over_' + xaxis
        rate_lag = rate + '_lag'
        rate_lead = rate + '_lead'

        first_y = first_row[yaxis]

        df = (
            df
                .withColumn(yaxis, sparkdf[yaxis].cast(DoubleType()))
                .withColumn(yaxis_lag, lag(yaxis, window_size, first_y).over(window))
                .withColumn(yaxis_delta, delta_fn(col(yaxis), col(yaxis_lag)))
        )

        if monotonically_increasing:
            df[yaxis_delta] = df[yaxis_delta].map(lambda d: max(0, d))

        df = (
            df
                .withColumn(rate, div_fn(col(yaxis_delta), col(xaxis_delta)))
                .drop(yaxis_lag)
                .drop(yaxis_delta)

                # Determine when the delta changes (lead and lag by 1)
                .withColumn(rate_lag, lag(rate, 1, 1).over(window))
                .withColumn(rate_lead, lead(rate, 1, 1).over(window))

                # Only get values where rate is different from before or after
                .filter(rate + '!=' + rate_lag + ' OR ' + rate + '!=' + rate_lead)

                .drop(rate_lag)
                .drop(rate_lead)
        )

    return df.drop(xaxis_delta)



def discrete_derivatives(sparkdf, column, window_size, slide_length):
    """
    Calculate discrete derivatives for column variable.
    :param sparkdf: Input Spark dataframe.
    :param column: Variable for which to calculate discrete derivative.
    :param window_size: Time range (secs) over which to calculate derivatives.
    :param slide_length: Amount of time (secs) to slide window at each step.
    :return: A Spark dataframe with discrete derivative calculated at each timestep.
    """
    sparkdf = timestamp_to_double(sparkdf)
            
    def range_windows(column, window, slide):
        first = int(window + slide * ((column - window) // slide + 1))
        last = int(window + slide * (column // slide))
        return [first + slide * x for x in range(0, (last - first) // slide + 1)]
    range_windows = udf(range_windows, ArrayType(IntegerType()))
    
    return (
        sparkdf
            .withColumn('range', range_windows(col(column), lit(window_size), lit(slide_length)))
            .select(explode(col('range')).alias('time'))
            .select(col('time').cast(TimestampType()))
            .groupBy('time')
            .count()
            .sort('time')
            .withColumn('count', udf(lambda x: x / window_size, DoubleType())(col('count')))
    )

def discrete_integrals(sparkdf, start_column, end_column, slide_length):
    """
    Calculate discrete integrals (number of active jobs vs. time).
    :param sparkdf: Input Spark dataframe.
    :param start_column: Start time column name.
    :param end_column: End time column name.
    :param slide_length: Amount of time (secs) between each timestep.
    :return: A Spark dataframe with discrete integral calculated at each timestep.
    """
    sparkdf = timestamp_to_double(sparkdf)
            
    def range_times(col1, col2, slide):
        first = int((col1 // slide + 1) * slide)
        last = int((col2 // slide) * slide)
        if first > last:
            return []
        return [first + slide * x for x in range(0, (last - first) // slide + 1)]
    range_times = udf(range_times, ArrayType(IntegerType()))
    
    return (
        sparkdf
            .withColumn('range', range_times(col(start_column), col(end_column), lit(slide_length)))
            .select(explode(col('range')).alias('time'))
            .select(col('time').cast(TimestampType()))
            .groupBy('time')
            .count()
            .sort('time')
            .where(col('time').isNotNull())
    )

def timestamp_to_double(sparkdf):
    """
    Utility function to cast columns of type 'timestamp' to type 'double.'
    """
    for dtype in sparkdf.dtypes:
        if dtype[1] == 'timestamp':
            sparkdf = sparkdf.withColumn(dtype[0], col(dtype[0]).cast(DoubleType()))
    return sparkdf

def sorted_dicts(sparkdf, start_column, end_column, var):
    """
    Utility function to create sorted dicts for pool function.
    """
    pandasdf = sparkdf.toPandas()
    
    starts_dict = dict(zip(pandasdf[start_column], pandasdf[var]))
    ends_dict = dict(zip(pandasdf[end_column], pandasdf[var]))
    starts_sorted = sorted(starts_dict)
    ends_sorted = sorted(ends_dict)
    
    return starts_dict, ends_dict, starts_sorted, ends_sorted

def pool(sparkdf, start_column, end_column, var):
    """
    Generate pools and calculate maximum var unpooled.
    :param sparkdf: Input Spark dataframe.
    :param start_column: Start time column name.
    :param end_column: End time column name.
    :param var: Variable for which to calculate metric.
    :return: A Spark dataframe with pools (sizes and counts).
    :return: Maximum active metric for var.
    """
    starts_dict, ends_dict, starts_sorted, ends_sorted = sorted_dicts(sparkdf, start_column, end_column, var)
    
    size_groups = {s:{'current': 0, 'max': 0} for s in [r.size for r in sparkdf.select(var).distinct().collect()]}
    active = {'current': 0, 'max': 0}
    
    start_index, end_index = 0, 0
    while start_index < len(starts_sorted) or end_index < len(ends_sorted):
        start, end = None, ends_sorted[end_index]
        if start_index < len(starts_sorted):
            start = starts_sorted[start_index]

        if start is None or start > end:
            group = size_groups[ends_dict[end]]
            group['current'] -= 1
            
            active['current'] -= ends_dict[end]
            
            end_index += 1
        else:
            group = size_groups[starts_dict[start]]
            group['current'] += 1
            if group['current'] > group['max']:
                group['max'] = group['current']
                
            active['current'] += starts_dict[start]
            if active['current'] > active['max']:
                active['max'] = active['current']
                
            start_index += 1
    
    pool_counts = [{var: int(s), 'count': int(size_groups[s]['max'])} for s in size_groups.keys()]
    max_unpooled = active['max']
    
    return pool_counts, max_unpooled


    
    
