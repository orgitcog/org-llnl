# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType

from sonar_driver.spark import analytics

def pool_counts(sparkdf, start_column, end_column, var):
    """
    Calculate minimum pools for each unique allocation size.
    :param sparkdf: Input Spark dataframe.
    :param start_column: Start time column name
    :param end_column: End time column name
    :param var: Variable on which to generate pool counts.
    :return: A list of dicts with size and count of each pool.
    """
    return analytics.pool(sparkdf, start_column, end_column, var)[0]
    
def max_memory_unpooled(sparkdf, start_column, end_column, var):
    """
    Calculate maximum active memory needed for an unpooled allocation configuration.
    :param sparkdf: Input Spark dataframe.
    :param start_column: Start time column name
    :param end_column: End time column name
    :param var: Variable for which to calculate metric.
    :return: Float of maximum active memory for unpooled allocations.
    """
    return analytics.pool(sparkdf, start_column, end_column, var)[1]

def max_memory_pooled(sparkdf, start_column, end_column, var):
    """
    Calculate maximum active memory needed for a pooled allocation configuration.
    :param sparkdf: Input Spark dataframe.
    :param start_column: Start time column name
    :param end_column: End time column name
    :param var: Variable for which to calculate metric.
    :return: Float of maximum active memory for pooled allocations.
    """
    pools = pool_counts(sparkdf, start_column, end_column, var)
    return sum([p[var] * p['count'] for p in pools])

def total_bytesecs_unpooled(sparkdf, start_column, end_column, var, precision=1e9):
    """
    Calculate total bytesecs (1 byte of memory allocated for 1 sec) needed for an 
        unpooled allocation configuration.
    :param sparkdf: Input Spark dataframe.
    :param start_column: Start time column name
    :param end_column: End time column name
    :param var: Variable for which to calculate metric.
    :return: Float of total bytesecs for unpooled allocations.
    """
    calc_bytesecs = udf(lambda size, alloc, free: size * (free - alloc), DoubleType())

    return (
        sparkdf
            .withColumn('bytesecs', calc_bytesecs(col(var), col(start_column), col(end_column)))
    ).rdd.map(lambda x: float(x['bytesecs'])).reduce(lambda x, y: x+y) / precision

def total_bytesecs_pooled(sparkdf, start_column, end_column, var, precision=1e9):
    """
    Calculate total bytesecs (1 byte of memory allocated for 1 sec) needed for a
        pooled allocation configuration.
    :param sparkdf: Input Spark dataframe.
    :param start_column: Start time column name
    :param end_column: End time column name
    :param var: Variable for which to calculate metric.
    :return: Float of total bytesecs for pooled allocations.
    """
    pools = pool_counts(sparkdf, start_column, end_column, var)
    
    min_time = sparkdf.agg({start_column: "min"}).collect()[0][0] / precision
    max_time = sparkdf.agg({end_column: "max"}).collect()[0][0] / precision
    range_time = max_time - min_time

    return sum([p[var] * p['count'] * range_time for p in pools])