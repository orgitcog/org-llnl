# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import re

from pyspark.sql.functions import udf, col, explode, lit, split
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, IntegerType, StringType, TimestampType

from sonar_driver.spark import analytics
from sonar_driver.spark import query_utils

def query_jobs(sparkdf, schema, time_range=None, filter_starts=False, filter_ends=False, time_type=TimestampType,
               nodes=None, users=None):
    """
    Query jobs within a time range, on certain clusters, nodes, and run by certain users.
    :param sparkdf: Input Spark dataframe.
    :param schema: Dict which should follow this format:
           {
               'table': str of table name,
               'start': str of start time column name,
               'end': str of end time column name,
               'nodes': str of node column name,
               'users': str of user column name
           }
    :param time_range: List, array-like of time range start and end and optional third argument which, if set,
           only start times ('StartTime') or end times ('EndTime') will be within time range. 
    :param nodes: List, array-like of clusters and nodes to filter. Format should follow schema of specified table.
    :param users: List, array-like of users to query.
    :return: A Spark dataframe with jobs whose start times or end times are within specified time range.
    """
    if 'name' in schema:
        table = schema['name']
    if 'start' in schema:
        start_column = schema['start']
    if 'end' in schema:
        end_column = schema['end']
    if 'nodes' in schema:
        nodes_column = schema['nodes']
    if 'users' in schema:
        users_column = schema['users']
    
    if time_range:
        sparkdf = query_utils.query_time_range(sparkdf, start_column, end_column, time_range, 
                                               filter_starts, filter_ends, time_type)
        
    if nodes:
        sparkdf = query_utils.query_nodes(sparkdf, table, nodes_column, nodes)

    if users:
        sparkdf = query_utils.query_users(sparkdf, users_column, users)
        
    return sparkdf
    
    