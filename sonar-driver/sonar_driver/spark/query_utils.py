# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import re

from pyspark.sql.functions import udf, col, explode, lit, split
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, IntegerType, StringType, TimestampType

def query_time_range(sparkdf, start_column, end_column, time_range, 
                     filter_starts=False, filter_ends=False, time_type=TimestampType):
    """
    Query objects within a time range.
    :param sparkdf: Input Spark dataframe.
    :param start_column: Str of start column name.
    :param end_column: Str of end column name.
    :param time_range: List, array-like of time range start and end.
    :param filter_starts: Bool, if set True, will only include objects with start times within time range.
    :param filter_ends: Bool, if set True, will only include objects with end times within time range.
    :param time_type: Type of time - usually DoubleType or TimestampType
    :return: A Spark dataframe with jobs whose start times or end times are within specified time range.
    """
    start_range, end_range = time_range[0], time_range[1]
    
    sparkdf = (
        sparkdf
            .where(
                (
                    (col(start_column) > lit(start_range).cast(time_type())) & 
                    (col(start_column) < lit(end_range).cast(time_type()))
                ) 
                | 
                (
                    (col(end_column) > lit(start_range).cast(time_type())) & 
                    (col(end_column) < lit(end_range).cast(time_type()))
                )
            )
    )

    if filter_starts:
        sparkdf = (
            sparkdf
                .where(
                    (col(start_column) > lit(start_range).cast(time_type())) & 
                    (col(start_column) < lit(end_range).cast(time_type()))
                )
        )

    if filter_ends:
        sparkdf = (
            sparkdf
                .where(
                    (col(end_column) > lit(start_range).cast(time_type())) & 
                    (col(end_column) < lit(end_range).cast(time_type()))
                )
        )
        
    return sparkdf
    
def nodes_funcs_dict():
    def jobdata_func(node):
        if '[' in node:
            splits = [i for i, v in enumerate(node) if 
                      v == '-' or v == ',' or v == '[' or v == ']']

            cluster_name = node[:splits[0]]
            node_lst = []
            for i in range(len(splits) - 1):
                node_lst.append(int(node[splits[i] + 1: splits[i + 1]]))

            return cluster_name, node_lst

        elif node.isalpha():
            return node, 0

        else:
            m = re.search("\d", node)
            i = m.start()
            
            cluster_name, node_num = node[:i], int(node[i:])
            return cluster_name, [node_num, node_num]
        
    return {
        'jobdata': jobdata_func
    }


def query_nodes(sparkdf, table, nodes_column, nodes):
    """
    Query objects on certain nodes.
    :param sparkdf: Input Spark dataframe.
    :param table: Str of table name.
    :param nodes_column: Str of nodes column name.
    :param nodes: List, array-like of clusters and nodes to filter. Format should follow schema of specified table.
    :return: A Spark dataframe with objects on specified nodes.
    """
    nodes_func = nodes_funcs_dict()[table]
    
    input_nodes = {}
    for n in nodes:
        cluster_name, node_lst = nodes_func(n)
        input_nodes[cluster_name] = node_lst
        
    def isin_nodes(node):
        cluster_name, node_lst = nodes_func(node)
        if cluster_name in input_nodes:
            input_node_lst = input_nodes[cluster_name]

            if input_node_lst == 0:
                return True

            for i in range(len(node_lst) // 2):
                on_nodes = False
                for j in range(len(input_node_lst) // 2):
                    within_nodes = (
                        node_lst[2 * i] >= input_node_lst[2 * j] and 
                        node_lst[2 * i + 1] <= input_node_lst[2 * j + 1]
                    )
                    if within_nodes:
                        on_nodes = True
                        break

                if not on_nodes:
                    return False

            return True

        return False    
        
    isin_nodes_udf = udf(isin_nodes, BooleanType())
    return sparkdf.filter(isin_nodes_udf(col(nodes_column)))

def query_users(sparkdf, users_column, users):
    return sparkdf.filter(col(users_column).isin(*users) == True)