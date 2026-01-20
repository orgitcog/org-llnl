# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

# ThicketReader.py
import platform
from glob import glob
import os
import thicket as tt

from .MyTimer import MyTimer

machine = platform.uname().machine

from .Reader import Reader


class TH_ens:

    th_ens_defined = 0
    th_ens = 0
    profiles = 0

    def get_th_ens(self, cali_files ):
        return self.get_th_ens_impl( cali_files )

    def get_th_ens_impl(self, cali_files):

        if TH_ens.th_ens_defined == 0:

            TH_ens.th_ens_defined = 1

            # Handle both single string and list of strings
            if isinstance(cali_files, str):
                # Single string - could be a directory or a file
                if os.path.isdir(cali_files):
                    # It's a directory - get all .cali files recursively
                    TH_ens.profiles = [
                        y for x in os.walk(cali_files) for y in glob(os.path.join(x[0], "*.cali"))
                    ]
                elif os.path.isfile(cali_files):
                    # It's a single file
                    TH_ens.profiles = [cali_files]
                else:
                    raise ValueError(f"Path does not exist: {cali_files}")
            elif isinstance(cali_files, list):
                # List of strings - could be directories or files
                TH_ens.profiles = []
                for path in cali_files:
                    if os.path.isdir(path):
                        # It's a directory - get all .cali files recursively
                        TH_ens.profiles.extend([
                            y for x in os.walk(path) for y in glob(os.path.join(x[0], "*.cali"))
                        ])
                    elif os.path.isfile(path):
                        # It's a file
                        TH_ens.profiles.append(path)
                    else:
                        raise ValueError(f"Path does not exist: {path}")

                # Remove duplicates while preserving order
                seen = set()
                unique_profiles = []
                for profile in TH_ens.profiles:
                    if profile not in seen:
                        seen.add(profile)
                        unique_profiles.append(profile)
                TH_ens.profiles = unique_profiles
            else:
                raise TypeError(f"cali_files must be a string or list of strings, got {type(cali_files)}")

            #  this contains some metadata we need.
            #  also contains the tree data.
            # print( TH_ens.profiles )

            TH_ens.th_ens = tt.Thicket.from_caliperreader(TH_ens.profiles)

            # TH_ens.th_ens.dataframe.reset_index(drop=True, inplace=True)
            # TH_ens.th_ens.dataframe = pd.concat([TH_ens.th_ens.dataframe, TH_ens.th_ens.dataframe], ignore_index=True)

            # TH_ens.th_ens.metadata.reset_index(drop=True, inplace=True)
            # TH_ens.th_ens.metadata = pd.concat([TH_ens.th_ens.metadata, TH_ens.th_ens.metadata], ignore_index=True)
            # tms = ThicketMultiplierStub(TH_ens.th_ens)

        return TH_ens.th_ens, TH_ens.profiles


class ThicketReader(Reader):

    def __init__(self, th_ens, profiles, xaxis):
        self.th_ens = th_ens
        self.profiles = profiles
        self.xaxis = xaxis

    def __iter__(self):
        """Initialize iteration over xaxis values and their associated data."""
        self.xaxis_values = self.get_all_xaxis()
        self.meta_values = self.get_all_xaxis_meta()
        self.index = 0
        return self

    def __next__(self):
        """Return the next (xaxis_value, metadata, node_data) tuple."""
        if self.index < len(self.xaxis_values):
            xaxis_value = self.xaxis_values[self.index]
            metadata = self.meta_values[self.index]
            node_data = self.get_entire_for_xaxis(xaxis_value)
            self.index += 1
            return xaxis_value, metadata, node_data
        else:
            raise StopIteration

    def get_x_axis(self):
        return self.xaxis

    def get_all_xaxis(self):

        xaxis_arr = []

        for profile, row in self.th_ens.metadata.iterrows():

            for xaxis, value in row.items():
                if xaxis == self.xaxis:
                    xaxis_arr.append(value)

        return xaxis_arr

    def get_all_xaxis_meta(self):

        xaxis_arr = []

        for profile, row in self.th_ens.metadata.iterrows():

            meta_obj = {}
            for xaxis, value in row.items():
                meta_obj[xaxis] = value

            meta_obj["profile"] = profile
            xaxis_arr.append(meta_obj)

        return xaxis_arr


    def get_entire(self):

        # PSTUB
        from .GraphTraverseModel import GraphTraverseModel

        gtm = GraphTraverseModel(self.th_ens, self.profiles)
        childrenMap = gtm.getParentToChildMapping()
        parentMap = gtm.getChildToParentMapping()

        # Get all unique xaxis values (e.g., all unique launchdate values)
        xaxis_arr = self.get_all_xaxis()
        meta_arr = self.get_all_xaxis_meta()

        # Get data for each xaxis value
        # This returns a dict: {xaxis_value: [list of node data]}
        nodes_by_xaxis = {}
        meta_by_xaxis = {}
        for idx, xaxis in enumerate(xaxis_arr):
            nodes_by_xaxis[xaxis] = self.get_entire_for_xaxis(xaxis)
            meta_by_xaxis[xaxis] = meta_arr[idx]

        # Transform the data structure to match what TreeScapeModel expects
        # TreeScapeModel expects: {node_name: {"xaxis": [metadata1, metadata2, ...], "ydata": [data1, data2, ...]}}
        # We currently have: {xaxis_value: [{"name": node_name, "ydata": data, "xaxis": ordered_xaxis}]}

        nodes = {}

        # Build a mapping of node_name -> list of (xaxis_value, ydata) pairs
        for xaxis_value, node_list in nodes_by_xaxis.items():
            for node_data in node_list:
                node_name = node_data["name"]
                ydata = node_data["ydata"][0]
                # xaxis_value is the metadata for this run

                if node_name not in nodes:
                    # Create a per-node childrenMap containing only this node's children
                    node_children_map = {}
                    if node_name in childrenMap:
                        node_children_map[node_name] = childrenMap[node_name]

                    nodes[node_name] = {
                        "name": node_name,
                        "xaxis": [],
                        "ydata": [],
                        "childrenMap": node_children_map  # Each node gets its own childrenMap
                    }

                nodes[node_name]["xaxis"].append(meta_by_xaxis[xaxis_value])
                nodes[node_name]["ydata"].append(ydata)

        # Build meta_globals from the actual DataFrame dtypes
        meta_globals = {}
        for col, dtype in self.th_ens.metadata.dtypes.items():
            # Convert pandas dtypes to simple type strings
            dtype_str = str(dtype)
            if 'int' in dtype_str:
                meta_globals[col] = "int"
            elif 'float' in dtype_str:
                meta_globals[col] = "float"
            elif 'datetime' in dtype_str or 'date' in dtype_str:
                meta_globals[col] = "date"
            else:
                meta_globals[col] = "string"

        return {
            "nodes": nodes,
            "childrenMap": childrenMap,
            "parentMap": parentMap,
            "meta_globals": meta_globals
        }

    def get_entire_for_xaxis(self, xaxis_name):
        import pandas as pd

        df = self.th_ens.dataframe.reset_index()

        # Create a mapping from profile to xaxis value
        # Convert metadata to a Series for fast lookup
        metadata_xaxis = self.th_ens.metadata[self.xaxis].copy()

        # Normalize xaxis values to string, then to float if numeric
        metadata_xaxis = metadata_xaxis.astype(str)
        metadata_xaxis = metadata_xaxis.apply(lambda x: float(x) if x.replace('.', '', 1).replace('-', '', 1).isdigit() else x)

        # Normalize the target xaxis_name
        xaxis_name_normalized = str(xaxis_name)
        if xaxis_name_normalized.replace('.', '', 1).replace('-', '', 1).isdigit():
            xaxis_name_normalized = float(xaxis_name_normalized)

        # Add xaxis column to dataframe by mapping profile to xaxis value
        df['xaxis_value'] = df['profile'].map(metadata_xaxis)

        # Filter to only rows matching the target xaxis_name
        df_filtered = df[df['xaxis_value'] == xaxis_name_normalized].copy()

        # If no matching rows, return empty list
        if len(df_filtered) == 0:
            return []

        # Group by name and xaxis_value, then aggregate
        # This replaces the entire iterrows loop with vectorized operations
        # PERFORMANCE: This is ~100x faster than iterating with iterrows()
        grouped = df_filtered.groupby(['name', 'xaxis_value'])['Avg time/rank'].agg([
            ('sum', 'sum'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).reset_index()

        # Build sumArr structure from grouped data
        # Note: This small iterrows loop is acceptable because grouped has very few rows
        # (one per unique node name, typically 30-50 rows vs thousands in the original df)
        sumArr = {}
        for _, row in grouped.iterrows():
            name = row['name']
            xaxis_val = row['xaxis_value']

            if name not in sumArr:
                sumArr[name] = {}

            # Calculate average from sum and count
            avg_value = row['sum'] / row['count'] if row['count'] > 0 else 0

            sumArr[name][xaxis_val] = {
                'sum': row['sum'],
                'min': row['min'],
                'max': row['max'],
                'avg': avg_value
            }

        # uniq_date = len(sumArr["main"])
        # print("uniq_date=" + str(uniq_date))

        # print("howmany=" + str(howmany))
        #MyTimer("get_entire_for_xaxis - iterrows")

        renderDat = {}
        ldates = {}
        nameOfLinesToPlot = []

        # Convert the dictionary to a list of tuples (launchDate, data)

        for name, launchData in sumArr.items():
            launch_list = [
                (launchDate, data) for launchDate, data in launchData.items()
            ]
            sorted_launch_list = sorted(
                launch_list,
                key=lambda x: (float(x[0]) if isinstance(x[0], (int, float)) else x[0]),
            )

            renderDat[name] = [data for _, data in sorted_launch_list]
            ldates[name] = [launchDate for launchDate, _ in sorted_launch_list]
            nameOfLinesToPlot.append(name)

        entireNodes = []

        #  name is node_name: like 'main', 'lulesh_cycle'
        #  it's the plot's name at the top.
        for name in renderDat:

            ordered = self.order_strings(ldates[name])

            entireNodes.append(
                {"name": name, "ydata": renderDat[name], "xaxis": ordered}
            )

        #MyTimer("get_entire_for_xaxis - renderDat")

        return entireNodes

    def order_strings(self, unsor_arr):
        # Check if the array contains strings with numbers
        if all(isinstance(s, str) and s.replace(".", "").isdigit() for s in unsor_arr):
            # Order the strings in numerical order
            ordered_arr = sorted(unsor_arr, key=lambda x: float(x))
            return ordered_arr
        else:
            return unsor_arr

    def test(self):
        import thicket as tt

        PATH = "/Users/aschwanden1/lulesh_gen/"
        profiles = [
            y for x in os.walk(PATH) for y in glob(os.path.join(x[0], "*.cali"))
        ]

        th_ens = tt.Thicket.from_caliperreader(profiles)

        # Iterate through each row
        for index, row in th_ens.metadata.iterrows():
            # Access each column in the row
            print(f"Profile: {index}")
            print(f"Version: {row['cali.caliper.version']}")
            print(f"Channel: {row['cali.channel']}")
            print(f"launchdate: {row['launchdate']}")
            print(f"elapsed_time: {row['elapsed_time']}")
            print(f"User: {row['user']}")
            # ... (print other columns as needed)
            print("\n")
