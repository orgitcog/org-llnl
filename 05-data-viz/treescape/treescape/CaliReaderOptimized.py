# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import os
import sys
import glob
import multiprocessing
import caliperreader as cr
from .Reader import Reader
from .CaliMapMaker import CaliMapMaker


def process_cali_file_batch(file_batch, inclusive_strings):
    """
    Optimized function to process a batch of cali files in a single worker.
    This reduces the overhead of creating many small tasks.
    """
    batch_results = {}

    for cali_file in file_batch:
        try:
            # Read file once
            r = cr.CaliperReader()
            r.read(cali_file)

            # Read globals once
            glob0 = cr.read_caliper_globals(cali_file)

            # Process all records for this file
            file_data = {}
            for rec in r.records:
                if "path" in rec:
                    path = rec["path"][-1]

                    if path not in file_data:
                        file_data[path] = {"name": path, "xaxis": [], "ydata": []}

                    file_data[path]["xaxis"].append(glob0)

                    # Extract metrics efficiently
                    metrics = {}
                    for i, metric_key in enumerate(["min", "max", "avg", "sum"]):
                        if inclusive_strings[i] in rec:
                            metrics[metric_key] = rec[inclusive_strings[i]]
                        else:
                            print(
                                f"Warning: {inclusive_strings[i]} not found in {cali_file}"
                            )
                            metrics[metric_key] = 0.0

                    file_data[path]["ydata"].append(metrics)

            # Merge file data into batch results
            for path, data in file_data.items():
                if path not in batch_results:
                    batch_results[path] = {
                        "name": data["name"],
                        "xaxis": [],
                        "ydata": [],
                    }

                batch_results[path]["xaxis"].extend(data["xaxis"])
                batch_results[path]["ydata"].extend(data["ydata"])

        except Exception as e:
            print(f"Error processing {cali_file}: {e}")
            continue

    return batch_results


class CaliReaderOptimized(Reader):

    def __init__(self, path="", pool_size=None, inclusive_strings=None):

        # Auto-detect optimal pool size
        if pool_size is None:
            pool_size = min(
                multiprocessing.cpu_count(), 20
            )  # Cap at 20 to avoid overhead

        self.pool_size = pool_size
        self.path = path
        self.mapMaker = CaliMapMaker()

        self.entireForest = {"nodes": {}}

        self.inclusive_strings = [
            "min#inclusive#sum#time.duration",
            "max#inclusive#sum#time.duration",
            "avg#inclusive#sum#time.duration",
            "sum#inclusive#sum#time.duration",
        ]

        if inclusive_strings is not None:
            self.inclusive_strings = inclusive_strings

        self.xy_idx_by_drill_level = {}
        self.init()

    def __iter__(self):
        self.keys = list(self.xy_idx_by_drill_level.keys())
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.keys):
            key = self.keys[self.index]
            self.index += 1
            return key, self.xy_idx_by_drill_level[key]
        else:
            raise StopIteration

    def get_cali_files(self, path):
        """Efficiently get all .cali files"""
        if not os.path.exists(path) or not os.path.isdir(path):
            print(f"Invalid or non-existent directory: {path}")
            sys.exit()

        # Use glob for better performance
        pattern = os.path.join(path, "**", "*.cali")
        profiles = glob.glob(pattern, recursive=True)

        if not profiles:
            print(f"No .cali files found in {path}")
            sys.exit()

        return profiles

    def create_optimal_batches(self, profiles, pool_size):
        """Create optimally sized batches for multiprocessing"""
        total_files = len(profiles)

        # Calculate optimal batch size
        # Aim for 2-4 batches per worker to balance load
        target_batches = pool_size * 3
        batch_size = max(1, total_files // target_batches)

        # Create batches
        batches = []
        for i in range(0, total_files, batch_size):
            batch = profiles[i : i + batch_size]
            batches.append(batch)

        print(
            f"Processing {total_files} files in {len(batches)} batches "
            f"(~{batch_size} files per batch) using {pool_size} workers"
        )

        return batches

    def combine_batch_results(self, batch_results_list):
        """Efficiently combine results from all batches"""
        combined = {}

        for batch_results in batch_results_list:
            for path, data in batch_results.items():
                if path not in combined:
                    combined[path] = {"name": data["name"], "xaxis": [], "ydata": []}

                combined[path]["xaxis"].extend(data["xaxis"])
                combined[path]["ydata"].extend(data["ydata"])

        return combined

    def init(self):
        """Optimized initialization"""

        if not self.path:
            print("Path must be defined.")
            sys.exit()

        print(f"Initializing CaliReader with {self.pool_size} workers...")

        # Get all cali files
        profiles = self.get_cali_files(self.path)
        print(f"Found {len(profiles)} .cali files")

        # Store first profile for metadata
        self.first_profile = profiles[0]

        # Read first profile for metadata
        self.r = cr.CaliperReader()
        self.r.read(self.first_profile)

        # Create optimal batches
        file_batches = self.create_optimal_batches(profiles, self.pool_size)

        # Process files in parallel with optimized batching
        print("Processing files in parallel...")
        with multiprocessing.Pool(self.pool_size) as pool:
            # Use partial to pass inclusive_strings to each worker
            from functools import partial

            worker_func = partial(
                process_cali_file_batch, inclusive_strings=self.inclusive_strings
            )

            batch_results = pool.map(worker_func, file_batches)

        print("Combining results...")
        # Combine all batch results
        self.xy_idx_by_drill_level = self.combine_batch_results(batch_results)

        # Get metadata
        self.meta_globals = self.get_meta_globals()

        print(
            f"Initialization complete. Processed {len(self.xy_idx_by_drill_level)} unique paths."
        )

    def get_meta_globals(self):
        """Get metadata globals from first profile"""
        glob0 = cr.read_caliper_globals(self.first_profile)
        meta_globals = {}

        for key in glob0:
            attr_type = self.r.attribute(key).get("adiak.type")
            meta_globals[key] = attr_type

        return meta_globals

    def make_child_map(self):
        """Build child map for tree structure"""
        for rec in self.r.records:
            if "path" in rec:
                if not isinstance(rec["path"], str):
                    self.mapMaker.make(rec["path"])

    def get_entire(self):
        """Get the entire processed dataset"""
        self.make_child_map()
        cm = self.mapMaker.getChildrenMap()

        return {
            "meta_globals": self.meta_globals,
            "nodes": self.xy_idx_by_drill_level,
            "childrenMap": cm,
        }
