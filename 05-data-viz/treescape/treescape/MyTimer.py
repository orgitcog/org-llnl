# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import time


class MyTimer:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.marks = [("Start", self.start_time)]

    def mark(self, label):
        now = time.perf_counter()
        self.marks.append((label, now))

    def print(self):
        results = []
        prev_label, prev_time = self.marks[0]
        for label, current_time in self.marks[1:]:
            elapsed = current_time - prev_time
            total = current_time - self.start_time
            results.append((prev_label, label, f"{elapsed:.6f}", f"{total:.6f}"))
            prev_label, prev_time = label, current_time

        # Print header
        headers = ("From", "To", "Interval (s)", "Total (s)")
        col_widths = [
            max(len(str(row[i])) for row in ([headers] + results)) for i in range(4)
        ]
        row_format = " | ".join(f"{{:<{w}}}" for w in col_widths)

        print(row_format.format(*headers))
        print("-+-".join("-" * w for w in col_widths))
        for row in results:
            print(row_format.format(*row))
