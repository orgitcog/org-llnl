# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod


class Reader(ABC):
    @abstractmethod
    def get_entire(self, xaxis_name):
        pass
