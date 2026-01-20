# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import os
import pytest

from hatchet import GraphFrame
from thicket import Thicket


def test_empty_iterable():
    with pytest.raises(ValueError, match="Iterable must contain at least one file"):
        Thicket.reader_dispatch(
            GraphFrame.from_caliperreader,
            False,
            True,
            True,
            [],
        )

    with pytest.raises(ValueError, match="Iterable must contain at least one file"):
        Thicket.reader_dispatch(
            GraphFrame.from_caliperreader,
            False,
            True,
            True,
            tuple([]),
        )


def test_file_not_found():
    with pytest.raises(ValueError, match="Path 'blah' not found"):
        Thicket.reader_dispatch(
            GraphFrame.from_caliperreader,
            False,
            True,
            True,
            "blah",
        )

    with pytest.raises(FileNotFoundError, match="File 'blah' not found"):
        Thicket.reader_dispatch(
            GraphFrame.from_caliperreader,
            False,
            True,
            True,
            ["blah"],
        )


def test_valid_type():
    with pytest.raises(TypeError, match="'int' is not a valid type to be read from"):
        Thicket.reader_dispatch(
            GraphFrame.from_caliperreader,
            False,
            True,
            True,
            -1,
        )


def test_valid_inputs(rajaperf_cali_1trial, data_dir):

    # Works with list
    Thicket.reader_dispatch(
        GraphFrame.from_caliperreader,
        False,
        True,
        True,
        rajaperf_cali_1trial,
    )

    # Works with single file
    Thicket.reader_dispatch(
        GraphFrame.from_caliperreader,
        False,
        True,
        True,
        rajaperf_cali_1trial[0],
    )

    # Works with directory
    Thicket.reader_dispatch(
        GraphFrame.from_caliperreader,
        False,
        True,
        True,
        f"{data_dir}/rajaperf/lassen/clang10.0.1_nvcc10.2.89_1048576/1/",
    )


def test_error_file(mpi_scaling_cali, data_dir):

    # Create a temporarily empty file
    empty_file_path = os.path.join(f"{data_dir}/mpi_scaling_cali", "empty.cali")
    with open(empty_file_path, "w"):
        pass  # This creates an empty file

    # list
    with pytest.raises(Exception, match="Failed to read file"):
        Thicket.reader_dispatch(
            GraphFrame.from_caliperreader,
            False,
            True,
            True,
            mpi_scaling_cali + [empty_file_path],
        )

    # directory
    with pytest.raises(Exception, match="Failed to read file"):
        Thicket.reader_dispatch(
            GraphFrame.from_caliperreader,
            False,
            True,
            True,
            f"{data_dir}/mpi_scaling_cali/",
        )

    # Remove the file
    os.remove(empty_file_path)
