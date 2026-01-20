# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pytest


def test_get_node(literal_thickets):
    tk, _, _ = literal_thickets

    # Check error raised
    with pytest.raises(KeyError):
        tk.get_node("Foo")

    # Check case which="first"
    qux1 = tk.get_node("Qux", which="first")
    assert qux1.frame["name"] == "Qux"
    assert qux1._hatchet_nid == 1

    # Check case which="last"
    qux2 = tk.get_node("Qux", which="last")
    assert qux2.frame["name"] == "Qux"
    assert qux2._hatchet_nid == 2

    # Check case which="all"
    qux_all = tk.get_node("Qux", which="all")
    assert len(qux_all) == 2
