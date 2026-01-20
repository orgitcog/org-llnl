#!/usr/bin/env python
#
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Huy Le <le35@llnl.gov>
# LLNL-CODE-734258
#
# All rights reserved.
# This file is part of STK Address Converter. For details, see
# https://github.com/LLNL/STKAddressConverter. Licensed under the
# Apache License, Version 2.0 (the “Licensee”); you may not use
# this file except in compliance with the License. You may
# obtain a copy of the License at:
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the license.
##
##
# Unit testing that tests the address translation of acs2internal.
#

from unittest import TestCase, main
from os import path as os_path
from sys import path as sys_path

# Find the directory where unit_test.py is.
TESTS_DIR = os_path.dirname(os_path.realpath(__file__))
sys_path.insert(0, TESTS_DIR + "/../")

from acs2internal import acs2internal

class UnitTests(TestCase):
    """Unit tests for acs2internal.py.
    """

    def test_run_command(self):
        """Utility function that runs arbitrary commands.
        """
        assert acs2internal.run_command("ls", 1, "ls")

    def test_acsls_to_internal(self):
        """Equivalent to `python acs2internal.py -d 1,10,1,4`
           ACSLS address to internal address.
        """
        self.assertEqual("3,3,-1,1,1",
                         acs2internal.acsls_addr_to_internal_addr( \
                         acs_address="1,10,1,4"))

    def test_internal_to_acsls(self):
        """Equivalent to `python acs2internal.py -i 3,3,-1,1,1`
           Internal address to acsls address.
        """
        self.assertEqual("1,10,1,4",
                         acs2internal.internal_addr_to_acsls_addr( \
                         internal_address="3,3,-1,1,1"))

if __name__ == "__main__":
    main()
