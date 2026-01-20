#! /bin/bash
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

# Simple script that copies from first argument to second argument.
# Used in test.ats.

if [ $# -ne 2 ] ; then
    echo "Usage ./copy.sh {FROM} {TO}"
    exit 1
fi

FROM=$1
TO=$2

cp -rf $FROM $TO
