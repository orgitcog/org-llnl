# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""
Printing Utilities
"""

from __future__ import print_function
import json
import sys

from pygments import highlight, lexers, formatters


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def pretty_print(data, title=None, lexer=lexers.JsonLexer(), colorize=True):
    """ Format and print pretty output using pygments """

    toprint = data

    if lexer.name == 'JSON':
        toprint = json.dumps(data, sort_keys=True, indent=4)

    if colorize:
        toprint = highlight(toprint, lexer, formatters.TerminalFormatter())

    if title is not None:
        print(title + ':')

    print(toprint)

