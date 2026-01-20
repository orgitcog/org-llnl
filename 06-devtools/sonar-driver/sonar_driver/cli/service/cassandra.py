# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import click
import os


@click.group()
@click.option('--cassandra-password-file', default=None, help='File containing password for Cassandra connection')
@click.option('--cassandra-hosts', default=os.environ.get('CQLSH_HOST', 'localhost'), help='Cassandra hosts (comma-separated, no spaces) to connect to (default: value of CQLSH_PORT or 9042)')
@click.option('--cassandra-port', default=os.environ.get('CQLSH_PORT', '9042'), help='Cassandra port to connect to (default: value of CQLSH_PORT or 9042)')
@click.pass_context
def cassandra(ctx):
    pass

# TODO: this
