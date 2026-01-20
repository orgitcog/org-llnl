# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import click

from sonar_driver.cli.kafka_connect.kafka_connect import kafka_connect
from sonar_driver.cli.cassandra.cassandra import cassandra


@click.group()
@click.option('--debug/--no-debug', default=False, help="Run in debug (verbose) mode")
@click.option('--dry/--no-dry', default=False, help="Run in dry mode (don't execute commands)")
@click.pass_context
def cli(ctx, debug, dry):
    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj['DEBUG'] = debug
    ctx.obj['DRY'] = dry


from sonar_driver.cli.service.service import service

cli.add_command(service)
cli.add_command(kafka_connect)
cli.add_command(cassandra)
