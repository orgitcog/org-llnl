# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import click

from sonar_driver.print_utils import pretty_print


@click.group()
@click.pass_context
def schema(ctx):
    pass

@click.command()
@click.pass_context
def list(ctx):
    pretty_print(ctx.obj['KafkaConnectSession'].get_schemas().json())


schema.add_command(list)
