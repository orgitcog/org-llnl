# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import click


@click.group()
@click.pass_context
def service(ctx):
    pass


from sonar_driver.cli.service.cassandra import cassandra
from sonar_driver.cli.service.kafka_connect import kafka_connect

service.add_command(cassandra)
service.add_command(kafka_connect)
