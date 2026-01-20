# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import click
import os


@click.group()
@click.option('--kafka-rest-url', default=os.environ.get('KAFKA_REST_URL', 'localhost'), help='Kafka REST URL to connect to (default: value of KAFKA_REST_URL or localhost)')
@click.option('--kafka-connect-port', default=os.environ.get('KAFKA_CONNECT_PORT', '8083'), help='Kafka REST URL to connect to (default: value of KAFKA_CONNECT_PORT or 8083)')
@click.option('--kafka-schema-registry-port', default=os.environ.get('KAFKA_SCHEMA_REGISTRY_PORT', '8081'), help='Kafka REST URL to connect to (default: value of KAFKA_SCHEMA_REGISTRY_PORT or 8081)')
@click.pass_context
def kafka_connect(ctx, kafka_rest_url, kafka_connect_port, kafka_schema_registry_port):
    from sonar_driver.kafka_connect.session import KafkaConnectSession

    ctx.obj['KafkaConnectSession'] = KafkaConnectSession(
        kafka_rest_url,
        kafka_connect_port,
        kafka_schema_registry_port,
        ctx.obj['DEBUG'],
        ctx.obj['DRY']
    )


from sonar_driver.cli.kafka_connect.connector.connector import connector
from sonar_driver.cli.kafka_connect.schema.schema import schema


kafka_connect.add_command(connector)
kafka_connect.add_command(schema)
