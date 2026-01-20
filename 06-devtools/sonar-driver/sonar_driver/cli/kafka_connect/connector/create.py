# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import click
import os

from sonar_driver.print_utils import pretty_print


@click.group()
@click.pass_context
def create(ctx):
    pass


@click.command()
@click.argument('topic_name')
@click.argument('ingest_dir')
@click.argument('completed_dir')
@click.argument('avro_schema_file')
@click.option('--tasks-max', default=1, help='Maximum number of concurrent ingestion tasks')
@click.option('--file-format', default='csv', help='File format (csv or json)')
@click.option('--format-options', default='{}', help='File format additional options')
@click.option('--batch-rows', default=500, help='Number of rows to read and ingest at a time')
@click.option('--batch-files', default=10, help='Number of files to read and ingest at a time')
@click.option('--zookeeper-host', default=os.environ.get('ZOOKEEPER_HOST', 'localhost'), help='Zookeeper host to connect to (default: value of ZOOKEEPER_PORT or 2181)')
@click.option('--zookeeper-port', default=os.environ.get('ZOOKEEPER_PORT', '2181'), help='Zookeeper port to connect to (default: value of ZOOKEEPER_PORT or 2181)')
@click.pass_context
def sonar_directory_source(ctx,
                           topic_name,
                           ingest_dir,
                           completed_dir,
                           avro_schema_file,
                           tasks_max,
                           file_format,
                           format_options,
                           batch_rows,
                           batch_files,
                           zookeeper_host,
                           zookeeper_port):

    import avro.schema

    from sonar_driver.kafka_connect.connector import Connector
    from sonar_driver.kafka_connect.sonar_directory_source_config import SonarDirectorySourceConfig

    with open(avro_schema_file, 'r') as f:
        avro_schema = avro.schema.Parse(f.read())
    if ctx.obj['DEBUG']:
        pretty_print(avro_schema.to_json(), title="Avro schema")

    connector = Connector(
        "sonar_directory_source-" + topic_name,
        SonarDirectorySourceConfig(
            topic_name,
            ingest_dir,
            completed_dir,
            avro_schema,
            tasks_max,
            file_format,
            format_options,
            batch_rows,
            batch_files,
            zookeeper_host,
            zookeeper_port
        )
    )

    if ctx.obj['DEBUG']:
        pretty_print(connector.json(), title="Connector")

    pretty_print(connector.json(), colorize=False)


@click.command()
@click.argument('topic_name')
@click.argument('keyspace')
@click.argument('table')
@click.argument('cassandra_username')
@click.argument('cassandra_password_file')
@click.option('--tasks-max', default=1, help='Maximum number of concurrent ingestion tasks')
@click.option('--cassandra-hosts', default=os.environ.get('CQLSH_HOST', 'localhost'), help='Cassandra hosts (comma-separated, no spaces) to connect to (default: value of CQLSH_PORT or 9042)')
@click.option('--cassandra-port', default=os.environ.get('CQLSH_PORT', '9042'), help='Cassandra port to connect to (default: value of CQLSH_PORT or 9042)')
@click.pass_context
def cassandra_sink(ctx,
                   topic_name,
                   keyspace,
                   table,
                   cassandra_username,
                   cassandra_password_file,
                   tasks_max,
                   cassandra_hosts,
                   cassandra_port):

    from sonar_driver.kafka_connect.connector import Connector
    from sonar_driver.kafka_connect.cassandra_sink_config import CassandraSinkConfig

    connector = Connector(
        "cassandra_sink-" + topic_name,
        CassandraSinkConfig(
            topic_name,
            keyspace,
            table,
            cassandra_username,
            cassandra_password_file,
            cassandra_hosts.split(','),
            cassandra_port,
            None,
            tasks_max,
            False # TODO: when delete sunk works, enable this
        )
    )

    if ctx.obj['DEBUG']:
        pretty_print(connector.json(), title="Connector")

    pretty_print(connector.json(), colorize=False)


create.add_command(sonar_directory_source)
create.add_command(cassandra_sink)
