# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import click
import os

from sonar_driver.print_utils import pretty_print

@click.group()
@click.option('--cassandra-password-file', default=None, help='File containing password for Cassandra connection')
@click.option('--cassandra-hosts', default=os.environ.get('CQLSH_HOST', 'localhost'), help='Cassandra hosts (comma-separated, no spaces) to connect to (default: value of CQLSH_PORT or 9042)')
@click.option('--cassandra-port', default=os.environ.get('CQLSH_PORT', '9042'), help='Cassandra port to connect to (default: value of CQLSH_PORT or 9042)')
@click.pass_context
def cassandra(ctx,
              cassandra_password_file,
              cassandra_hosts,
              cassandra_port):

    from sonar_driver.cassandra.session import SonarCassandraSession

    ctx.obj['CassandraSession'] = SonarCassandraSession(
        password_filename=cassandra_password_file,
        hosts=cassandra_hosts.split(','),
        port=cassandra_port,
        dry=ctx.obj['DRY'],
        debug=ctx.obj['DEBUG']
    )

    pretty_print(ctx.obj['CassandraSession'].__dict__, title='Cassandra Session')


@click.command()
@click.pass_context
def auth(ctx):
    cassandra_session = ctx.obj['CassandraSession']
    print(cassandra_session.token)


@click.command()
@click.argument('keyspace')
@click.argument('table')
@click.argument('avro_schema_file')
@click.argument('partition_key')
@click.option('--cluster-key', help='One or more cluster keys, comma-separated, no spaces')
@click.pass_context
def create_table(ctx,
                 keyspace,
                 table,
                 avro_schema_file,
                 partition_key,
                 cluster_key):

    import avro.schema

    cassandra_session = ctx.obj['CassandraSession']

    with open(avro_schema_file, 'r') as f:
        avro_schema = avro.schema.Parse(f.read())
    if ctx.obj['DEBUG']:
        pretty_print(avro_schema.to_json(), title="Avro schema")

    if not ctx.obj['DRY'] and cassandra_session.table_exists(keyspace, table):
        raise Exception("Table {}.{} already exists!".format(keyspace, table))
    else:
        cassandra_session.create_table_from_avro_schema(
            keyspace,
            table,
            avro_schema,
            partition_key,
            cluster_key
        )


cassandra.add_command(auth)
cassandra.add_command(create_table)
