# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import click
import json

from sonar_driver.print_utils import pretty_print


@click.group()
@click.pass_context
def connector(ctx):
    pass


def get_connector_args(ctx, connector_names, a):
    """
    Utility to either get a list of provided connectors or all of them if -a is provided
    Used for all connector-specific commands (install, uninstall, status, config ...)
    """
    if a:
        connector_names = ctx.obj['KafkaConnectSession'].get_connectors().json()
    if len(connector_names) == 0:
        raise ValueError("No connectors specified!")
    ctx.obj['KafkaConnectors'] = connector_names


@click.command()
@click.pass_context
def list(ctx):
    pretty_print(ctx.obj['KafkaConnectSession'].get_connectors().json())


@click.command()
@click.argument('connector_files', nargs=-1, type=click.File('r'))
@click.pass_context
def install(ctx, connector_files):
    for connector_file in connector_files:
        connector_json = json.load(connector_file)
        ctx.obj['KafkaConnectSession'].install_connector(connector_json)


@click.command()
@click.argument('connector_names', nargs=-1)
@click.option('-a', is_flag=True)
@click.pass_context
def uninstall(ctx, connector_names, a):
    get_connector_args(ctx, connector_names, a)
    for connector_name in ctx.obj['KafkaConnectors']:
        ctx.obj['KafkaConnectSession'].uninstall_connector(connector_name)


@click.command()
@click.argument('connector_names', nargs=-1)
@click.option('-a', is_flag=True)
@click.pass_context
def status(ctx, connector_names, a):
    statuses = []
    get_connector_args(ctx, connector_names, a)
    for connector_name in ctx.obj['KafkaConnectors']:
        ret = ctx.obj['KafkaConnectSession'].get_connector_status(connector_name)
        if not ctx.obj['DRY']:
            statuses.append(ret.json())
    pretty_print(statuses)


@click.command()
@click.argument('connector_name', default=None)
@click.option('-a', is_flag=True)
@click.argument('config_cmds', nargs=-1)
@click.pass_context
def config(ctx, connector_name, a, config_cmds):
    if a:
        ctx.obj['KafkaConnectors'] = ctx.obj['KafkaConnectSession'].get_connectors().json()
    else:
        ctx.obj['KafkaConnectors'] = [connector_name]

    set_vars = [s.split('=') for s in config_cmds]
    if not all([len(s) == 2 for s in set_vars]):
        raise ValueError("Invalid config command(s), must be key=value")

    for connector_name in ctx.obj['KafkaConnectors']:
        c = ctx.obj['KafkaConnectSession'].get_connector_config(connector_name)
        if not ctx.obj['DRY']:
            c = c.json()
            if len(set_vars) > 0:
                for v in set_vars:
                    c[v[0]] = v[1]
                ctx.obj['KafkaConnectSession'].set_connector_config(connector_name, c)
            else:
                pretty_print(c)


from sonar_driver.cli.kafka_connect.connector.create import create

connector.add_command(create)
connector.add_command(list)
connector.add_command(install)
connector.add_command(uninstall)
connector.add_command(status)
connector.add_command(config)


