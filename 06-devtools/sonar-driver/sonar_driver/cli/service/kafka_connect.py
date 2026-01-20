# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import click

import os
import signal
import subprocess


@click.group()
@click.option('--zookeeper-properties-file', default='/etc/kafka/zookeeper.properties')
@click.option('--server-properties-file', default='/etc/kafka/server.properties')
@click.option('--rest-properties-file', default='/etc/kafka-rest/kafka-rest.properties')
@click.option('--schema-registry-properties-file', default='/etc/schema-registry/schema-registry.properties')
@click.option('--connect-properties-file', default='/etc/schema-registry/connect-avro-distributed.properties')
@click.option('--zookeeper-log-file', default='/var/log/kafka/zookeeper.log')
@click.option('--kafka-log-file', default='/var/log/kafka/kafka.log')
@click.option('--rest-log-file', default='/var/log/kafka/rest.log')
@click.option('--schema-registry-log-file', default='/var/log/kafka/schema_registry.log')
@click.option('--connect-log-file', default='/var/log/kafka/connect.log')
@click.pass_context
def kafka_connect(ctx,
                  zookeeper_properties_file,
                  server_properties_file,
                  rest_properties_file,
                  schema_registry_properties_file,
                  connect_properties_file,

                  zookeeper_log_file,
                  kafka_log_file,
                  rest_log_file,
                  schema_registry_log_file,
                  connect_log_file):

    ctx.obj['zookeeper_properties_file'] = zookeeper_properties_file
    ctx.obj['server_properties_file'] = server_properties_file
    ctx.obj['rest_properties_file'] = rest_properties_file
    ctx.obj['schema_registry_properties_file'] = schema_registry_properties_file
    ctx.obj['connect_properties_file'] = connect_properties_file

    ctx.obj['zookeeper_log_file'] = zookeeper_log_file
    ctx.obj['kafka_log_file'] = kafka_log_file
    ctx.obj['rest_log_file'] = rest_log_file
    ctx.obj['schema_registry_log_file'] = schema_registry_log_file
    ctx.obj['connect_log_file'] = connect_log_file


def start_service(cmd, logfile, *args):
    with open(logfile, 'w') as logfile_out:
        with open(cmd + '.pid', 'w') as pidfile_out:
            proc = subprocess.Popen([cmd] + list(args), stdout=logfile_out, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
            pidfile_out.write(str(proc.pid))
            subprocess.Popen(['ps', 'aux', str(proc.pid)]).wait()


def stop_service(cmd):
    with open(cmd + '.pid', 'r') as pidfile_out:
        pid = int(pidfile_out.read())

        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            print("{} not active on pid {}".format(cmd, pid))


def status_service(cmd):
    with open(cmd + '.pid', 'r') as pidfile_out:
        pid = int(pidfile_out.read())

        try:
            os.kill(pid, 0)
            return [str(pid), cmd, '[\033[92m', 'UP', '\033[0m]']
        except ProcessLookupError:
            return [str(pid), cmd, '[\033[91m', 'DOWN', '\033[0m]']


@click.command()
@click.pass_context
def start(ctx):
    start_service('zookeeper-server-start', ctx.obj['zookeeper_log_file'], ctx.obj['zookeeper_properties_file'])
    start_service('kafka-server-start', ctx.obj['kafka_log_file'], ctx.obj['server_properties_file'])
    start_service('kafka-rest-start', ctx.obj['rest_log_file'], ctx.obj['rest_properties_file'])
    start_service('schema-registry-start', ctx.obj['schema_registry_log_file'], ctx.obj['schema_registry_properties_file'])
    start_service('connect-distributed', ctx.obj['connect_log_file'], ctx.obj['connect_properties_file'])


@click.command()
@click.pass_context
def stop(ctx):
    stop_service('connect-distributed')
    stop_service('schema-registry-start')
    stop_service('kafka-rest-start')
    stop_service('kafka-server-start')
    stop_service('zookeeper-server-start')


@click.command()
@click.pass_context
def status(ctx):
    statuses = [
        status_service('connect-distributed'),
        status_service('schema-registry-start'),
        status_service('kafka-rest-start'),
        status_service('kafka-server-start'),
        status_service('zookeeper-server-start')
    ]

    widths = [max(map(len, col)) for col in zip(*statuses)]
    for row in statuses:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))


kafka_connect.add_command(start)
kafka_connect.add_command(stop)
kafka_connect.add_command(status)
