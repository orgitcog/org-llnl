# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from pygments import lexers
from IPython.display import clear_output

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import NoHostAvailable
from cassandra import AuthenticationFailed

import json
import getpass

from sonar_driver.print_utils import pretty_print, eprint


class SonarCassandraSession():
    def __init__(self, password_filename=None, password=None, tokencache_configfile=None, hosts=['localhost'], port=9042, dry=False, debug=False):

        self.dry = dry
        self.debug = debug
        self.username = getpass.getuser()
        self.hosts = hosts
        self.hosts_string = ','.join(hosts)

        # Check if a password is provided
        password_args = sum(1 for p in [password_filename, password] if p is not None)
        if password_args > 1:
            raise Exception("Multiple password arguments were supplied! Failing due to ambiguity.")

        if not self.dry:

            # Use a password arg
            if password_args == 1:

                # Password file
                if password_filename is not None:
                    with open(password_filename, 'r') as password_file:
                        self.password = password_file.read().replace('\n', '')

                # Password provided directly
                elif password is not None:
                    self.password = password

            # Interactively authenticate
            else:
                self.init_interactive(tokencache_configfile)

            # Create auth provider and connect (even if init_interactive was already run, in case we have a token now)
            auth_provider = PlainTextAuthProvider(username=self.username, password=self.password)
            cluster = Cluster(self.hosts, port=port, auth_provider=auth_provider)
            self.session = cluster.connect()

    def init_interactive(self, tokencache_configfile=None):

        # 3 retries
        for x in range(3):

            self.password = getpass.getpass("Enter password for user {}:".format(self.username))
            clear_output()

            auth_provider = PlainTextAuthProvider(username=self.username, password=self.password)
            cluster = Cluster(self.hosts, auth_provider=auth_provider)

            try:
                self.session = cluster.connect()

                if tokencache_configfile is not None:
                    self.password = self.get_token_from_tokencache(tokencache_configfile)

                return

            except NoHostAvailable as e:
                if isinstance(list(e.errors.values())[0], AuthenticationFailed):
                    eprint("Authentication failure")

        raise AuthenticationFailed("Failed 3 times, aborting!")

    def get_token_from_tokencache(self, tokencache_configfile):
        """ Reads a JSON file describing how to obtain a token from a token cache in Cassandra """

        with open(tokencache_configfile) as f:
            tokencache_config = json.loads(f.read())

            tokencache_query = self.session.prepare("SELECT ? FROM ? WHERE ?", keyspace=tokencache_config['keyspace'])
            tokencache_boundquery = tokencache_query.bind(tokencache_config['value'], tokencache_config['table'], tokencache_config['whereclause'])
            results = self.session.execute(tokencache_boundquery, timeout=30)

            return results[0].value

    def table_exists(self, keyspace, table):

        exists_query = "SELECT table_name FROM system_schema.tables WHERE keyspace_name='{}' AND table_name='{}'".format(keyspace, table)

        if self.debug:
            pretty_print(exists_query, title="Check for Cassandra table CQL", lexer=lexers.SqlLexer())

        if not self.dry:
            try:
                results = self.session.execute(exists_query)
            except AuthenticationFailed:
                raise Exception("Cassandra user '{}' unauthorized to view system_schema.tables on hosts '{}'!".format(self.username, self.hosts))

            if self.debug:
                pretty_print(results.current_rows, title="Query results")

            if results.current_rows:
                return True
            else:
                return False
        else:
            return True

    @staticmethod
    def avro2cass(avro_dtype):
        AVRO_CASSANDRA_TYPEMAP = {
            "string" : "text",
            "long" : "bigint"
        }

        if isinstance(avro_dtype, dict):
            if avro_dtype['type'] == 'array':
                return "list<" + SonarCassandraSession.avro2cass(avro_dtype['values']) + ">"
            elif avro_dtype['type'] == 'map':
                return "map<text," + SonarCassandraSession.avro2cass(avro_dtype['values']) + ">"

        return AVRO_CASSANDRA_TYPEMAP[avro_dtype] if avro_dtype in AVRO_CASSANDRA_TYPEMAP else avro_dtype

    @staticmethod
    def primary_key(partition_key, cluster_key):
        partition_key_parts = partition_key.split(',')
        partition_key_quoted = ','.join(map(lambda s: "\"" + s + "\"", partition_key_parts))

        if cluster_key:
            cluster_key_parts = cluster_key.split(',')
            cluster_key_quoted = ','.join(map(lambda s: "\"" + s + "\"", cluster_key_parts))
            return "(({}),{})".format(partition_key_quoted, cluster_key_quoted)

        return "(({}))".format(partition_key_quoted)

    def create_table_from_avro_schema(self, keyspace, table, avro_schema, partition_key, cluster_key):

        avro_json = avro_schema.to_json()
        columns_clause = ', '.join(map(lambda f: "\"" + f['name'] + "\"" + ' ' + SonarCassandraSession.avro2cass(f['type']), avro_json['fields']))
        primary_key_clause = SonarCassandraSession.primary_key(partition_key, cluster_key)

        create_query = "CREATE TABLE {}.{} ({}, PRIMARY KEY {})".format(keyspace, table, columns_clause, primary_key_clause)

        if self.debug or self.dry:
            pretty_print(create_query, title="Create table CQL", lexer=lexers.SqlLexer())
        if not self.dry:
            self.session.execute(create_query, timeout=None) 
